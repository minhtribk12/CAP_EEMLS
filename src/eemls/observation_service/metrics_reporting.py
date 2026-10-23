from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

import httpx

from eemls.domain.measuring import Measuring
from eemls.domain.records import (
    Decision,
    Feedback,
    InferenceServiceProfile,
    Instance,
    RunOwner,
    ServedQuery,
)
from eemls.domain.types import FrozenModel, MemberPrediction, Row, Timestamp
from eemls.metrics_database.buffered_writer import Batch, BufferedWriter, WriteError
from eemls.metrics_database.database import MetricsDatabase
from eemls.observation_service.observation_service_api import ObservationServiceError

logger = logging.getLogger(__name__)

SENT_TABLES: tuple[str, ...] = ("queries", "member_predictions", "feedback", "decisions")


class ClosableDatabase(MetricsDatabase, Protocol):
    def close(self) -> None: ...


class MetricsReporter:
    def __init__(
        self,
        *,
        base_url: str,
        run_id: str,
        sender: str,
        monitoring: Measuring,
        buffer_rows: int,
        flush_interval_ms: float,
        request_timeout_ms: float,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run_id = run_id
        self._sender = sender
        self._buffer_rows = buffer_rows
        self._client = httpx.Client(
            base_url=base_url, timeout=request_timeout_ms / 1000.0, transport=transport
        )
        self._writer = BufferedWriter(
            self._sent,
            monitoring=monitoring,
            buffer_rows=buffer_rows,
            flush_interval_ms=flush_interval_ms,
            clock=clock,
        )

    def add(self, table: str, record: FrozenModel) -> None:
        if table not in SENT_TABLES:
            raise ValueError(f"{table} is not a table the Observation Service takes rows for")
        try:
            self._writer.add(table, record.model_dump(mode="json"))
        except WriteError as refused:
            logger.warning(
                f"stage_tag=store process_tag=control_plane error_type=rows_not_sent "
                f"rows={self._writer.depth} recovery_action=rows_held_for_the_next_send "
                f"detail={refused}"
            )

    def flush(self) -> None:
        self._writer.flush()

    def flush_quietly(self) -> None:
        self._writer.flush_quietly()

    def close(self) -> None:
        try:
            self._writer.close_quietly()
        finally:
            self._client.close()

    def _sent(self, batch: Batch) -> None:
        dropped = 0
        try:
            for piece in _pieces(batch, self._buffer_rows):
                body = {"run_id": self._run_id, "sender": self._sender, **piece}
                answer = self._client.post("/rows", json=body)
                if answer.status_code == httpx.codes.CONFLICT:
                    dropped += sum(len(rows) for rows in piece.values())
                elif answer.status_code != httpx.codes.OK:
                    raise ObservationServiceError(
                        f"the Observation Service answered {answer.status_code}: {answer.text}"
                    )
        finally:
            if dropped:
                logger.error(
                    f"stage_tag=store process_tag=control_plane error_type=wrong_run "
                    f"run_id={self._run_id} rows={dropped} recovery_action=dropped"
                )


def _pieces(batch: Batch, size: int) -> Iterator[dict[str, list[Row]]]:
    held = [(table, row) for table, rows in batch.items() for row in rows]
    for start in range(0, len(held), size):
        piece: dict[str, list[Row]] = {}
        for table, row in held[start : start + size]:
            piece.setdefault(table, []).append(row)
        yield piece


class ReportingDatabase:
    def __init__(self, database: ClosableDatabase, reporter: MetricsReporter) -> None:
        self._database = database
        self._reporter = reporter

    def apply_schema(self) -> None:
        self._database.apply_schema()

    def claim_run(self, owner: RunOwner) -> str:
        return self._database.claim_run(owner)

    def register_instance(self, instance: Instance) -> None:
        self._database.register_instance(instance)

    def write_instance_stopped(self, instance_id: str, stopped_at: Timestamp) -> None:
        self._database.write_instance_stopped(instance_id, stopped_at)

    def write_query(self, query: ServedQuery) -> None:
        self._reporter.add("queries", query)

    def write_member_predictions(self, predictions: Sequence[MemberPrediction]) -> None:
        for one in predictions:
            self._reporter.add("member_predictions", one)

    def write_feedback(self, feedback: Feedback) -> None:
        self._reporter.add("feedback", feedback)

    def write_decision(self, decision: Decision) -> None:
        self._reporter.add("decisions", decision)

    def write_profile(self, profile: InferenceServiceProfile) -> None:
        self._database.write_profile(profile)

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        self._flush_for_a_reader()
        return self._database.read_window(consumer_id, limit)

    def read_member_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        self._flush_for_a_reader()
        return self._database.read_member_window(consumer_id, limit)

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self._flush_for_a_reader()
        return self._database.read_member_answers(since=since, until=until)

    def read_served(self, *, consumer_id: str, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self._flush_for_a_reader()
        return self._database.read_served(consumer_id=consumer_id, since=since, until=until)

    def read_rows(self, table: str) -> Sequence[Row]:
        self._flush_for_a_reader()
        return self._database.read_rows(table)

    def read_decisions(self, limit: int) -> Sequence[Decision]:
        self._flush_for_a_reader()
        return self._database.read_decisions(limit)

    def read_profiles(self) -> Sequence[InferenceServiceProfile]:
        self._flush_for_a_reader()
        return self._database.read_profiles()

    def flush(self) -> None:
        self._reporter.flush()
        self._database.flush()

    def close(self) -> None:
        try:
            self._reporter.close()
        finally:
            self._database.close()

    def _flush_for_a_reader(self) -> None:
        self._reporter.flush_quietly()
