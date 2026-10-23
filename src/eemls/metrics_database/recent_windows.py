from __future__ import annotations

import bisect
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime

from eemls.domain.records import (
    Decision,
    Feedback,
    InferenceServiceProfile,
    Instance,
    RunOwner,
    ServedQuery,
)
from eemls.domain.types import MemberPrediction, Row, Timestamp
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.schema import FROM_FEEDBACK, FROM_THE_SLOWEST_MEMBER, table_row

Order = tuple[datetime, str]


class RecentWindows:
    def __init__(self, database: MetricsDatabase, *, keep: int) -> None:
        self._database = database
        self._keep = keep
        self._lock = threading.Lock()
        self._order: dict[str, list[Order]] = {}
        self._queries: dict[str, dict[str, object]] = {}
        self._members: dict[str, list[dict[str, object]]] = {}
        self._consumer_of: dict[str, str] = {}
        self._labels: dict[str, tuple[str, dict[str, object]]] = {}

    def seed(self, consumer_ids: Iterable[str]) -> None:
        for consumer_id in consumer_ids:
            queries = self._database.read_window(consumer_id, self._keep)
            members = self._database.read_member_window(consumer_id, self._keep)
            with self._lock:
                for row in queries:
                    self._keep_query(consumer_id, dict(row))
                    if row["true_label"] is not None:
                        self._labels[str(row["query_id"])] = (
                            consumer_id,
                            {name: row[name] for name in FROM_FEEDBACK},
                        )
                for row in members:
                    self._members.setdefault(str(row["query_id"]), []).append(dict(row))

    def write_query(self, query: ServedQuery) -> None:
        self._database.write_query(query)
        row = table_row("queries", query)
        row.update(dict.fromkeys(FROM_FEEDBACK))
        with self._lock:
            self._keep_query(query.consumer_id, row)
            self._slowest(query.query_id)
            self._labeled(query.query_id)

    def write_member_predictions(self, predictions: Sequence[MemberPrediction]) -> None:
        self._database.write_member_predictions(predictions)
        with self._lock:
            for one in predictions:
                answers = self._members.setdefault(one.query_id, [])
                key = (one.member.service_name, one.in_ensemble)
                if any((row["service_name"], row["in_ensemble"]) == key for row in answers):
                    continue
                row = table_row("member_predictions", one)
                label = self._labels.get(one.query_id)
                row["true_label"] = label[1]["true_label"] if label is not None else None
                answers.append(row)
            for query_id in {one.query_id for one in predictions}:
                self._slowest(query_id)

    def write_feedback(self, feedback: Feedback) -> None:
        self._database.write_feedback(feedback)
        returned = table_row("feedback", feedback)
        fields = {name: returned[name] for name in FROM_FEEDBACK}
        with self._lock:
            if feedback.query_id in self._labels:
                return
            self._labels[feedback.query_id] = (feedback.consumer_id, fields)
            self._labeled(feedback.query_id)
            waiting = [key for key in self._labels if key not in self._queries]
            for forgotten in waiting[: max(0, len(waiting) - self._keep)]:
                del self._labels[forgotten]

    def _labeled(self, query_id: str) -> None:
        label = self._labels.get(query_id)
        held = self._queries.get(query_id)
        if label is None or held is None:
            return
        consumer_id, fields = label
        if self._consumer_of[query_id] == consumer_id:
            held.update(fields)
        for row in self._members.get(query_id, ()):
            row["true_label"] = fields["true_label"]

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        if limit > self._keep:
            return self._database.read_window(consumer_id, limit)
        with self._lock:
            return [dict(self._queries[query_id]) for query_id in self._newest(consumer_id, limit)]

    def read_member_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        if limit > self._keep:
            return self._database.read_member_window(consumer_id, limit)
        with self._lock:
            return [
                dict(row)
                for query_id in self._newest(consumer_id, limit)
                for row in sorted(
                    self._members.get(query_id, ()),
                    key=lambda one: (str(one["service_name"]), not one["in_ensemble"]),
                )
            ]

    def _keep_query(self, consumer_id: str, row: dict[str, object]) -> None:
        query_id = str(row["query_id"])
        order = self._order.setdefault(consumer_id, [])
        key = (_moment(row["t_arrive"]), query_id)
        if query_id in self._queries:
            return
        if len(order) >= self._keep and key < order[0]:
            return
        bisect.insort(order, key)
        self._queries[query_id] = row
        self._consumer_of[query_id] = consumer_id
        while len(order) > self._keep:
            _, dropped = order.pop(0)
            self._queries.pop(dropped, None)
            self._members.pop(dropped, None)
            self._consumer_of.pop(dropped, None)
            self._labels.pop(dropped, None)

    def _newest(self, consumer_id: str, limit: int) -> list[str]:
        order = self._order.get(consumer_id, [])
        return [query_id for _, query_id in order[max(0, len(order) - limit) :]] if limit else []

    def _slowest(self, query_id: str) -> None:
        held = self._queries.get(query_id)
        if held is None:
            return
        answered = [
            row
            for row in self._members.get(query_id, ())
            if row["in_ensemble"] and row["status"] == "ok"
        ]
        slowest = max(
            answered,
            key=lambda row: (float(str(row["total_ms"])), str(row["service_name"])),
            default=None,
        )
        held.update(
            {name: slowest[name] for name in FROM_THE_SLOWEST_MEMBER}
            if slowest is not None
            else dict.fromkeys(FROM_THE_SLOWEST_MEMBER)
        )

    def apply_schema(self) -> None:
        self._database.apply_schema()

    def claim_run(self, owner: RunOwner) -> str:
        return self._database.claim_run(owner)

    def register_instance(self, instance: Instance) -> None:
        self._database.register_instance(instance)

    def write_instance_stopped(self, instance_id: str, stopped_at: Timestamp) -> None:
        self._database.write_instance_stopped(instance_id, stopped_at)

    def write_decision(self, decision: Decision) -> None:
        self._database.write_decision(decision)

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        return self._database.read_member_answers(since=since, until=until)

    def read_served(self, *, consumer_id: str, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        return self._database.read_served(consumer_id=consumer_id, since=since, until=until)

    def read_rows(self, table: str) -> Sequence[Row]:
        return self._database.read_rows(table)

    def read_decisions(self, limit: int) -> Sequence[Decision]:
        return self._database.read_decisions(limit)

    def read_profiles(self) -> Sequence[InferenceServiceProfile]:
        return self._database.read_profiles()

    def write_profile(self, profile: InferenceServiceProfile) -> None:
        self._database.write_profile(profile)

    def flush(self) -> None:
        self._database.flush()


def _moment(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"a query arrived at {value!r}, which is not a moment")
    return value
