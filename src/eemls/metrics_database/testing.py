from __future__ import annotations

from collections.abc import Iterable, Sequence

from eemls.domain.records import (
    Decision,
    Feedback,
    InferenceServiceProfile,
    Instance,
    RunOwner,
    ServedQuery,
)
from eemls.domain.types import FrozenModel, MemberPrediction, Row, Timestamp
from eemls.metrics_database.schema import exported_order, readable_columns, table_row


def window_columns() -> tuple[str, ...]:
    return tuple(sorted(readable_columns() - {"run_id"}))


class MemoryMetricsDatabase:
    def __init__(self, run_id: str = "test") -> None:
        self.run_id = run_id
        self.applied = 0
        self.owner: RunOwner | None = None
        self.instances: dict[str, Instance] = {}
        self.queries: dict[str, ServedQuery] = {}
        self.predictions: dict[tuple[str, str, bool], MemberPrediction] = {}
        self.feedback: dict[str, Feedback] = {}
        self.decisions: dict[str, Decision] = {}
        self.profiles: dict[tuple[str, str], InferenceServiceProfile] = {}
        self.flushes = 0
        self.closed = False

    def apply_schema(self) -> None:
        self.applied += 1

    def claim_run(self, owner: RunOwner) -> str:
        self.owner = self.owner or owner
        return self.owner.instance_id

    def register_instance(self, instance: Instance) -> None:
        self.instances.setdefault(instance.instance_id, instance)

    def write_instance_stopped(self, instance_id: str, stopped_at: Timestamp) -> None:
        held = self.instances.get(instance_id)
        if held is not None and held.stopped_at is None:
            self.instances[instance_id] = held.model_copy(update={"stopped_at": stopped_at})

    def write_query(self, query: ServedQuery) -> None:
        self.queries.setdefault(query.query_id, query)

    def write_member_predictions(self, predictions: Sequence[MemberPrediction]) -> None:
        for one in predictions:
            # A service holds one row of a query while serving and one while probed.
            held = (one.query_id, one.member.service_name, one.in_ensemble)
            self.predictions.setdefault(held, one)

    def write_feedback(self, feedback: Feedback) -> None:
        self.feedback.setdefault(feedback.query_id, feedback)

    def write_decision(self, decision: Decision) -> None:
        self.decisions.setdefault(decision.decision_id, decision)

    def write_profile(self, profile: InferenceServiceProfile) -> None:
        self.profiles.setdefault((profile.instance_id, profile.window_end.isoformat()), profile)

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        mine = sorted(
            (held for held in self.queries.values() if held.consumer_id == consumer_id),
            key=lambda held: (held.t_arrive, held.query_id),
        )
        rows: list[Row] = []
        for query in mine[-max(0, int(limit)) :]:
            returned = self.feedback.get(query.query_id)
            if returned is not None and returned.consumer_id != consumer_id:
                returned = None
            answered = [
                one
                for one in self.predictions.values()
                if one.query_id == query.query_id and one.in_ensemble and one.status == "ok"
            ]
            row = dict.fromkeys(window_columns())
            row |= table_row("queries", query)
            row["true_label"] = returned.true_label if returned else None
            row["t_feedback"] = returned.t_feedback if returned else None
            row["delay_ms"] = returned.delay_ms if returned else None
            if answered:
                slowest = max(answered, key=lambda one: (one.total_ms, one.member.service_name))
                row |= {
                    "total_ms": slowest.total_ms,
                    "fetch_ms": slowest.fetch_ms,
                    "inference_ms": slowest.inference_ms,
                }
            rows.append(row)
        return rows

    def read_member_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        mine = sorted(
            (held for held in self.queries.values() if held.consumer_id == consumer_id),
            key=lambda held: (held.t_arrive, held.query_id),
        )
        newest = mine[-max(0, int(limit)) :] if limit else []
        place = {held.query_id: index for index, held in enumerate(newest)}
        rows: list[Row] = []
        for one in sorted(
            (held for held in self.predictions.values() if held.query_id in place),
            key=lambda held: (
                place[held.query_id],
                held.member.service_name,
                not held.in_ensemble,
            ),
        ):
            returned = self.feedback.get(one.query_id)
            row = table_row("member_predictions", one)
            row["true_label"] = returned.true_label if returned else None
            rows.append(row)
        return rows

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        within = {held.query_id for held in self.queries.values() if since <= held.t_arrive < until}
        rows: list[Row] = []
        ordered = sorted(
            self.predictions.values(), key=lambda p: (p.query_id, p.member.service_name)
        )
        for one in ordered:
            if one.query_id not in within:
                continue
            returned = self.feedback.get(one.query_id)
            row = table_row("member_predictions", one)
            row["true_label"] = returned.true_label if returned else None
            rows.append(row)
        return rows

    def read_served(self, *, consumer_id: str, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        held = sorted(
            (
                one
                for one in self.queries.values()
                if one.consumer_id == consumer_id and since <= one.t_arrive < until
            ),
            key=lambda one: (one.t_arrive, one.query_id),
        )
        return [table_row("queries", one) for one in held]

    def read_rows(self, table: str) -> Sequence[Row]:
        order = exported_order(table)
        held: dict[str, Iterable[FrozenModel]] = {
            "instances": self.instances.values(),
            "queries": self.queries.values(),
            "member_predictions": self.predictions.values(),
            "feedback": self.feedback.values(),
            "decisions": self.decisions.values(),
            "inference_service_profiles": self.profiles.values(),
        }
        rows = [table_row(table, one) for one in held[table]]
        return sorted(rows, key=lambda row: tuple(str(row[name]) for name in order))

    def read_decisions(self, limit: int) -> Sequence[Decision]:
        newest = sorted(
            self.decisions.values(), key=lambda one: (one.t, one.decision_id), reverse=True
        )
        return newest[: max(0, int(limit))]

    def read_profiles(self) -> Sequence[InferenceServiceProfile]:
        latest: dict[str, InferenceServiceProfile] = {}
        for held in sorted(
            self.profiles.values(), key=lambda one: (one.instance_id, one.window_end)
        ):
            latest[held.instance_id] = held
        return tuple(latest[name] for name in sorted(latest))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.flush()
        self.closed = True
