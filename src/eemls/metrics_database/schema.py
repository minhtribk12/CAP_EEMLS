from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence

from eemls.config.consumers import ConsumerConfig
from eemls.config.reading import ConfigError
from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.errors import EemlsError
from eemls.domain.records import (
    Decision,
    Feedback,
    InferenceServiceProfile,
    Instance,
    RunOwner,
    ServedQuery,
)
from eemls.domain.types import FrozenModel, MemberPrediction


class CoverageError(EemlsError): ...


TABLES: Mapping[str, str] = {
    "run_owners": """
CREATE TABLE IF NOT EXISTS run_owners (
    run_id          text        NOT NULL,
    instance_id     text        NOT NULL,
    claimed_at      timestamptz NOT NULL,
    PRIMARY KEY (run_id)
)""",
    "instances": """
CREATE TABLE IF NOT EXISTS instances (
    run_id          text        NOT NULL,
    instance_id     text        NOT NULL,
    service_name    text        NOT NULL,
    model           text        NOT NULL,
    device          text        NOT NULL,
    node            text        NOT NULL,
    replica         integer     NOT NULL,
    provider        text        NOT NULL,
    number_format       text        NOT NULL,
    runtime_version text        NOT NULL,
    image_tag       text        NOT NULL,
    config_hash     text        NOT NULL,
    started_at      timestamptz NOT NULL,
    stopped_at      timestamptz,
    PRIMARY KEY (run_id, instance_id)
)""",
    "queries": """
CREATE TABLE IF NOT EXISTS queries (
    run_id               text        NOT NULL,
    query_id             text        NOT NULL,
    consumer_id          text        NOT NULL,
    image_id             text        NOT NULL,
    dataset              text,
    image_quality              text        NOT NULL,
    time_of_day          double precision,
    t_arrive             timestamptz NOT NULL,
    t_result_ready         timestamptz,
    response_time_ms           double precision,
    wait_ms              double precision,
    aggregate_ms              double precision,
    t_publish            timestamptz,
    publish_ms           double precision,
    period_name                text        NOT NULL,
    period_index         integer     NOT NULL,
    condition             text        NOT NULL,
    weather              text        NOT NULL,
    event_kind           text,
    event_id             text,
    deployment_id           text,
    n_members            integer,
    n_answered           integer,
    predicted_label      text,
    predicted_confidence double precision,
    cost_per_1k    double precision,
    throughput_rps       double precision,
    outcome              text        NOT NULL,
    PRIMARY KEY (run_id, query_id)
)""",
    "member_predictions": """
CREATE TABLE IF NOT EXISTS member_predictions (
    run_id          text             NOT NULL,
    query_id        text             NOT NULL,
    instance_id     text,
    service_name    text             NOT NULL,
    model           text             NOT NULL,
    device          text             NOT NULL,
    node            text,
    replica         integer,
    in_ensemble     boolean          NOT NULL,
    predicted_label text,
    confidence      double precision,
    fetch_ms        double precision NOT NULL,
    inference_ms        double precision NOT NULL,
    total_ms        double precision NOT NULL,
    provider        text,
    number_format       text,
    cache_hit       boolean,
    status          text             NOT NULL,
    PRIMARY KEY (run_id, query_id, service_name, in_ensemble)
)""",
    "feedback": """
CREATE TABLE IF NOT EXISTS feedback (
    run_id      text             NOT NULL,
    query_id    text             NOT NULL,
    consumer_id text             NOT NULL,
    true_label  text             NOT NULL,
    t_feedback  timestamptz      NOT NULL,
    delay_ms    double precision NOT NULL,
    PRIMARY KEY (run_id, query_id)
)""",
    "decisions": """
CREATE TABLE IF NOT EXISTS decisions (
    run_id         text             NOT NULL,
    decision_id    text             NOT NULL,
    consumer_id    text             NOT NULL,
    t              timestamptz      NOT NULL,
    policy         text             NOT NULL,
    action         jsonb            NOT NULL,
    context        jsonb            NOT NULL,
    accepted       boolean          NOT NULL,
    reason         text             NOT NULL,
    deployment_before text             NOT NULL,
    deployment_after  text             NOT NULL,
    score          double precision,
    cdm_values     jsonb            NOT NULL,
    candidates     integer          NOT NULL,
    scored         integer,
    decide_ms      double precision NOT NULL,
    cdm_ms         double precision,
    contribution_ms      double precision,
    select_ms      double precision,
    PRIMARY KEY (run_id, decision_id)
)""",
    "inference_service_profiles": """
CREATE TABLE IF NOT EXISTS inference_service_profiles (
    run_id            text             NOT NULL,
    model             text             NOT NULL,
    device            text             NOT NULL,
    node              text             NOT NULL,
    instance_id       text             NOT NULL,
    service_name      text             NOT NULL,
    window_start      timestamptz      NOT NULL,
    window_end        timestamptz      NOT NULL,
    mean_ms           double precision NOT NULL,
    p50_ms            double precision NOT NULL,
    p95_ms            double precision NOT NULL,
    max_ms            double precision NOT NULL,
    throughput_rps    double precision NOT NULL,
    n_queries         integer          NOT NULL,
    energy_wh_per_1k  double precision NOT NULL,
    cost_per_1k       double precision NOT NULL,
    general_accuracy  double precision NOT NULL,
    accuracy_by_class jsonb            NOT NULL,
    confidence_by_class jsonb          NOT NULL,
    source            text             NOT NULL,
    PRIMARY KEY (run_id, instance_id, window_end)
)""",
}

INDEXES: Sequence[str] = (
    "CREATE INDEX IF NOT EXISTS queries_window ON queries (run_id, consumer_id, t_arrive)",
    "CREATE INDEX IF NOT EXISTS queries_by_deployment ON queries (run_id, deployment_id)",
    "CREATE INDEX IF NOT EXISTS member_predictions_window ON member_predictions"
    " (run_id, service_name, query_id)",
    "CREATE INDEX IF NOT EXISTS feedback_window ON feedback (run_id, consumer_id, t_feedback)",
    "CREATE INDEX IF NOT EXISTS decisions_window ON decisions (run_id, consumer_id, t)",
    "CREATE INDEX IF NOT EXISTS inference_service_profiles_latest ON"
    " inference_service_profiles (run_id, instance_id, window_end DESC)",
)

WRITTEN_BY: Mapping[str, type[FrozenModel]] = {
    "run_owners": RunOwner,
    "instances": Instance,
    "queries": ServedQuery,
    "member_predictions": MemberPrediction,
    "feedback": Feedback,
    "decisions": Decision,
    "inference_service_profiles": InferenceServiceProfile,
}

STORED_FLAT: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("queries", "context"): (
        "period_name",
        "period_index",
        "condition",
        "weather",
        "event_kind",
        "event_id",
    ),
    ("member_predictions", "member"): ("service_name", "model", "device"),
}

EXPORTED: dict[str, tuple[str, ...]] = {
    "instances": ("started_at", "instance_id"),
    "queries": ("t_arrive", "query_id"),
    "member_predictions": ("query_id", "service_name", "in_ensemble"),
    "feedback": ("t_feedback", "query_id"),
    "decisions": ("t", "decision_id"),
    "inference_service_profiles": ("window_end", "instance_id"),
}


def exported_order(table: str) -> tuple[str, ...]:
    if table not in EXPORTED:
        raise CoverageError(f"{table} is not a table a run is exported from: {sorted(EXPORTED)}")
    return EXPORTED[table]


COLUMN = re.compile(r"^\s{4}(\w+)\s+\S")
CONSTRAINTS = ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")


@functools.cache
def _parsed(table: str) -> tuple[tuple[str, bool], ...]:
    if table not in TABLES:
        raise CoverageError(f"{table} is not a table of this store")
    return tuple(
        (match.group(1), " jsonb" in line)
        for line in TABLES[table].splitlines()
        if (match := COLUMN.match(line)) and match.group(1) not in CONSTRAINTS
    )


def table_columns(table: str) -> tuple[str, ...]:
    return tuple(name for name, _ in _parsed(table))


def field_columns(table: str, field: str) -> tuple[str, ...]:
    expanded = STORED_FLAT.get((table, field))
    if expanded is not None:
        return expanded
    if field not in table_columns(table):
        raise CoverageError(f"{table} has no column for {field}")
    return (field,)


def json_columns(table: str) -> frozenset[str]:
    return frozenset(name for name, as_json in _parsed(table) if as_json)


def schema_statements() -> tuple[str, ...]:
    return (*TABLES.values(), *INDEXES)


# Columns of a window read.
FROM_FEEDBACK: Sequence[str] = ("true_label", "t_feedback", "delay_ms")
FROM_THE_SLOWEST_MEMBER: Sequence[str] = ("total_ms", "fetch_ms", "inference_ms")


def readable_columns() -> set[str]:
    return set(table_columns("queries")) | set(FROM_FEEDBACK) | set(FROM_THE_SLOWEST_MEMBER)


def run_row(table: str, value: FrozenModel, run_id: str) -> dict[str, object]:
    return {"run_id": run_id, **table_row(table, value)}


def table_row(table: str, value: FrozenModel) -> dict[str, object]:
    written = WRITTEN_BY[table]
    if not isinstance(value, written):
        raise CoverageError(f"{table} holds {written.__name__}, not {type(value).__name__}")
    row: dict[str, object] = {}
    for name in (*written.model_fields, *written.model_computed_fields):
        held = getattr(value, name)
        columns = field_columns(table, name)
        if columns == (name,):
            row[name] = _stored(held)
            continue
        for column in columns:
            row[column] = _stored(getattr(held, column))
    return row


def _stored(value: object) -> object:
    if isinstance(value, FrozenModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def check_coverage(declared: CdmSet) -> None:
    readable = readable_columns()
    for name, definition in declared.definitions.items():
        unknown = sorted(definition.attributes() - readable)
        if unknown:
            raise CoverageError(f"{name} reads {unknown}, which no table holds")


def check_every_scored_cdm_can_be_computed(consumers: Mapping[str, ConsumerConfig]) -> None:
    for consumer_id, consumer in consumers.items():
        try:
            check_coverage(CdmSet(definitions=consumer.cdms))
        except CoverageError as error:
            raise ConfigError(f"consumers/{consumer_id}/cdm.yaml: {error}") from error
