from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from eemls.domain.types import ActionKind, MemberStatus, QueryOutcome, ResultOutcome

Kind = Literal["counter", "histogram", "gauge"]

ALLOWED_VALUES: Mapping[str, tuple[str, ...]] = {
    "outcome": get_args(QueryOutcome),
    "status": get_args(MemberStatus),
    "action": get_args(ActionKind),
    "result": get_args(ResultOutcome),
    "cache": ("hit", "miss"),
    "accepted": ("true", "false"),
    "in_ensemble": ("true", "false"),
}


@dataclass(frozen=True)
class DeclaredMetric:
    kind: Kind
    labels: tuple[str, ...]
    description: str
    fast: bool = False

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("a declared metric needs a description")
        if any(character in self.description for character in "|\n\r"):
            raise ValueError(f"description {self.description!r} holds a pipe or a line break")
        if self.fast and self.kind != "histogram":
            raise ValueError(f"a {self.kind} has no buckets, so it cannot be fast")


DECLARED: Mapping[str, DeclaredMetric] = {
    "eemls_queries_total": DeclaredMetric(
        "counter", ("consumer", "outcome"), "Queries that arrived, accepted, refused or failed"
    ),
    "eemls_response_time_seconds": DeclaredMetric(
        "histogram", ("consumer",), "Arrival to the result being ready, the response time"
    ),
    "eemls_member_call_seconds": DeclaredMetric(
        "histogram",
        ("model", "device", "in_ensemble"),
        "One call to one member, seen by the control plane",
    ),
    "eemls_member_calls_total": DeclaredMetric(
        "counter",
        ("model", "device", "status", "in_ensemble"),
        "Calls to one member and how each ended, a refusal included",
    ),
    "eemls_client_requests_total": DeclaredMetric(
        "counter",
        ("consumer", "outcome"),
        "Queries one consumer sent, and how the control plane answered",
    ),
    "eemls_client_results_total": DeclaredMetric(
        "counter", ("consumer", "result"), "Results a client matched, dropped, or never got"
    ),
    "eemls_client_dropped_total": DeclaredMetric(
        "counter",
        ("consumer",),
        "Queries the rate called for while the client already held its window open",
    ),
    "eemls_client_round_trip_seconds": DeclaredMetric(
        "histogram", ("consumer",), "Sending a query to its result arriving, as the client sees it"
    ),
    "eemls_images_served_total": DeclaredMetric(
        "counter", ("outcome",), "Images the data hub served, refused or could not read"
    ),
    "eemls_image_fetch_seconds": DeclaredMetric(
        "histogram",
        ("model", "cache"),
        "Fetching the image from the data hub by its id",
        fast=True,
    ),
    "eemls_inference_requests_total": DeclaredMetric(
        "counter",
        ("model", "device", "outcome"),
        "Requests one inference service accepted, refused or failed",
    ),
    "eemls_model_inference_seconds": DeclaredMetric(
        "histogram", ("model", "device"), "The model run alone, inside the inference service"
    ),
    "eemls_publish_seconds": DeclaredMetric(
        "histogram", ("consumer",), "Handing the result to the broker", fast=True
    ),
    "eemls_queries_in_progress": DeclaredMetric(
        "gauge", ("consumer",), "Queries in flight per consumer"
    ),
    "eemls_ensemble_size": DeclaredMetric("gauge", ("consumer",), "Members serving a consumer"),
    "eemls_decisions_total": DeclaredMetric(
        "counter",
        ("consumer", "policy", "action", "accepted"),
        "Decisions a policy took, and whether the ensemble changed",
    ),
    "eemls_decide_seconds": DeclaredMetric(
        "histogram", ("consumer", "policy"), "One decision, from reading the window to applying it"
    ),
    "eemls_decision_failures_total": DeclaredMetric(
        "counter", ("consumer",), "Ticks that could not decide for one consumer"
    ),
    "eemls_decisions_skipped_total": DeclaredMetric(
        "counter", ("consumer",), "Intervals that came due while the last decision was running"
    ),
    "eemls_cdm_value": DeclaredMetric(
        "gauge", ("consumer", "metric"), "Last value of a declared CDM"
    ),
    "eemls_feedback_total": DeclaredMetric(
        "counter", ("consumer",), "Feedback messages a consumer returned"
    ),
    "eemls_feedback_ignored_total": DeclaredMetric(
        "counter", ("consumer", "reason"), "Returned labels the store did not keep"
    ),
    "eemls_metrics_database_write_seconds": DeclaredMetric(
        "histogram", (), "One flush of the metrics_database buffer", fast=True
    ),
    "eemls_metrics_database_buffer_rows": DeclaredMetric(
        "gauge", (), "Rows waiting in the metrics_database buffer"
    ),
    "eemls_metrics_database_rows_refused_total": DeclaredMetric(
        "counter", ("table",), "Rows the database refused one at a time and dropped"
    ),
    "eemls_metrics_database_write_failures_total": DeclaredMetric(
        "counter", (), "Flushes of the metrics_database buffer that the store refused"
    ),
}


@dataclass(frozen=True)
class Instruments:
    counters: Mapping[str, Counter]
    histograms: Mapping[str, Histogram]
    gauges: Mapping[str, Gauge]

    def counter(self, name: str, labels: Mapping[str, str]) -> Counter:
        held = self.counters[name]
        return held.labels(**labels) if labels else held

    def histogram(self, name: str, labels: Mapping[str, str]) -> Histogram:
        held = self.histograms[name]
        return held.labels(**labels) if labels else held

    def gauge(self, name: str, labels: Mapping[str, str]) -> Gauge:
        held = self.gauges[name]
        return held.labels(**labels) if labels else held


def build_instruments(
    buckets_seconds: Sequence[float],
    fast_buckets_seconds: Sequence[float],
    *,
    enabled: bool = True,
) -> tuple[CollectorRegistry, Instruments]:
    """Registry of every declared metric; empty when off."""
    registry = CollectorRegistry()
    counters: dict[str, Counter] = {}
    histograms: dict[str, Histogram] = {}
    gauges: dict[str, Gauge] = {}
    if not enabled:
        return registry, Instruments(counters, histograms, gauges)
    for name, declared in DECLARED.items():
        if declared.kind == "counter":
            counters[name] = Counter(
                name.removesuffix("_total"),
                declared.description,
                declared.labels,
                registry=registry,
            )
        elif declared.kind == "gauge":
            gauges[name] = Gauge(name, declared.description, declared.labels, registry=registry)
        else:
            edges = fast_buckets_seconds if declared.fast else buckets_seconds
            histograms[name] = Histogram(
                name,
                declared.description,
                declared.labels,
                buckets=tuple(edges),
                registry=registry,
            )
    return registry, Instruments(counters, histograms, gauges)


def expose(registry: CollectorRegistry) -> tuple[str, str]:
    return generate_latest(registry).decode("utf-8"), CONTENT_TYPE_LATEST
