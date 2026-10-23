from __future__ import annotations

from pydantic import computed_field, model_validator

from eemls.domain.types import (
    DEFAULT_IMAGE_QUALITY,
    Action,
    Digest,
    FrozenMap,
    FrozenModel,
    Identifier,
    ImageQuality,
    Milliseconds,
    NonNegativeCount,
    NonNegativeNumber,
    NumberFormat,
    PositiveCount,
    ProfileSource,
    QueryOutcome,
    Ratio,
    RuntimeContext,
    Score,
    Text,
    TimeOfDay,
    Timestamp,
)


class ServedQuery(FrozenModel):
    query_id: Identifier
    consumer_id: Identifier
    image_id: Identifier
    dataset: str | None = None
    image_quality: ImageQuality = DEFAULT_IMAGE_QUALITY
    time_of_day: TimeOfDay | None = None
    t_arrive: Timestamp
    t_result_ready: Timestamp | None
    wait_ms: Milliseconds | None
    aggregate_ms: Milliseconds | None
    t_publish: Timestamp | None = None
    context: RuntimeContext
    deployment_id: Identifier | None
    n_members: PositiveCount | None
    n_answered: PositiveCount | None = None
    predicted_label: str | None
    predicted_confidence: Ratio | None
    cost_per_1k: NonNegativeNumber | None
    throughput_rps: NonNegativeNumber | None
    outcome: QueryOutcome

    @computed_field  # type: ignore[prop-decorator]
    @property
    def response_time_ms(self) -> float | None:
        if self.t_result_ready is None:
            return None
        return (self.t_result_ready - self.t_arrive).total_seconds() * 1000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def publish_ms(self) -> float | None:
        if self.t_publish is None or self.t_result_ready is None:
            return None
        return (self.t_publish - self.t_result_ready).total_seconds() * 1000.0

    @model_validator(mode="after")
    def _the_clock_runs_forward(self) -> ServedQuery:
        if self.t_result_ready is not None and self.t_result_ready < self.t_arrive:
            raise ValueError("the result was ready before the query arrived")
        if self.t_publish is not None and self.t_result_ready is None:
            raise ValueError("the result was published although none was ready")
        published_early = (
            self.t_publish is not None
            and self.t_result_ready is not None
            and self.t_publish < self.t_result_ready
        )
        if published_early:
            raise ValueError("the result was published before it was ready")
        return self

    @model_validator(mode="after")
    def _the_parts_fit_inside_the_whole(self) -> ServedQuery:
        whole = self.response_time_ms
        if whole is None:
            return self
        parts = sum(part for part in (self.wait_ms, self.aggregate_ms) if part)
        if parts > whole + 1e-6:
            raise ValueError(f"the parts take {parts} ms, longer than the {whole} ms they sit in")
        return self

    @model_validator(mode="after")
    def _only_a_served_query_carries_an_answer(self) -> ServedQuery:
        answered = (
            self.predicted_label is not None
            and self.predicted_confidence is not None
            and self.deployment_id is not None
            and self.t_result_ready is not None
        )
        explained = (
            self.wait_ms is not None
            and self.aggregate_ms is not None
            and self.cost_per_1k is not None
            and self.throughput_rps is not None
        )
        if self.outcome == "accepted" and not answered:
            raise ValueError("the query was accepted but carries no answer")
        if self.outcome == "accepted" and not explained:
            raise ValueError("the query was accepted but carries no time split or no cost")
        if self.outcome != "accepted" and answered:
            raise ValueError(f"the query was not served, outcome {self.outcome}, but carries one")
        return self


class Feedback(FrozenModel):
    query_id: Identifier
    consumer_id: Identifier
    true_label: Identifier
    t_feedback: Timestamp
    delay_ms: Milliseconds


class Decision(FrozenModel):
    decision_id: Identifier
    consumer_id: Identifier
    t: Timestamp
    policy: Identifier
    action: Action
    context: RuntimeContext
    accepted: bool
    reason: Text
    deployment_before: Identifier
    deployment_after: Identifier
    score: Score | None
    cdm_values: FrozenMap[float]
    candidates: NonNegativeCount
    scored: NonNegativeCount | None
    decide_ms: Milliseconds
    cdm_ms: Milliseconds | None
    contribution_ms: Milliseconds | None
    select_ms: Milliseconds | None

    @model_validator(mode="after")
    def _the_time_split_fits_inside_the_decision(self) -> Decision:
        parts = sum(part for part in (self.cdm_ms, self.contribution_ms, self.select_ms) if part)
        if parts > self.decide_ms + 1e-6:
            raise ValueError(
                f"the parts take {parts} ms, longer than the {self.decide_ms} ms they sit in"
            )
        return self

    @model_validator(mode="after")
    def _the_version_changes_exactly_when_something_was_done(self) -> Decision:
        changes_one_member = self.accepted and self.action.kind in ("add", "remove", "replace")
        if changes_one_member and self.deployment_before == self.deployment_after:
            raise ValueError("the action was accepted but the version did not change")
        may_change = self.accepted and self.action.kind != "keep"
        if not may_change and self.deployment_before != self.deployment_after:
            raise ValueError("nothing was done but the version changed")
        return self


class RunOwner(FrozenModel):
    instance_id: Identifier
    claimed_at: Timestamp


class Instance(FrozenModel):
    instance_id: Identifier
    service_name: Identifier
    model: Identifier
    device: Identifier
    node: Identifier
    replica: PositiveCount
    provider: Identifier
    number_format: NumberFormat
    runtime_version: Identifier
    image_tag: Identifier
    config_hash: Digest
    started_at: Timestamp
    stopped_at: Timestamp | None = None

    @model_validator(mode="after")
    def _it_cannot_stop_before_it_started(self) -> Instance:
        if self.stopped_at is not None and self.stopped_at < self.started_at:
            raise ValueError("the instance stopped before it started")
        return self


class InferenceServiceProfile(FrozenModel):
    model: Identifier
    device: Identifier
    node: Identifier
    instance_id: Identifier
    service_name: Identifier
    window_start: Timestamp
    window_end: Timestamp
    mean_ms: Milliseconds
    p50_ms: Milliseconds
    p95_ms: Milliseconds
    max_ms: Milliseconds
    throughput_rps: NonNegativeNumber
    n_queries: NonNegativeCount
    energy_wh_per_1k: NonNegativeNumber
    cost_per_1k: NonNegativeNumber
    general_accuracy: Ratio
    accuracy_by_class: FrozenMap[Ratio]
    confidence_by_class: FrozenMap[Ratio]
    source: ProfileSource

    @model_validator(mode="after")
    def _the_profile_is_possible(self) -> InferenceServiceProfile:
        if self.window_end <= self.window_start:
            raise ValueError("the window has no length")
        if not self.p50_ms <= self.p95_ms <= self.max_ms:
            raise ValueError("the percentiles are not in rising order")
        if not self.mean_ms <= self.max_ms:
            raise ValueError(f"the mean {self.mean_ms} ms sits outside the range it summarizes")
        if self.n_queries == 0 and self.throughput_rps > 0.0:
            raise ValueError("the window held no query but reports a throughput")
        return self
