from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from eemls.domain.errors import EemlsError
from eemls.domain.measuring import Measuring, milliseconds_since
from eemls.domain.records import Decision, InferenceServiceProfile
from eemls.domain.statistics import is_number
from eemls.ensemble_optimization.approach import (
    ConsumerSpecification,
    Observation,
    Policy,
    PolicyDecision,
)
from eemls.evaluation_engine.aggregation_functions import Value
from eemls.evaluation_engine.consumer_evaluation import evaluated_window, window_size_for
from eemls.evaluation_engine.window import Window
from eemls.metrics_database.database import MetricsDatabase
from eemls.observation_service.profile_updates import ProfileUpdating
from eemls.observation_service.runtime_context import ServingContext
from eemls.orchestration.ensemble_deployments import EnsembleDeployments

logger = logging.getLogger(__name__)


class NoContractError(EemlsError): ...


@dataclass(frozen=True, slots=True)
class DecisionTimings:
    candidates: int
    scored: int | None
    contribution_ms: float | None
    select_ms: float | None


def utc_now() -> datetime:
    return datetime.now(UTC)


class DecisionLoop:
    def __init__(
        self,
        *,
        deployments: EnsembleDeployments,
        database: MetricsDatabase,
        policy: Policy,
        consumers: Mapping[str, ConsumerSpecification],
        monitoring: Measuring,
        runtime_context: ServingContext,
        now: Callable[[], datetime] = utc_now,
        policy_window: int,
        profiles: ProfileUpdating,
    ) -> None:
        self._deployments = deployments
        self._database = database
        self._policy = policy
        self._consumers = consumers
        self._rows = {
            name: window_size_for(one.cdms, at_least=policy_window)
            for name, one in consumers.items()
        }
        self._monitoring = monitoring
        self._runtime_context = runtime_context
        self._now = now
        self._profiles = profiles
        self._taken: dict[str, int] = {}
        self._deciding = threading.Lock()
        self._contracted(deployments.consumers())

    def decide(self, consumer_id: str) -> Decision:
        with self._deciding:
            return self._decided(consumer_id)

    def _decided(self, consumer_id: str) -> Decision:
        at = self._now()
        profiles = self._profiles.latest(at)
        started = time.perf_counter()
        observed, cdm_ms = self._observed(consumer_id, at, profiles)
        asking = time.perf_counter()
        decided = self._policy.decide(observed)
        reported = self._reported(consumer_id, decided, milliseconds_since(asking))
        applied = self._deployments.apply(consumer_id=consumer_id, action=decided.action)
        decision = Decision(
            decision_id=self._named(consumer_id),
            consumer_id=consumer_id,
            t=at,
            policy=self._policy.name,
            action=decided.action,
            context=observed.context,
            accepted=applied.accepted,
            reason=applied.reason,
            deployment_before=applied.before.deployment_id,
            deployment_after=applied.after.deployment_id,
            score=observed.score.total if observed.score is not None else None,
            cdm_values=_numbers(observed.cdm_values),
            candidates=reported.candidates,
            scored=reported.scored,
            decide_ms=milliseconds_since(started),
            cdm_ms=cdm_ms,
            contribution_ms=reported.contribution_ms,
            select_ms=reported.select_ms,
        )
        self._database.write_decision(decision)
        self._report(decision, applied.after.n_members, observed.cdm_values)
        return decision

    def _observed(
        self, consumer_id: str, at: datetime, profiles: Mapping[str, InferenceServiceProfile]
    ) -> tuple[Observation, float]:
        consumer = self._consumers[consumer_id]
        deployment = self._deployments.current(consumer_id)
        rows = self._rows[consumer_id]
        window = Window(self._database.read_window(consumer_id, rows))
        answered = self._database.read_member_window(consumer_id, rows)
        members = Window([row for row in answered if row["in_ensemble"]])
        self._runtime_context.read_from(consumer_id, window)
        context = self._runtime_context(consumer_id)
        reading = time.perf_counter()
        values, parts = evaluated_window(consumer.contract, consumer.cdms, window)
        cdm_ms = milliseconds_since(reading)
        observed = Observation(
            consumer=consumer,
            t=at,
            context=context,
            deployment=deployment,
            available=self._deployments.available_inference_services,
            profiles=profiles,
            window=window,
            member_window=members,
            cdm_values=MappingProxyType(dict(values)),
            score=parts,
        )
        return observed, cdm_ms

    def _contracted(self, served: tuple[str, ...]) -> None:
        undeclared = sorted(set(served) - set(self._consumers))
        if undeclared:
            raise NoContractError(f"{undeclared} are served but declared no contract")
        unserved = sorted(set(self._consumers) - set(served))
        if unserved:
            raise NoContractError(f"{unserved} declared a contract but are served by nothing")

    def _reported(
        self, consumer_id: str, decided: PolicyDecision, asked_ms: float
    ) -> DecisionTimings:
        candidates = decided.candidates
        if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 0:
            self._unusable(consumer_id, "candidates", candidates)
            candidates = 0
        scored = decided.scored
        if scored is not None and (
            not isinstance(scored, int) or isinstance(scored, bool) or scored < 0
        ):
            self._unusable(consumer_id, "scored", scored)
            scored = None
        parts = [self._measured(consumer_id, name, part) for name, part in _parts_of(decided)]
        accounted = sum(part for part in parts if part is not None)
        if accounted > asked_ms + 1e-6:
            self._unusable(consumer_id, "the parts of deciding", accounted)
            return DecisionTimings(candidates, scored, None, None)
        return DecisionTimings(candidates, scored, parts[0], parts[1])

    def _measured(self, consumer_id: str, name: str, part: float | None) -> float | None:
        if part is None:
            return None
        if not is_number(part):
            self._unusable(consumer_id, name, part)
            return None
        if not math.isfinite(part) or part < 0.0:
            self._unusable(consumer_id, name, part)
            return None
        return float(part)

    def _unusable(self, consumer_id: str, name: str, held: object) -> None:
        logger.error(
            f"stage_tag=decide process_tag=orchestrator error_type=unusable_report "
            f"consumer={consumer_id} policy={self._policy.name} reported={name}={held!r} "
            f"recovery_action=the_decision_is_stored_without_it"
        )

    def _named(self, consumer_id: str) -> str:
        taken = self._taken.get(consumer_id, 0) + 1
        self._taken[consumer_id] = taken
        return f"{consumer_id}-{taken:06d}"

    def _report(self, decision: Decision, serving: int, values: Mapping[str, Value]) -> None:
        self._monitoring.count(
            "eemls_decisions_total",
            consumer=decision.consumer_id,
            policy=decision.policy,
            action=decision.action.kind,
            accepted="true" if decision.accepted else "false",
        )
        self._monitoring.observe(
            "eemls_decide_seconds",
            seconds=decision.decide_ms / 1000.0,
            consumer=decision.consumer_id,
            policy=decision.policy,
        )
        self._monitoring.gauge(
            "eemls_ensemble_size", value=float(serving), consumer=decision.consumer_id
        )
        for name, value in _numbers(values).items():
            self._monitoring.gauge(
                "eemls_cdm_value", value=value, consumer=decision.consumer_id, metric=name
            )
        if not decision.accepted:
            logger.info(
                f"stage_tag=decide process_tag=orchestrator error_type=action_refused "
                f"consumer={decision.consumer_id} policy={decision.policy} "
                f"action={decision.action.kind} recovery_action=the_ensemble_is_unchanged "
                f"detail={decision.reason}"
            )


def _parts_of(decided: PolicyDecision) -> tuple[tuple[str, float | None], ...]:
    return (("contribution_ms", decided.contribution_ms), ("select_ms", decided.select_ms))


def _numbers(values: Mapping[str, Value]) -> dict[str, float]:
    return {name: float(value) for name, value in values.items() if is_number(value)}
