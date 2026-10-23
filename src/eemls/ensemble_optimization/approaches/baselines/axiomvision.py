from __future__ import annotations

import math
from collections.abc import Sequence

from eemls.domain.types import Action, PositiveCount, PositiveNumber, Ratio, RuntimeContext
from eemls.ensemble_optimization.approach import (
    API_VERSION,
    Approach,
    ApproachSettings,
    Observation,
    PolicyDecision,
    within_evaluation_window,
)

CONTEXT_KEY = "period_name"


class AxiomVision:
    name = "axiomvision"

    def __init__(
        self,
        *,
        max_ensemble_size: int,
        evaluation_window: int,
        exploration_alpha: float,
        prior_zeta: float,
        accuracy_target: float,
    ) -> None:
        self._limit = max_ensemble_size
        self._window = evaluation_window
        self._alpha = exploration_alpha
        self._zeta = prior_zeta
        self._target = accuracy_target
        self._right: dict[tuple[str, str], int] = {}
        self._tried: dict[tuple[str, str], int] = {}
        self._prior: dict[str, float] = {}
        self._counted: dict[str, set[tuple[str, str]]] = {}

    def decide(self, observed: Observation) -> PolicyDecision:
        observed = within_evaluation_window(observed, self._window)
        self._learn(observed)
        profiles = observed.profiles
        self._prior.update({name: one.general_accuracy for name, one in profiles.items()})
        if not profiles:
            return PolicyDecision(action=Action(kind="keep"))
        period = _period_of(observed.context)
        if not any(seen == period for seen, _ in self._tried):
            return PolicyDecision(action=Action(kind="keep"), candidates=len(profiles))
        return PolicyDecision(
            action=Action(kind="select", members=self._selected(period, sorted(profiles))),
            candidates=len(profiles),
        )

    def _selected(self, period: str, names: Sequence[str]) -> tuple[str, ...]:
        ranked = sorted(names, key=lambda name: (-self._ranked_value(period, name), name))
        chosen: list[str] = []
        for name in ranked[: self._limit]:
            chosen.append(name)
            if self._estimate(period, chosen) >= self._target:
                break
        return tuple(chosen)

    def _estimate(self, period: str, serving: Sequence[str]) -> float:
        return 1.0 - math.prod(1.0 - self._mean(period, name) for name in serving)

    def _mean(self, period: str, service_name: str) -> float:
        key = (period, service_name)
        prior = self._zeta * self._prior.get(service_name, 0.0)
        return (self._right.get(key, 0) + prior) / (self._zeta + self._tried.get(key, 0))

    def _ranked_value(self, period: str, service_name: str) -> float:
        tried = self._tried.get((period, service_name), 0)
        return self._mean(period, service_name) + self._alpha * math.sqrt(
            1.0 / (self._zeta + tried)
        )

    def _learn(self, observed: Observation) -> None:
        periods = {
            str(row["query_id"]): str(row[CONTEXT_KEY])
            for row in observed.window
            if row.get(CONTEXT_KEY) is not None
        }
        consumer_id = observed.consumer.contract.consumer_id
        counted = self._counted.get(consumer_id, set())
        present: set[tuple[str, str]] = set()
        for row in observed.member_window:
            query_id = str(row["query_id"])
            service_name = str(row["service_name"])
            if query_id not in periods:
                continue
            if row.get("true_label") is None or row.get("predicted_label") is None:
                continue
            seen = (query_id, service_name)
            present.add(seen)
            if seen in counted:
                continue
            key = (periods[query_id], service_name)
            self._tried[key] = self._tried.get(key, 0) + 1
            if row["predicted_label"] == row["true_label"]:
                self._right[key] = self._right.get(key, 0) + 1
        self._counted[consumer_id] = present


def _period_of(context: RuntimeContext) -> str:
    return str(getattr(context, CONTEXT_KEY))


class AxiomVisionSettings(ApproachSettings):
    evaluation_window: PositiveCount
    exploration_alpha: PositiveNumber
    prior_zeta: PositiveNumber
    accuracy_target: Ratio


def build(settings: AxiomVisionSettings, *, classes: Sequence[str], seed: int) -> AxiomVision:
    return AxiomVision(
        max_ensemble_size=settings.max_ensemble_size,
        evaluation_window=settings.evaluation_window,
        exploration_alpha=settings.exploration_alpha,
        prior_zeta=settings.prior_zeta,
        accuracy_target=settings.accuracy_target,
    )


approach = Approach(
    name=AxiomVision.name, api_version=API_VERSION, Settings=AxiomVisionSettings, build=build
)
