from __future__ import annotations

from collections.abc import Mapping, Sequence

from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Action, PositiveCount, PositiveNumber, Ratio
from eemls.ensemble_optimization.approach import (
    API_VERSION,
    Approach,
    ApproachSettings,
    Observation,
    PolicyDecision,
    window_accuracy,
    within_evaluation_window,
)
from eemls.evaluation_engine.window import Window


class FrugalMl:
    name = "frugalml"

    def __init__(
        self,
        *,
        max_ensemble_size: int,
        evaluation_window: int,
        budget_per_1k: float,
        quality_level: float,
        failing_intervals: int,
    ) -> None:
        self._limit = max_ensemble_size
        self._window = evaluation_window
        self._budget = budget_per_1k
        self._level = quality_level
        self._failing_intervals = failing_intervals
        self._deployment: dict[str, str] = {}
        self._failing: dict[str, int] = {}

    def decide(self, observed: Observation) -> PolicyDecision:
        observed = within_evaluation_window(observed, self._window)
        consumer_id = observed.consumer.contract.consumer_id
        deployment = observed.deployment.deployment_id
        if self._deployment.get(consumer_id) != deployment:
            self._deployment[consumer_id] = deployment
            self._failing[consumer_id] = 0
        profiles = observed.profiles
        serving = observed.deployment.service_names
        spent = sum(profiles[name].cost_per_1k for name in serving if name in profiles)
        if profiles and spent > self._budget:
            cheapest = min(sorted(profiles), key=lambda name: profiles[name].cost_per_1k)
            return PolicyDecision(action=Action(kind="select", members=(cheapest,)))
        accuracy = window_accuracy(
            Window([row for row in observed.window if row.get("deployment_id") == deployment])
        )
        if accuracy is None:
            return PolicyDecision(action=Action(kind="keep"))
        if accuracy >= self._level:
            self._failing[consumer_id] = 0
            return PolicyDecision(action=Action(kind="keep"))
        self._failing[consumer_id] += 1
        if self._failing[consumer_id] < self._failing_intervals:
            return PolicyDecision(action=Action(kind="keep"))
        return PolicyDecision(action=self._escalated(serving, profiles))

    def _escalated(
        self, serving: Sequence[str], profiles: Mapping[str, InferenceServiceProfile]
    ) -> Action:
        held = [name for name in serving if name in profiles]
        spent = sum(profiles[name].cost_per_1k for name in held)
        outside = sorted(
            (name for name in profiles if name not in set(serving)),
            key=lambda name: (-profiles[name].general_accuracy, name),
        )
        if len(serving) < self._limit:
            for name in outside:
                if spent + profiles[name].cost_per_1k <= self._budget:
                    return Action(kind="add", member_in=name)
            return Action(kind="keep")
        if not held:
            return Action(kind="keep")
        weakest = min(held, key=lambda name: (profiles[name].general_accuracy, name))
        for name in outside:
            better = profiles[name].general_accuracy > profiles[weakest].general_accuracy
            cost = spent - profiles[weakest].cost_per_1k + profiles[name].cost_per_1k
            if better and cost <= self._budget:
                return Action(kind="replace", member_in=name, member_out=weakest)
        return Action(kind="keep")


class FrugalMlSettings(ApproachSettings):
    evaluation_window: PositiveCount
    budget_per_1k: PositiveNumber
    quality_level: Ratio
    failing_intervals: PositiveCount


def build(settings: FrugalMlSettings, *, classes: Sequence[str], seed: int) -> FrugalMl:
    return FrugalMl(
        max_ensemble_size=settings.max_ensemble_size,
        evaluation_window=settings.evaluation_window,
        budget_per_1k=settings.budget_per_1k,
        quality_level=settings.quality_level,
        failing_intervals=settings.failing_intervals,
    )


approach = Approach(
    name=FrugalMl.name, api_version=API_VERSION, Settings=FrugalMlSettings, build=build
)
