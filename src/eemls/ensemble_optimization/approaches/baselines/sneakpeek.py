from __future__ import annotations

from collections.abc import Mapping, Sequence

from eemls.domain.contract import ConsumerContract
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Action, PositiveCount
from eemls.ensemble_optimization.approach import (
    API_VERSION,
    Approach,
    ApproachSettings,
    Observation,
    PolicyDecision,
    within_evaluation_window,
)
from eemls.evaluation_engine.contract_scoring import mapped_cost
from eemls.evaluation_engine.window import class_share


class SneakPeek:
    name = "sneakpeek"

    def __init__(
        self, *, classes: Sequence[str], max_ensemble_size: int, evaluation_window: int
    ) -> None:
        self._classes = tuple(classes)
        self._limit = max_ensemble_size
        self._window = evaluation_window

    def decide(self, observed: Observation) -> PolicyDecision:
        observed = within_evaluation_window(observed, self._window)
        share = dict(
            zip(
                self._classes,
                class_share(observed.window, self._classes, attribute="true_label"),
                strict=True,
            )
        )
        profiles = observed.profiles
        if not any(share.values()) or not profiles:
            return PolicyDecision(action=Action(kind="keep"), candidates=len(profiles))
        contract = observed.consumer.contract
        ranked = sorted(
            profiles, key=lambda name: (-_estimated(profiles[name], share, contract), name)
        )
        return PolicyDecision(
            action=Action(kind="select", members=tuple(ranked[: self._limit])),
            candidates=len(profiles),
        )


def _estimated(
    one: InferenceServiceProfile, share: Mapping[str, float], contract: ConsumerContract
) -> float:
    accuracy = sum(held * one.accuracy_by_class.get(name, 0.0) for name, held in share.items())
    return accuracy - _cost_of(one, contract)


def _cost_of(one: InferenceServiceProfile, contract: ConsumerContract) -> float:
    weights = sum(asked.weight for asked in contract.objectives.values())
    return contract.cost_weight / weights * mapped_cost(contract, one.cost_per_1k)


class SneakPeekSettings(ApproachSettings):
    evaluation_window: PositiveCount


def build(settings: SneakPeekSettings, *, classes: Sequence[str], seed: int) -> SneakPeek:
    return SneakPeek(
        classes=classes,
        max_ensemble_size=settings.max_ensemble_size,
        evaluation_window=settings.evaluation_window,
    )


approach = Approach(
    name=SneakPeek.name, api_version=API_VERSION, Settings=SneakPeekSettings, build=build
)
