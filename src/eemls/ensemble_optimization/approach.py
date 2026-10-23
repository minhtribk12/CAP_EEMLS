from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar, runtime_checkable

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.contract import ConsumerContract
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import (
    Action,
    EnsembleDeployment,
    EnsembleMember,
    FrozenModel,
    NonNegativeCount,
    PositiveCount,
    RuntimeContext,
    Timestamp,
)
from eemls.evaluation_engine.aggregation_functions import Value
from eemls.evaluation_engine.contract_scoring import ScoreParts
from eemls.evaluation_engine.window import Window

API_VERSION = 1


class ApproachSettings(FrozenModel):
    kind: str
    max_ensemble_size: PositiveCount
    evaluation_window: NonNegativeCount = 0


@dataclass(frozen=True, slots=True)
class ConsumerSpecification:
    contract: ConsumerContract
    cdms: CdmSet


@dataclass(frozen=True, slots=True)
class Observation:
    """`score` is None when no objective is answerable or nothing was charged."""

    consumer: ConsumerSpecification
    t: Timestamp
    context: RuntimeContext
    deployment: EnsembleDeployment
    available: tuple[EnsembleMember, ...]
    profiles: Mapping[str, InferenceServiceProfile]
    window: Window
    member_window: Window
    cdm_values: Mapping[str, Value]
    score: ScoreParts | None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: Action
    candidates: int = 0
    scored: int | None = None
    contribution_ms: float | None = None
    select_ms: float | None = None


@runtime_checkable
class Policy(Protocol):
    name: str

    def decide(self, observed: Observation) -> PolicyDecision: ...


SettingsT = TypeVar("SettingsT", bound=ApproachSettings)
TakenT = TypeVar("TakenT", bound=ApproachSettings, contravariant=True)


class PolicyFactory(Protocol[TakenT]):
    def __call__(self, settings: TakenT, /, *, classes: Sequence[str], seed: int) -> Policy: ...


@dataclass(frozen=True, slots=True)
class Approach(Generic[SettingsT]):
    name: str
    api_version: int
    Settings: type[SettingsT]
    build: PolicyFactory[SettingsT]


def rows_of_the_same_queries(member_window: Window, window: Window) -> Window:
    wanted = {str(row["query_id"]) for row in window}
    return Window([row for row in member_window if str(row["query_id"]) in wanted])


def within_evaluation_window(observed: Observation, evaluation_window: int) -> Observation:
    window = observed.window.last(evaluation_window)
    return replace(
        observed,
        window=window,
        member_window=rows_of_the_same_queries(observed.member_window, window),
    )


def window_accuracy(window: Window) -> float | None:
    labeled = [row for row in window if row.get("true_label") is not None]
    if not labeled:
        return None
    return sum(1 for row in labeled if row["predicted_label"] == row["true_label"]) / len(labeled)
