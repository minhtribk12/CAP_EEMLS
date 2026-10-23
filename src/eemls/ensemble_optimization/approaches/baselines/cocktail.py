from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Action, PositiveCount, Ratio, Row
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


class Cocktail:
    name = "cocktail"

    def __init__(
        self,
        *,
        max_ensemble_size: int,
        evaluation_window: int,
        accuracy_target: float,
        accuracy_target_range: float,
    ) -> None:
        self._limit = max_ensemble_size
        self._window = evaluation_window
        self._target = accuracy_target
        self._range = accuracy_target_range

    def decide(self, observed: Observation) -> PolicyDecision:
        observed = within_evaluation_window(observed, self._window)
        accuracy = window_accuracy(observed.window)
        serving = observed.deployment.service_names
        if accuracy is None:
            return PolicyDecision(action=Action(kind="keep"))
        if accuracy < self._target:
            return PolicyDecision(action=_grown(serving, observed.profiles, self._limit))
        if accuracy > self._target + self._range:
            return PolicyDecision(
                action=_shrunk(serving, observed.profiles, observed.member_window)
            )
        return PolicyDecision(action=Action(kind="keep"))


def _grown(
    serving: Sequence[str], profiles: Mapping[str, InferenceServiceProfile], limit: int
) -> Action:
    if len(serving) >= limit:
        return Action(kind="keep")
    outside = sorted(name for name in profiles if name not in set(serving))
    if not outside:
        return Action(kind="keep")
    best = max(outside, key=lambda name: profiles[name].general_accuracy)
    return Action(kind="add", member_in=best)


def _shrunk(
    serving: Sequence[str],
    profiles: Mapping[str, InferenceServiceProfile],
    member_window: Window,
) -> Action:
    majority = len(serving) // 2 + 1
    if len(serving) <= majority or _vote_mode(serving, member_window) <= majority:
        return Action(kind="keep")
    accuracy = _member_accuracy(serving, member_window)
    profiled = {
        name: profiles[name].general_accuracy if name in profiles else -1.0 for name in serving
    }
    ranked = sorted(
        serving,
        key=lambda name: (accuracy.get(name, profiled[name]), profiled[name], name),
    )
    dropped = ranked[: len(serving) - majority]
    if len(dropped) == 1:
        return Action(kind="remove", member_out=dropped[0])
    return Action(kind="select", members=tuple(name for name in serving if name not in dropped))


def _answers(serving: Sequence[str], member_window: Window) -> list[Row]:
    members = set(serving)
    return [
        row
        for row in member_window
        if row.get("service_name") in members and row.get("predicted_label") is not None
    ]


def _vote_mode(serving: Sequence[str], member_window: Window) -> int:
    by_query: dict[str, Counter[str]] = {}
    for row in _answers(serving, member_window):
        by_query.setdefault(str(row["query_id"]), Counter())[str(row["predicted_label"])] += 1
    if not by_query:
        return 0
    counts = Counter(max(votes.values()) for votes in by_query.values())
    return max(counts, key=lambda votes: (counts[votes], -votes))


def _member_accuracy(serving: Sequence[str], member_window: Window) -> dict[str, float]:
    right: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    for row in _answers(serving, member_window):
        if row.get("true_label") is None:
            continue
        name = str(row["service_name"])
        seen[name] += 1
        right[name] += row["predicted_label"] == row["true_label"]
    return {name: right[name] / seen[name] for name in seen}


class CocktailSettings(ApproachSettings):
    evaluation_window: PositiveCount
    accuracy_target: Ratio
    accuracy_target_range: Ratio


def build(settings: CocktailSettings, *, classes: Sequence[str], seed: int) -> Cocktail:
    return Cocktail(
        max_ensemble_size=settings.max_ensemble_size,
        evaluation_window=settings.evaluation_window,
        accuracy_target=settings.accuracy_target,
        accuracy_target_range=settings.accuracy_target_range,
    )


approach = Approach(
    name=Cocktail.name, api_version=API_VERSION, Settings=CocktailSettings, build=build
)
