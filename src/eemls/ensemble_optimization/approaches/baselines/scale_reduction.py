from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime

from eemls.domain.measuring import milliseconds_since
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import (
    Action,
    FrozenModel,
    NonNegativeCount,
    PositiveCount,
    Score,
)
from eemls.ensemble_optimization.approach import (
    API_VERSION,
    Approach,
    ApproachSettings,
    ConsumerSpecification,
    Observation,
    PolicyDecision,
    within_evaluation_window,
)
from eemls.evaluation_engine.aggregation_functions import Value
from eemls.evaluation_engine.contract_scoring import NotScored, score
from eemls.evaluation_engine.ensemble_scoring import MemberRow, aggregated_window, scored_window
from eemls.evaluation_engine.profile_estimates import estimated_member_rows, member_rows
from eemls.evaluation_engine.window import Window

logger = logging.getLogger(__name__)


class SearchSpace(FrozenModel):
    ensembles: NonNegativeCount
    naive: NonNegativeCount
    scale_reduction: NonNegativeCount


class ScoredDeployment(FrozenModel):
    members: tuple[str, ...]
    total: Score


def scales_for(rate_per_s: float, throughputs: Sequence[float]) -> tuple[int, ...]:
    return tuple(math.ceil(rate_per_s / throughput) for throughput in throughputs)


def reduced_scales(scales: Sequence[int]) -> tuple[int, ...]:
    smallest = min(scales)
    return tuple(math.ceil(scale / smallest) for scale in scales)


def search_space(
    throughputs: Mapping[str, Sequence[float]],
    *,
    rate_per_s: float,
    max_ensemble_size: int,
    device_types: int | None = None,
) -> SearchSpace:
    models = sorted(throughputs)
    placements = {
        name: device_types if device_types is not None else len(throughputs[name])
        for name in models
    }
    ensembles = naive = reduced = 0
    for ensemble in _ensembles(models, max_ensemble_size):
        scales = scales_for(rate_per_s, [min(throughputs[name]) for name in ensemble])
        ensembles += 1
        naive += _placed(ensemble, scales, placements)
        reduced += _placed(ensemble, reduced_scales(scales), placements)
    return SearchSpace(ensembles=ensembles, naive=naive, scale_reduction=reduced)


def scaled_up(
    placed: Mapping[str, int],
    *,
    by: int,
    profiles: Mapping[str, InferenceServiceProfile],
    rate_per_s: float,
) -> dict[str, int]:
    replicas = {name: count * by for name, count in placed.items()}
    carried = sum(profiles[name].throughput_rps * count for name, count in replicas.items())
    for name in sorted(replicas, key=lambda held: (-_replica_cost(profiles[held]), held)):
        throughput = profiles[name].throughput_rps
        while replicas[name] > 0 and carried - throughput >= rate_per_s:
            replicas[name] -= 1
            carried -= throughput
    return {name: count for name, count in sorted(replicas.items()) if count > 0}


class ScaleReduction:
    name = "scale_reduction"

    def __init__(self, *, max_ensemble_size: int, evaluation_window: int) -> None:
        self._limit = max_ensemble_size
        self._window = evaluation_window

    def decide(self, observed: Observation) -> PolicyDecision:
        started = time.perf_counter()
        observed = within_evaluation_window(observed, self._window)
        profiles = _scalable(observed)
        by_model = _by_model(profiles)
        rate_per_s = request_rate(observed)
        throughputs = {
            model: [profiles[name].throughput_rps for name in names]
            for model, names in by_model.items()
        }
        space = search_space(throughputs, rate_per_s=rate_per_s, max_ensemble_size=self._limit)
        best: ScoredDeployment | None = None
        scored = 0
        for one in _scored_deployments(observed, profiles, by_model, rate_per_s, self._limit):
            scored += 1
            if best is None or one.total > best.total:
                best = one
        select_ms = milliseconds_since(started)
        logger.info(
            f"stage_tag=select process_tag=scale_reduction "
            f"consumer={observed.consumer.contract.consumer_id} "
            f"candidates={space.scale_reduction} scored={scored} select_ms={select_ms:.3f}"
        )
        if best is None:
            return PolicyDecision(
                action=Action(kind="keep"),
                candidates=space.scale_reduction,
                scored=scored,
                select_ms=select_ms,
            )
        return PolicyDecision(
            action=Action(kind="select", members=best.members),
            candidates=space.scale_reduction,
            scored=scored,
            select_ms=select_ms,
        )


def _scored_deployments(
    observed: Observation,
    profiles: Mapping[str, InferenceServiceProfile],
    by_model: Mapping[str, Sequence[str]],
    rate_per_s: float,
    limit: int,
) -> Iterator[ScoredDeployment]:
    rows = labeled_rows(observed, profiles)
    if rows:
        yield from _scored(
            observed.consumer, rows, profiles, by_model, rate_per_s, limit, observed.window
        )


def members_of(
    replicas: Mapping[str, int], profiles: Mapping[str, InferenceServiceProfile]
) -> tuple[str, ...]:
    chosen: dict[str, str] = {}
    for name in sorted(
        replicas, key=lambda held: (-replicas[held], profiles[held].cost_per_1k, held)
    ):
        chosen.setdefault(profiles[name].model, name)
    return tuple(sorted(chosen.values()))


def request_rate(observed: Observation) -> float:
    arrivals = sorted(
        held for row in observed.window if isinstance(held := row.get("t_arrive"), datetime)
    )
    if len(arrivals) > 1:
        span = (arrivals[-1] - arrivals[0]).total_seconds()
        if span > 0.0:
            return (len(arrivals) - 1) / span
    return observed.consumer.contract.request_rate_per_s


def _scalable(observed: Observation) -> dict[str, InferenceServiceProfile]:
    return {name: held for name, held in observed.profiles.items() if held.throughput_rps > 0.0}


def _ensembles(models: Sequence[str], limit: int) -> Iterator[tuple[str, ...]]:
    for size in range(1, min(limit, len(models)) + 1):
        yield from itertools.combinations(models, size)


def _placed(ensemble: Sequence[str], scales: Sequence[int], placements: Mapping[str, int]) -> int:
    return math.prod(
        placements[name] ** scale for name, scale in zip(ensemble, scales, strict=True)
    )


def _replica_cost(held: InferenceServiceProfile) -> float:
    return held.cost_per_1k * held.throughput_rps


def _by_model(profiles: Mapping[str, InferenceServiceProfile]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for name in sorted(profiles):
        grouped.setdefault(profiles[name].model, []).append(name)
    return {model: tuple(names) for model, names in grouped.items()}


def _splits(total: int, names: Sequence[str]) -> Iterator[dict[str, int]]:
    for cuts in itertools.combinations_with_replacement(range(total + 1), len(names) - 1):
        bounds = (0, *cuts, total)
        yield {
            name: bounds[index + 1] - bounds[index]
            for index, name in enumerate(names)
            if bounds[index + 1] > bounds[index]
        }


def _deployments(
    ensemble: Sequence[str],
    by_model: Mapping[str, Sequence[str]],
    profiles: Mapping[str, InferenceServiceProfile],
    rate_per_s: float,
) -> Iterator[dict[str, int]]:
    worst = [min(profiles[name].throughput_rps for name in by_model[model]) for model in ensemble]
    scales = scales_for(rate_per_s, worst)
    smallest = min(scales)
    per_model = [
        [
            scaled_up(split, by=smallest, profiles=profiles, rate_per_s=rate_per_s)
            for split in _splits(reduced, by_model[model])
        ]
        for model, reduced in zip(ensemble, reduced_scales(scales), strict=True)
    ]
    for chosen in itertools.product(*per_model):
        yield {name: count for one in chosen for name, count in one.items()}


def _scored(
    consumer: ConsumerSpecification,
    rows: Sequence[MemberRow],
    profiles: Mapping[str, InferenceServiceProfile],
    by_model: Mapping[str, Sequence[str]],
    rate_per_s: float,
    limit: int,
    window: Window,
) -> Iterator[ScoredDeployment]:
    answered = {row.service_name for row in rows}
    values: dict[tuple[str, ...], Mapping[str, Value]] = {}
    for ensemble in _ensembles(sorted(by_model), limit):
        for replicas in _deployments(ensemble, by_model, profiles, rate_per_s):
            members = members_of(replicas, profiles)
            if not set(replicas) <= answered:
                continue
            if members not in values:
                values[members], _ = scored_window(
                    consumer.contract,
                    consumer.cdms,
                    aggregated_window(rows, members, window=window),
                )
            cost = (
                sum(count * _replica_cost(profiles[name]) for name, count in replicas.items())
                / rate_per_s
            )
            parts = score(consumer.contract, values[members], cost_per_1k=cost)
            if not isinstance(parts, NotScored):
                yield ScoredDeployment(members=members, total=parts.total)


def labeled_rows(
    observed: Observation, profiles: Mapping[str, InferenceServiceProfile]
) -> tuple[MemberRow, ...]:
    labeled = member_rows(observed.member_window.labeled("true_label"), profiles)
    return tuple(
        row
        for name in sorted(profiles)
        for row in estimated_member_rows(profiles[name], labeled=labeled)
    )


class ScaleReductionSettings(ApproachSettings):
    evaluation_window: PositiveCount


def build(settings: ScaleReductionSettings, *, classes: Sequence[str], seed: int) -> ScaleReduction:
    return ScaleReduction(
        max_ensemble_size=settings.max_ensemble_size,
        evaluation_window=settings.evaluation_window,
    )


approach = Approach(
    name=ScaleReduction.name, api_version=API_VERSION, Settings=ScaleReductionSettings, build=build
)
