from __future__ import annotations

import math
from collections.abc import Mapping

from eemls.domain.contract import ConsumerContract, Objective
from eemls.domain.errors import EemlsError
from eemls.domain.statistics import as_number, is_number
from eemls.domain.types import FrozenMap, FrozenModel, NonNegativeNumber, Score


class ScoreError(EemlsError): ...


class NotScored(FrozenModel):
    missing: tuple[str, ...]
    reason: str


class ScoreParts(FrozenModel):
    total: Score
    quality: Score
    cost: NonNegativeNumber
    by_objective: FrozenMap[float]


def map_to_range(value: float, objective: Objective) -> float:
    asked = objective
    width = asked.max_value - asked.min_value
    share = min(max((value - asked.min_value) / width, 0.0), 1.0)
    scaled = _scaled(share, asked)
    return 1.0 - scaled if asked.direction == "minimize" else scaled


def _scaled(share: float, objective: Objective) -> float:
    if objective.mapping == "linear":
        return share
    base = objective.mapping_base
    if base is None:
        raise ScoreError(f"a {objective.mapping} objective reached the scorer with no base")
    return math.log(1.0 + (base - 1.0) * share, base)


def score(
    contract: ConsumerContract,
    values: Mapping[str, object],
    *,
    cost_per_1k: float,
) -> ScoreParts | NotScored:
    if cost_per_1k < 0:
        raise ScoreError(f"a cost of {cost_per_1k} per 1,000 queries is not a cost")
    missing = tuple(name for name in contract.objectives if not is_number(values.get(name)))
    for name in missing:
        held = values.get(name)
        if isinstance(held, Mapping):
            raise ScoreError(f"{name} came back per class, so it cannot be scored as one number")
    if len(missing) == len(contract.objectives):
        return NotScored(missing=missing, reason=f"the window could not answer {list(missing)}")
    answered = {name: asked for name, asked in contract.objectives.items() if name not in missing}
    by_objective = {
        name: asked.weight * map_to_range(as_number(values[name]), asked)
        for name, asked in answered.items()
    }
    quality = sum(by_objective.values())
    cost = contract.cost_weight * mapped_cost(contract, cost_per_1k)
    return ScoreParts(total=quality - cost, quality=quality, cost=cost, by_objective=by_objective)


def mapped_cost(contract: ConsumerContract, cost_per_1k: float) -> float:
    return min(cost_per_1k / contract.cost_range_max_per_1k, 1.0)
