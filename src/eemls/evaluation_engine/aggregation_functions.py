from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from eemls.domain.consumer_defined_metrics import MetricSpecification, Scope
from eemls.domain.errors import EemlsError
from eemls.domain.statistics import is_number, percentile
from eemls.domain.types import Row
from eemls.evaluation_engine.window import Window


class CdmError(EemlsError): ...


class Undefined:
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return f"Undefined({self.reason!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Undefined) and other.reason == self.reason

    def __hash__(self) -> int:
        return hash(("undefined", self.reason))


Value = float | int | Mapping[str, float] | Undefined


def in_scope(row: Row, one: Scope) -> bool:
    held = row.get(one.attribute)
    if one.observed is not None:
        return (held is not None) == one.observed
    if held is None:
        return False
    if one.value is not None:
        return bool(held == one.value)
    if one.one_of is not None:
        return held in one.one_of
    return bool(held == row.get(str(one.same_as)))


def rows_in_scope(window: Window, scopes: Sequence[Scope]) -> Window:
    if not scopes:
        return window
    return Window(tuple(row for row in window if all(in_scope(row, one) for one in scopes)))


def numeric_values(window: Window, attribute: str) -> list[float]:
    found: list[float] = []
    for row in window:
        held = row.get(attribute)
        if held is None:
            continue
        if not is_number(held):
            raise CdmError(f"{attribute} holds {held!r}, which is not a number")
        found.append(float(held))
    return found


def weighted_count(window: Window, group_by: str, weights: Mapping[str, float]) -> float:
    total = 0.0
    for row in window:
        group = row.get(group_by)
        if isinstance(group, str):
            total += weights.get(group, 0.0)
    return total


def _percentile(values: Sequence[float], specification: MetricSpecification) -> float:
    if specification.percentile_value is None:
        raise CdmError("a percentile reached the evaluator with no percentile_value")
    return percentile(values, specification.percentile_value)


OF_NUMBERS: Mapping[str, Callable[[Sequence[float], MetricSpecification], float]] = (
    MappingProxyType(
        {
            "sum": lambda values, _: sum(values),
            "average": lambda values, _: sum(values) / len(values),
            "min": lambda values, _: min(values),
            "max": lambda values, _: max(values),
            "percentile": _percentile,
        }
    )
)


def aggregate(specification: MetricSpecification, window: Window) -> Value:
    rows = rows_in_scope(window, specification.scopes)
    if specification.function == "count":
        return len(rows)
    if specification.function == "weighted_count":
        return weighted_count(rows, str(specification.group_by), specification.weights or {})
    of = OF_NUMBERS.get(specification.function)
    if of is None:
        raise CdmError(f"{specification.function} is not a function of the evaluator")
    values = numeric_values(rows, str(specification.attribute))
    if not values:
        return Undefined(
            f"{specification.function} of {specification.attribute} has no row to read"
        )
    return of(values, specification)
