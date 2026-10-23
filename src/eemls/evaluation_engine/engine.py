from __future__ import annotations

from collections.abc import Mapping

from eemls.domain.consumer_defined_metrics import CdmDefinition, CdmSet, MetricSpecification
from eemls.evaluation_engine.aggregation_functions import (
    CdmError,
    Undefined,
    Value,
    aggregate,
    rows_in_scope,
)
from eemls.evaluation_engine.window import Window


class EvaluationEngine:
    def __init__(self, declared: CdmSet) -> None:
        self.declared: Mapping[str, CdmDefinition] = declared.definitions

    def of(self, name: str, window: Window) -> Value:
        return self._named(name, window, ())

    def _named(self, name: str, window: Window, seen: tuple[str, ...]) -> Value:
        definition = self.declared.get(name)
        if definition is None:
            raise CdmError(f"{name} is declared by nothing")
        if name in seen:
            raise CdmError(f"{name} is built from itself: {' -> '.join((*seen, name))}")
        narrowed = window.last(definition.window) if definition.window else window
        return self._value(definition.specification, narrowed, (*seen, name))

    def _value(
        self, specification: MetricSpecification, window: Window, seen: tuple[str, ...]
    ) -> Value:
        if specification.answers_one_value_per_group():
            return self._grouped(specification, window, seen)
        if specification.function == "reference":
            return self._named(
                str(specification.cdm), rows_in_scope(window, specification.scopes), seen
            )
        if specification.function == "ratio":
            return self._ratio(specification, window, seen)
        return aggregate(specification, window)

    def _ratio(
        self, specification: MetricSpecification, window: Window, seen: tuple[str, ...]
    ) -> Value:
        rows = rows_in_scope(window, specification.scopes)
        top = self._value(specification.inputs[0], rows, seen)
        bottom = self._value(specification.inputs[1], rows, seen)
        return divide_values(top, bottom)

    def _grouped(
        self, specification: MetricSpecification, window: Window, seen: tuple[str, ...]
    ) -> Value:
        group_by = str(specification.group_by)
        rows = rows_in_scope(window, specification.scopes)
        held = [row.get(group_by) for row in rows]
        wrong = sorted(
            {type(one).__name__ for one in held if one is not None and not isinstance(one, str)}
        )
        if wrong:
            raise CdmError(f"{group_by} holds {wrong}, which cannot name a group")
        groups = {one for one in held if isinstance(one, str)}
        ungrouped = specification.model_copy(update={"group_by": None})
        values: dict[str, float] = {}
        for group in sorted(groups):
            inside = Window(tuple(row for row in rows if row.get(group_by) == group))
            value = self._value(ungrouped, inside, seen)
            if isinstance(value, Undefined):
                continue
            if isinstance(value, Mapping):
                raise CdmError(
                    f"{specification.function} cannot group a value that is already grouped"
                )
            values[group] = float(value)
        return values


def divide_values(top: Value, bottom: Value) -> Value:
    if isinstance(top, Undefined):
        return top
    if isinstance(bottom, Undefined):
        return bottom
    if isinstance(top, Mapping):
        raise CdmError("a ratio cannot divide a grouped numerator")
    if isinstance(bottom, Mapping):
        raise CdmError("a ratio cannot divide by a grouped denominator")
    if bottom == 0:
        return Undefined("the denominator covers no row, so the ratio has no value")
    return float(top) / float(bottom)
