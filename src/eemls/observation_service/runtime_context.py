from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from eemls.domain.types import Condition, Row, RuntimeContext, Weather
from eemls.evaluation_engine.window import Window
from eemls.observation_service.operating_conditions import CONDITIONS

UNNAMED = "unnamed"


@runtime_checkable
class ServingContext(Protocol):
    def __call__(self, consumer_id: str) -> RuntimeContext: ...

    def read_from(self, consumer_id: str, window: Window) -> None: ...


@dataclass(slots=True)
class _Read:
    name: str = UNNAMED
    index: int = 0
    condition: Condition = "day"


class RuntimeContextReader:
    def __init__(self, *, dominant_classes: Sequence[str], weather: Weather = "clear") -> None:
        self._dominant = tuple(dominant_classes)
        self._weather = weather
        self._read: dict[str, _Read] = {}

    def __call__(self, consumer_id: str) -> RuntimeContext:
        held = self._read.get(consumer_id, _Read())
        return RuntimeContext(
            period_name=held.name,
            period_index=held.index,
            condition=held.condition,
            weather=self._weather,
        )

    def read_from(self, consumer_id: str, window: Window) -> None:
        held = self._read.setdefault(consumer_id, _Read())
        self._name_the_period(held, window)
        _take_the_condition(held, window)

    def _name_the_period(self, held: _Read, window: Window) -> None:
        answered = [str(row["predicted_label"]) for row in window if row.get("predicted_label")]
        counted = {name: answered.count(name) for name in self._dominant}
        highest = max(counted.values(), default=0)
        leading = [name for name, count in counted.items() if count == highest]
        if highest == 0 or len(leading) != 1:
            return
        held.name = leading[0]
        held.index = self._dominant.index(leading[0]) + 1


def _take_the_condition(held: _Read, window: Window) -> None:
    for row in reversed(window):
        condition = _condition_of(row)
        if condition is not None:
            held.condition = condition
            return


def _condition_of(row: Row) -> Condition | None:
    held = row.get("condition")
    return next((one for one in CONDITIONS if one == held), None)
