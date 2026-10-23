from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import model_validator

from eemls.domain.types import (
    Condition,
    FrozenMap,
    FrozenModel,
    Identifier,
    ImageQuality,
    NonNegativeNumber,
    PositiveNumber,
    Weather,
)

PeriodEvent = Literal["period_change", "class_burst"]


class Period(FrozenModel):
    name: Identifier
    starts_at_s: NonNegativeNumber
    class_mix: FrozenMap[PositiveNumber]
    condition: Condition
    weather: Weather
    event: PeriodEvent | None = None

    @model_validator(mode="after")
    def _the_period_asks_about_something(self) -> Period:
        if not self.class_mix:
            raise ValueError(f"period {self.name} names no class to ask about")
        return self


class ImageQualityEvent(FrozenModel):
    starts_at_s: NonNegativeNumber
    ends_at_s: NonNegativeNumber
    image_quality: ImageQuality = "blurred"

    @model_validator(mode="after")
    def _the_event_lasts(self) -> ImageQualityEvent:
        if self.ends_at_s <= self.starts_at_s:
            raise ValueError(f"an event from {self.starts_at_s} to {self.ends_at_s} has no length")
        return self


class QuerySurge(FrozenModel):
    starts_at_s: NonNegativeNumber
    ends_at_s: NonNegativeNumber
    multiplier: PositiveNumber

    @model_validator(mode="after")
    def _the_surge_lasts_and_raises_the_rate(self) -> QuerySurge:
        if self.ends_at_s <= self.starts_at_s:
            raise ValueError(f"a surge from {self.starts_at_s} to {self.ends_at_s} has no length")
        if self.multiplier <= 1.0:
            raise ValueError(f"a surge of {self.multiplier} times does not raise the rate")
        return self


class Schedule(FrozenModel):
    request_rate_per_s: PositiveNumber
    periods: tuple[Period, ...]
    day_length_s: PositiveNumber
    events: tuple[ImageQualityEvent, ...] = ()
    surges: tuple[QuerySurge, ...] = ()

    @model_validator(mode="after")
    def _the_periods_cover_the_run_in_order(self) -> Schedule:
        if not self.periods:
            raise ValueError("a schedule holds no period")
        if self.periods[0].starts_at_s != 0.0:
            raise ValueError(
                f"the first period starts at {self.periods[0].starts_at_s}, and a run starts at 0"
            )
        starts = [period.starts_at_s for period in self.periods]
        if starts != sorted(set(starts)):
            raise ValueError(f"the periods start at {starts}, which is not a rising order")
        return self

    @model_validator(mode="after")
    def _no_two_events_of_one_kind_cover_one_moment(self) -> Schedule:
        _apart(self.events)
        _apart(self.surges)
        return self

    def period_at(self, elapsed_s: float) -> Period:
        held = self.periods[0]
        for period in self.periods:
            if period.starts_at_s > elapsed_s:
                break
            held = period
        return held

    def time_of_day_at(self, elapsed_s: float) -> float:
        return min((elapsed_s % self.day_length_s) / self.day_length_s, math.nextafter(1.0, 0.0))

    def image_quality_at(self, elapsed_s: float) -> ImageQuality:
        for event in self.events:
            if event.starts_at_s <= elapsed_s < event.ends_at_s:
                return event.image_quality
        return "clear"

    def due_by(self, elapsed_s: float) -> int:
        surged_s = sum(
            (surge.multiplier - 1.0) * max(0.0, min(surge.ends_at_s, elapsed_s) - surge.starts_at_s)
            for surge in self.surges
        )
        return math.floor(self.request_rate_per_s * (elapsed_s + surged_s))

    def planned_at(self, query: int) -> float:
        wanted = query / self.request_rate_per_s
        at = 0.0
        reached = 0.0
        for surge in sorted(self.surges, key=lambda one: one.starts_at_s):
            if reached + (surge.starts_at_s - at) >= wanted:
                break
            reached += surge.starts_at_s - at
            at = surge.starts_at_s
            within = (surge.ends_at_s - surge.starts_at_s) * surge.multiplier
            if reached + within >= wanted:
                return at + (wanted - reached) / surge.multiplier
            reached += within
            at = surge.ends_at_s
        return at + (wanted - reached)


def _apart(windows: Sequence[ImageQualityEvent | QuerySurge]) -> None:
    held = sorted(windows, key=lambda window: window.starts_at_s)
    for earlier, later in itertools.pairwise(held):
        if later.starts_at_s < earlier.ends_at_s:
            raise ValueError(
                f"an event from {later.starts_at_s} overlaps the one ending at {earlier.ends_at_s}"
            )


def sampled_label(class_mix: Mapping[str, float], draw: float) -> str:
    ordered = sorted(class_mix.items())
    total = sum(weight for _, weight in ordered)
    reached = draw * total
    running = 0.0
    for label, weight in ordered:
        running += weight
        if reached < running:
            return label
    return ordered[-1][0]
