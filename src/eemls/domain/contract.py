from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from eemls.domain.types import (
    FrozenMap,
    FrozenModel,
    Identifier,
    Number,
    PositiveCount,
    PositiveNumber,
    Ratio,
    Text,
    Timestamp,
    Weight,
)

Direction = Literal["maximize", "minimize"]
ValueMapping = Literal["linear", "log"]
Operator = Literal[">=", "<=", ">", "<", "=="]


class FeedbackSchedule(FrozenModel):
    fraction: Ratio
    delay_intervals: PositiveCount


class Objective(FrozenModel):
    weight: PositiveNumber
    min_value: Number = 0.0
    max_value: Number = 1.0
    direction: Direction = "maximize"
    mapping: ValueMapping = "linear"
    mapping_base: PositiveNumber | None = None

    @model_validator(mode="after")
    def _the_range_and_the_scale_agree(self) -> Objective:
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value {self.min_value} is not below max_value {self.max_value}")
        if self.mapping == "linear" and self.mapping_base is not None:
            raise ValueError(
                "mapping_base belongs to a log mapping; direction states which way linear goes"
            )
        if self.mapping == "log" and self.mapping_base is None:
            raise ValueError("a log mapping needs its mapping_base")
        if self.mapping_base is not None and self.mapping_base <= 1.0:
            raise ValueError(f"a log base of {self.mapping_base} is not above one")
        return self


class Requirement(FrozenModel):
    """A soft requirement on a CDM: a miss is a violation, never a refused query."""

    cdm: Identifier
    operator: Operator
    target_value: Number


class ConsumerContract(FrozenModel):
    consumer_id: Identifier
    description: Text
    prioritized_classes: tuple[str, ...]
    request_rate_per_s: PositiveNumber
    objectives: FrozenMap[Objective]
    cost_weight: Weight
    cost_range_max_per_1k: PositiveNumber
    feedback: FeedbackSchedule
    requirements: tuple[Requirement, ...] = ()
    valid_from: Timestamp | None = None
    valid_until: Timestamp | None = None

    @model_validator(mode="after")
    def _the_contract_is_scorable_and_dated(self) -> ConsumerContract:
        if not self.objectives:
            raise ValueError("objectives is empty")
        classes = list(self.prioritized_classes)
        repeated = {name for name in classes if classes.count(name) > 1}
        if repeated:
            raise ValueError(f"prioritized_classes names {sorted(repeated)} more than once")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("the contract ends before it starts")
        return self

    def scored(self) -> set[str]:
        return set(self.objectives)
