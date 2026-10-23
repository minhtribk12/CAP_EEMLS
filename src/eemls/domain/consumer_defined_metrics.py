from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import Field, model_validator

from eemls.domain.types import (
    FrozenMap,
    FrozenModel,
    Identifier,
    PositiveCount,
    Text,
    Weight,
)

NumberFunction = Literal["sum", "average", "min", "max", "percentile"]
Function = Literal[NumberFunction, "count", "ratio", "weighted_count", "reference"]
Measured = Literal["per_query", "per_window"]

READS_AN_ATTRIBUTE: tuple[str, ...] = get_args(NumberFunction)
Percentile = Annotated[float, Field(gt=0.0, le=100.0, strict=True, allow_inf_nan=False)]


class Scope(FrozenModel):
    attribute: Identifier
    value: str | float | int | None = None
    same_as: str | None = None
    one_of: tuple[str | float | int, ...] | None = None
    observed: bool | None = None

    @model_validator(mode="after")
    def _exactly_one_statement(self) -> Scope:
        given = [
            name
            for name, value in (
                ("value", self.value),
                ("same_as", self.same_as),
                ("one_of", self.one_of),
                ("observed", self.observed),
            )
            if value is not None
        ]
        if not given:
            raise ValueError(f"the context on {self.attribute} states nothing")
        if len(given) > 1:
            raise ValueError(f"the context on {self.attribute} states more than one thing: {given}")
        if self.one_of is not None and not self.one_of:
            raise ValueError(f"the context on {self.attribute} allows no value")
        if self.same_as == self.attribute:
            raise ValueError(f"the context on {self.attribute} compares it with itself")
        return self

    def attributes(self) -> set[str]:
        return {self.attribute} | ({self.same_as} if self.same_as else set())


class MetricSpecification(FrozenModel):
    function: Function
    attribute: str | None = None
    scopes: tuple[Scope, ...] = ()
    inputs: tuple[MetricSpecification, ...] = ()
    group_by: str | None = None
    weights: FrozenMap[Weight] | None = None
    percentile_value: Percentile | None = None
    cdm: str | None = None

    @model_validator(mode="after")
    def _the_function_gets_what_it_needs_and_nothing_else(self) -> MetricSpecification:
        self._check_attribute()
        self._check_inputs()
        self._check_percentile()
        self._check_weights()
        self._check_reference()
        return self

    def _check_attribute(self) -> None:
        if self.function in READS_AN_ATTRIBUTE and not self.attribute:
            raise ValueError(f"{self.function} needs an attribute to read")
        if self.function in ("ratio", "reference") and self.attribute:
            raise ValueError(f"{self.function} reads no attribute of its own")

    def _check_inputs(self) -> None:
        if self.function == "ratio" and len(self.inputs) != 2:
            raise ValueError(f"ratio takes exactly two inputs, got {len(self.inputs)}")
        if self.function != "ratio" and self.inputs:
            raise ValueError(f"{self.function} takes no inputs")

    def _check_percentile(self) -> None:
        if self.function == "percentile" and self.percentile_value is None:
            raise ValueError("percentile needs its percentile_value")
        if self.function != "percentile" and self.percentile_value is not None:
            raise ValueError(f"{self.function} takes no percentile_value")

    def _check_weights(self) -> None:
        if self.function == "weighted_count":
            if self.weights is None:
                raise ValueError("weighted_count needs its weights")
            if not self.group_by:
                raise ValueError("weighted_count needs the attribute to group_by")
        elif self.weights is not None:
            raise ValueError(f"{self.function} takes no weights")

    def _check_reference(self) -> None:
        if self.function == "reference" and not self.cdm:
            raise ValueError("reference names no cdm")
        if self.function != "reference" and self.cdm:
            raise ValueError(f"{self.function} names no cdm")

    def answers_one_value_per_group(self) -> bool:
        return bool(self.group_by) and self.function != "weighted_count"

    def attributes(self) -> set[str]:
        found: set[str] = {self.attribute} if self.attribute else set()
        if self.group_by:
            found.add(self.group_by)
        for one in self.scopes:
            found |= one.attributes()
        for inner in self.inputs:
            found |= inner.attributes()
        return found

    def references(self) -> set[str]:
        found: set[str] = {self.cdm} if self.cdm else set()
        for inner in self.inputs:
            found |= inner.references()
        return found


class CdmDefinition(FrozenModel):
    name: Identifier
    measured: Measured
    window: PositiveCount | None
    unit: Identifier
    description: Text
    specification: MetricSpecification

    @model_validator(mode="after")
    def _a_period_metric_has_a_window(self) -> CdmDefinition:
        if self.measured == "per_window" and self.window is None:
            raise ValueError("measured is per_window but window is missing")
        return self

    def attributes(self) -> set[str]:
        return self.specification.attributes()

    def references(self) -> set[str]:
        return self.specification.references()


class CdmSet(FrozenModel):
    definitions: FrozenMap[CdmDefinition]

    @model_validator(mode="before")
    @classmethod
    def _each_definition_takes_its_name_from_its_key(cls, data: object) -> object:
        if not isinstance(data, dict) or not isinstance(data.get("definitions"), dict):
            return data
        named = {}
        for key, entry in data["definitions"].items():
            if isinstance(entry, dict) and entry.get("name", key) != key:
                raise ValueError(f"the metric under {key} calls itself {entry['name']}")
            named[key] = {**entry, "name": key} if isinstance(entry, dict) else entry
        return {**data, "definitions": named}

    @model_validator(mode="after")
    def _at_least_one_definition(self) -> CdmSet:
        if not self.definitions:
            raise ValueError("definitions is empty")
        return self

    @model_validator(mode="after")
    def _every_reference_names_a_declared_metric(self) -> CdmSet:
        for name, definition in self.definitions.items():
            unknown = sorted(definition.references() - set(self.definitions))
            if unknown:
                raise ValueError(f"{name} refers to metrics nothing declares: {unknown}")
        return self

    def attributes(self) -> set[str]:
        return {
            attribute
            for definition in self.definitions.values()
            for attribute in definition.attributes()
        }
