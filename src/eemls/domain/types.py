from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    computed_field,
    model_validator,
)

_V = TypeVar("_V")


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_derived_fields(cls, data: object) -> object:
        derived = set(cls.model_computed_fields)
        if not derived or not isinstance(data, dict):
            return data
        return {key: value for key, value in data.items() if key not in derived}

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        if not update:
            return super().model_copy(deep=deep)
        held = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self).model_validate({**held, **dict(update)})


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _freeze(value: Mapping[str, _V]) -> Mapping[str, _V]:
    frozen: Mapping[str, _V] = MappingProxyType(
        {key: _deep_freeze(item) for key, item in value.items()}  # type: ignore[misc]
    )
    return frozen


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain(item) for item in value]
    return value


def _thaw(value: Mapping[str, _V]) -> dict[str, object]:
    return {key: _plain(item) for key, item in value.items()}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{value.isoformat()} carries no time zone")
    return value


FrozenMap = Annotated[Mapping[str, _V], AfterValidator(_freeze), PlainSerializer(_thaw)]
Timestamp = Annotated[datetime, AfterValidator(_aware)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S(.*\S)?$")]
Text = Identifier
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Ratio = Annotated[float, Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False)]
TimeOfDay = Annotated[float, Field(ge=0.0, lt=1.0, strict=True, allow_inf_nan=False)]
Milliseconds = Annotated[float, Field(ge=0.0, strict=True, allow_inf_nan=False)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeNumber = Annotated[float, Field(ge=0.0, strict=True, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(gt=0.0, strict=True, allow_inf_nan=False)]
Score = Number
Weight = NonNegativeNumber
PositiveCount = Annotated[int, Field(gt=0, strict=True)]
NonNegativeCount = Annotated[int, Field(ge=0, strict=True)]

Row = Mapping[str, object]

Condition = Literal["day", "night"]
Weather = Literal["clear", "rain", "fog"]
ActionKind = Literal["keep", "add", "remove", "replace", "select"]
MemberStatus = Literal["ok", "timeout", "error", "refused"]
QueryOutcome = Literal["accepted", "refused", "failed"]
ResultOutcome = Literal["matched", "duplicate", "late", "unknown", "lost"]
NumberFormat = Literal["float32", "float16"]
ImageQuality = Literal["clear", "blurred"]
ProfileSource = Literal["measured", "carried_over"]

DEFAULT_CONDITION: Condition = "day"
DEFAULT_IMAGE_QUALITY: ImageQuality = "clear"


def _service_name(member: object) -> str:
    if isinstance(member, Mapping):
        return str(member.get("service_name", ""))
    return str(getattr(member, "service_name", ""))


class RuntimeContext(FrozenModel):
    period_name: Identifier
    period_index: NonNegativeCount
    condition: Condition
    weather: Weather
    event_kind: str | None = None
    event_id: str | None = None


class Query(FrozenModel):
    query_id: Identifier
    consumer_id: Identifier
    image_id: Identifier
    reply_topic: Identifier
    correlation_id: Identifier
    t_arrive: Timestamp
    dataset: str | None = None
    image_quality: ImageQuality = DEFAULT_IMAGE_QUALITY
    time_of_day: TimeOfDay | None = None


class EnsembleMember(FrozenModel):
    service_name: Identifier
    model: Identifier
    device: Identifier


class EnsembleDeployment(FrozenModel):
    deployment_id: Identifier
    members: tuple[EnsembleMember, ...]

    @model_validator(mode="before")
    @classmethod
    def _order_members_by_service_name(cls, data: object) -> object:
        if not isinstance(data, dict) or "members" not in data:
            return data
        members = data["members"]
        if not isinstance(members, Sequence) or isinstance(members, str | bytes):
            return data
        return {**data, "members": tuple(sorted(members, key=_service_name))}

    @model_validator(mode="after")
    def _members_are_a_set(self) -> EnsembleDeployment:
        if not self.members:
            raise ValueError("an ensemble version holds no member")
        names = [member.service_name for member in self.members]
        repeated = {name for name in names if names.count(name) > 1}
        if repeated:
            raise ValueError(f"members name {sorted(repeated)} more than once")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(member.service_name for member in self.members)


class Action(FrozenModel):
    kind: ActionKind
    member_in: Identifier | None = None
    member_out: Identifier | None = None
    members: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _the_action_names_what_it_needs_and_nothing_else(self) -> Action:
        needs_in = self.kind in ("add", "replace")
        needs_out = self.kind in ("remove", "replace")
        needs_members = self.kind == "select"
        if needs_in and self.member_in is None:
            raise ValueError(f"{self.kind} needs a member_in")
        if needs_out and self.member_out is None:
            raise ValueError(f"{self.kind} needs a member_out")
        if not needs_in and self.member_in is not None:
            raise ValueError(f"{self.kind} names no member_in")
        if not needs_out and self.member_out is not None:
            raise ValueError(f"{self.kind} names no member_out")
        if needs_members and not self.members:
            raise ValueError(f"{self.kind} needs the members it will serve")
        if not needs_members and self.members:
            raise ValueError(f"{self.kind} names no members")
        repeated = sorted({one for one in self.members if self.members.count(one) > 1})
        if repeated:
            raise ValueError(f"{self.kind} names {repeated} more than once")
        if self.kind == "replace" and self.member_in == self.member_out:
            raise ValueError(f"replace swaps {self.member_in} with itself")
        return self


class MemberPrediction(FrozenModel):
    query_id: Identifier
    instance_id: Identifier | None = None
    member: EnsembleMember
    node: Identifier | None = None
    replica: PositiveCount | None = None
    in_ensemble: bool
    predicted_label: Identifier | None
    confidence: Ratio | None
    fetch_ms: Milliseconds
    inference_ms: Milliseconds
    total_ms: Milliseconds
    provider: Identifier | None = None
    number_format: NumberFormat | None = None
    cache_hit: bool | None = None
    status: MemberStatus

    @model_validator(mode="after")
    def _only_an_answer_carries_a_label(self) -> MemberPrediction:
        answered = self.predicted_label is not None and self.confidence is not None
        if self.status == "ok" and not answered:
            raise ValueError("status is ok but carries no label or no confidence")
        if self.status != "ok" and answered:
            raise ValueError(f"status {self.status} did not answer but carries one")
        return self

    @model_validator(mode="after")
    def _an_answer_says_which_copy_gave_it(self) -> MemberPrediction:
        reported = {
            "instance_id": self.instance_id,
            "node": self.node,
            "replica": self.replica,
            "provider": self.provider,
            "number_format": self.number_format,
            "cache_hit": self.cache_hit,
        }
        missing = sorted(name for name, value in reported.items() if value is None)
        if self.status == "ok" and missing:
            raise ValueError(f"status is ok but the answer names no {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _the_parts_fit_inside_the_whole(self) -> MemberPrediction:
        if self.fetch_ms + self.inference_ms > self.total_ms + 1e-6:
            raise ValueError(
                f"the fetch and the run take {self.fetch_ms + self.inference_ms} ms, "
                f"longer than the {self.total_ms} ms they sit in"
            )
        if self.status == "ok" and self.inference_ms <= 0.0:
            raise ValueError("status is ok but the model took no time to run")
        return self


class AggregatedResult(FrozenModel):
    query_id: Identifier
    predicted_label: Identifier
    confidence: Ratio
    deployment_id: Identifier
    n_members: PositiveCount
    n_answered: PositiveCount

    @model_validator(mode="after")
    def _no_more_answered_than_served(self) -> AggregatedResult:
        if self.n_answered > self.n_members:
            raise ValueError(
                f"{self.n_answered} of {self.n_members} members answered, which is more than serve"
            )
        return self
