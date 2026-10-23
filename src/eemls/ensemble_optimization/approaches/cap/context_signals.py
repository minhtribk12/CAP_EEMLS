from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

from eemls.domain.records import InferenceServiceProfile
from eemls.domain.statistics import as_number
from eemls.domain.types import Row
from eemls.evaluation_engine.window import Window, class_share


@dataclass(frozen=True, slots=True)
class ConsumerSignals:
    class_share: tuple[float, ...]
    class_share_change: tuple[float, ...]
    time_of_day: float
    mean_confidence: float
    n_members: float


@dataclass(frozen=True, slots=True)
class ServiceSignals:
    strong_class_share: float
    strong_class_share_change: float
    weighted_disagreement: float
    strong_class_share_labeled: float
    period_contribution: float
    condition_contribution: float


def SIGNAL_NAMES(classes: Sequence[str]) -> tuple[str, ...]:  # noqa: N802
    return tuple(
        name
        for signals in (ConsumerSignals, ServiceSignals)
        for one in fields(signals)
        for name in (
            [f"{one.name}_{label}" for label in classes]
            if str(one.type).startswith("tuple")
            else [one.name]
        )
    )


def mean_confidence(window: Window) -> float | None:
    held = [
        as_number(row["predicted_confidence"])
        for row in window
        if row.get("predicted_confidence") is not None
    ]
    return sum(held) / len(held) if held else None


def time_of_day(window: Window) -> float:
    held = next(
        (row["time_of_day"] for row in reversed(window) if row.get("time_of_day") is not None),
        0.0,
    )
    return as_number(held)


def strong_classes(profile: InferenceServiceProfile) -> frozenset[str]:
    return frozenset(
        name
        for name, accuracy in profile.accuracy_by_class.items()
        if accuracy > profile.general_accuracy
    )


def weighted_disagreement(member_window: Window, service_name: str) -> float:
    answered, against = 0, 0.0
    for rows in _by_query(member_window).values():
        mine = next((row for row in rows if str(row["service_name"]) == service_name), None)
        if mine is None or not mine.get("predicted_label"):
            continue
        answered += 1
        others = [
            str(row["predicted_label"])
            for row in rows
            if str(row["service_name"]) != service_name and row.get("predicted_label")
        ]
        if others and str(mine["predicted_label"]) not in _most(others):
            against += as_number(mine.get("confidence") or 0.0)
    return against / answered if answered else 0.0


def consumer_signals(
    *,
    window: Window,
    classes: Sequence[str],
    previous: tuple[float, ...] | None,
    n_members: int,
) -> ConsumerSignals:
    share = class_share(window, classes, attribute="predicted_label")
    before = previous if previous is not None and len(previous) == len(share) else share
    return ConsumerSignals(
        class_share=share,
        class_share_change=tuple(now - then for now, then in zip(share, before, strict=True)),
        time_of_day=time_of_day(window),
        mean_confidence=mean_confidence(window) or 0.0,
        n_members=float(n_members),
    )


def service_signals(
    *,
    strong: frozenset[str],
    window: Window,
    member_window: Window,
    service_name: str,
    previous_strong_share: float | None,
    period_contribution: float,
    condition_contribution: float,
) -> ServiceSignals:
    ordered = tuple(sorted(strong))
    held = sum(class_share(window, ordered, attribute="predicted_label"))
    before = held if previous_strong_share is None else previous_strong_share
    return ServiceSignals(
        strong_class_share=held,
        strong_class_share_change=held - before,
        weighted_disagreement=weighted_disagreement(member_window, service_name),
        strong_class_share_labeled=sum(class_share(window, ordered, attribute="true_label")),
        period_contribution=period_contribution,
        condition_contribution=condition_contribution,
    )


def vector(consumer: ConsumerSignals, service: ServiceSignals) -> tuple[float, ...]:
    held: list[float] = []
    for signals in (consumer, service):
        for one in fields(signals):
            value = getattr(signals, one.name)
            held.extend(value if isinstance(value, tuple) else (value,))
    return tuple(held)


def _by_query(member_window: Window) -> Mapping[str, list[Row]]:
    held: dict[str, list[Row]] = {}
    for row in member_window:
        held.setdefault(str(row["query_id"]), []).append(row)
    return held


def _most(labels: Sequence[str]) -> frozenset[str]:
    counted = {name: labels.count(name) for name in labels}
    highest = max(counted.values())
    return frozenset(name for name, held in counted.items() if held == highest)
