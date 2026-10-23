from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from eemls.domain.records import InferenceServiceProfile
from eemls.domain.statistics import as_number
from eemls.domain.types import Row
from eemls.evaluation_engine.confidence_vote import mean
from eemls.evaluation_engine.ensemble_scoring import MemberRow
from eemls.evaluation_engine.window import Window


def member_rows(
    member_window: Window, profiles: Mapping[str, InferenceServiceProfile]
) -> tuple[MemberRow, ...]:
    return tuple(
        _row(row, profiles[str(row["service_name"])])
        for row in member_window
        if str(row["service_name"]) in profiles
    )


def estimated_member_rows(
    one: InferenceServiceProfile, *, labeled: Sequence[MemberRow]
) -> tuple[MemberRow, ...]:
    held = []
    seen: dict[str, int] = {}
    frequency = _frequency_of(labeled)
    votes = _votes_of(labeled)
    for query_id, true_label in _labels_of(labeled).items():
        accuracy = one.accuracy_by_class.get(true_label)
        confidence = one.confidence_by_class.get(true_label)
        if accuracy is None or confidence is None:
            continue
        place = seen.get(true_label, 0)
        seen[true_label] = place + 1
        label = (
            true_label
            if _is_right(place, accuracy)
            else _wrong_label(true_label, votes.get(query_id, {}), frequency)
        )
        if label is None:
            continue
        held.append(
            MemberRow(
                query_id=query_id,
                service_name=one.service_name,
                predicted_label=label,
                confidence=confidence,
                true_label=true_label,
                total_ms=one.mean_ms,
                cost_per_1k=one.cost_per_1k,
                throughput_rps=one.throughput_rps,
            )
        )
    return tuple(held)


def _row(row: Row, one: InferenceServiceProfile) -> MemberRow:
    label = row.get("predicted_label")
    confidence = row.get("confidence")
    return MemberRow(
        query_id=str(row["query_id"]),
        service_name=str(row["service_name"]),
        predicted_label=None if label is None else str(label),
        confidence=None if confidence is None else as_number(confidence),
        true_label=None if row.get("true_label") is None else str(row["true_label"]),
        total_ms=as_number(row["total_ms"]),
        cost_per_1k=one.cost_per_1k,
        throughput_rps=one.throughput_rps,
    )


def _is_right(place: int, accuracy: float) -> bool:
    return math.floor((place + 1) * accuracy + 0.5) > math.floor(place * accuracy + 0.5)


def _wrong_label(
    true_label: str, votes: Mapping[str, list[float]], frequency: Mapping[str, int]
) -> str | None:
    others = (set(votes) | set(frequency)) - {true_label}
    if not others:
        return None
    return max(
        sorted(others),
        key=lambda label: (
            mean(votes[label]) if label in votes else 0.0,
            len(votes.get(label, [])),
            frequency.get(label, 0),
        ),
    )


def _votes_of(labeled: Sequence[MemberRow]) -> Mapping[str, Mapping[str, list[float]]]:
    held: dict[str, dict[str, list[float]]] = {}
    for row in labeled:
        if row.predicted_label is None or row.confidence is None:
            continue
        held.setdefault(row.query_id, {}).setdefault(row.predicted_label, []).append(row.confidence)
    return held


def _frequency_of(labeled: Sequence[MemberRow]) -> Mapping[str, int]:
    held: dict[str, int] = {}
    for row in labeled:
        for label in (row.predicted_label, row.true_label):
            if label is not None:
                held[label] = held.get(label, 0) + 1
    return held


def _labels_of(labeled: Sequence[MemberRow]) -> Mapping[str, str]:
    held: dict[str, str] = {}
    for row in labeled:
        if row.true_label is not None:
            held.setdefault(row.query_id, row.true_label)
    return held
