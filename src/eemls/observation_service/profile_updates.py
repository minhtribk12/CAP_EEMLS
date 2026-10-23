from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import httpx

from eemls.domain.cost import DeviceRate, cost_per_1k, energy_wh_per_1k, throughput_per_s
from eemls.domain.errors import EemlsError
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.statistics import as_number, percentile
from eemls.domain.types import Row, Timestamp
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.database import MetricsDatabase
from eemls.observation_service.observation_service_api import (
    ObservationServiceError,
    ProfileUpdateAnswer,
    ProfileUpdateRequest,
)

logger = logging.getLogger(__name__)


class ProfileError(EemlsError): ...


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    window_start: Timestamp
    window_end: Timestamp
    rates: Mapping[str, DeviceRate]
    min_labeled_per_class: int
    min_labeled_overall: int

    @classmethod
    def ending_at(
        cls,
        end: Timestamp,
        *,
        window_s: float,
        rates: Mapping[str, DeviceRate],
        min_labeled_per_class: int,
        min_labeled_overall: int,
    ) -> ProfileSettings:
        return cls(
            window_start=end - timedelta(seconds=window_s),
            window_end=end,
            rates=rates,
            min_labeled_per_class=min_labeled_per_class,
            min_labeled_overall=min_labeled_overall,
        )


def by_service(profiles: Sequence[InferenceServiceProfile]) -> dict[str, InferenceServiceProfile]:
    held: dict[str, InferenceServiceProfile] = {}
    for one in sorted(profiles, key=lambda p: (p.window_end, p.instance_id)):
        held[one.service_name] = one
    return held


def update_profiles(
    database: MetricsDatabase, settings: ProfileSettings
) -> tuple[InferenceServiceProfile, ...]:
    current = by_service(database.read_profiles())
    rows = database.read_member_answers(since=settings.window_start, until=settings.window_end)
    updated = updated_profiles(rows, current=current, settings=settings)
    for one in updated:
        database.write_profile(one)
    database.flush()
    return updated


class ProfileUpdating(Protocol):
    def latest(self, at: datetime) -> Mapping[str, InferenceServiceProfile]: ...


class ProfileUpdatesClient:
    def __init__(
        self,
        *,
        database: MetricsDatabase,
        base_url: str,
        interval_s: float,
        request_timeout_ms: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._database = database
        self._interval = timedelta(seconds=interval_s)
        self._client = httpx.Client(
            base_url=base_url, timeout=request_timeout_ms / 1000.0, transport=transport
        )
        self._held: dict[str, InferenceServiceProfile] | None = None
        self._updated_at: datetime | None = None

    def latest(self, at: datetime) -> Mapping[str, InferenceServiceProfile]:
        self._refresh(at)
        return MappingProxyType(dict(self._held or {}))

    def close(self) -> None:
        self._client.close()

    def _refresh(self, at: datetime) -> None:
        if self._held is None or self._updated_at is None:
            self._held = by_service(self._database.read_profiles())
            self._updated_at = at
        elif at - self._updated_at >= self._interval:
            self._updated_at = at
            try:
                self._update(self._held, at)
            except Exception as failed:
                logger.error(
                    f"stage_tag=profile process_tag=control_plane "
                    f"error_type=profile_rebuild_failed at={at.isoformat()} "
                    f"recovery_action=the_held_profiles_are_kept "
                    f"detail_type={type(failed).__name__} detail={failed}"
                )

    def _update(self, held: dict[str, InferenceServiceProfile], at: datetime) -> None:
        try:
            self._database.flush()
        except WriteError as refused:
            logger.warning(
                f"error_type=write_refused_before_rebuild "
                f"recovery_action=rebuild_from_what_is_stored detail={refused}"
            )
        answer = self._client.post(
            "/profiles/update", json=ProfileUpdateRequest(at=at).model_dump(mode="json")
        )
        if answer.status_code != httpx.codes.OK:
            raise ObservationServiceError(
                f"the Observation Service answered {answer.status_code}: {answer.text}"
            )
        for one in ProfileUpdateAnswer.model_validate_json(answer.content).profiles:
            held[one.service_name] = one


def updated_profiles(
    rows: Sequence[Row],
    *,
    current: Mapping[str, InferenceServiceProfile],
    settings: ProfileSettings,
) -> tuple[InferenceServiceProfile, ...]:
    if settings.window_end <= settings.window_start:
        raise ProfileError("the window has no length")
    held: dict[str, list[Row]] = {}
    for row in rows:
        name = str(row["service_name"])
        if name in current:
            held.setdefault(name, []).append(row)
    built = [
        one
        for name, mine in sorted(held.items())
        if (one := _profile(current[name], mine, settings))
    ]
    return tuple(built)


def _profile(
    before: InferenceServiceProfile, rows: Sequence[Row], settings: ProfileSettings
) -> InferenceServiceProfile | None:
    rate = settings.rates.get(before.device)
    if rate is None:
        raise ProfileError(f"{before.device} has no rate, so nothing can be charged for it")
    answered = [
        row for row in rows if row.get("predicted_label") is not None and row["status"] == "ok"
    ]
    if not answered:
        return None
    times = sorted(as_number(row["total_ms"]) for row in answered)
    mean_ms = sum(times) / len(times)
    labeled = [row for row in answered if row.get("true_label") is not None]
    classes = {
        name: mine
        for name, mine in _per_class(labeled).items()
        if len(mine) >= settings.min_labeled_per_class
    }
    return InferenceServiceProfile.model_validate(
        before.model_dump()
        | {
            "window_start": settings.window_start,
            "window_end": settings.window_end,
            "mean_ms": mean_ms,
            "p50_ms": percentile(times, 50.0),
            "p95_ms": percentile(times, 95.0),
            "max_ms": times[-1],
            "throughput_rps": throughput_per_s(mean_ms),
            "n_queries": len(answered),
            "energy_wh_per_1k": energy_wh_per_1k(rate, mean_response_time_ms=mean_ms),
            "cost_per_1k": cost_per_1k(rate, mean_response_time_ms=mean_ms),
            "general_accuracy": (
                _accuracy(labeled)
                if len(labeled) >= settings.min_labeled_overall
                else before.general_accuracy
            ),
            "accuracy_by_class": {
                **before.accuracy_by_class,
                **{name: _accuracy(mine) for name, mine in classes.items()},
            },
            "confidence_by_class": {
                **before.confidence_by_class,
                **_confidence_by_class(classes),
            },
            "source": "measured",
        }
    )


def _accuracy(rows: Sequence[Row]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["predicted_label"] == row["true_label"]) / len(rows)


def _confidence_by_class(classes: Mapping[str, Sequence[Row]]) -> dict[str, float]:
    held = {
        name: [as_number(row["confidence"]) for row in mine if row.get("confidence") is not None]
        for name, mine in classes.items()
    }
    return {name: sum(mine) / len(mine) for name, mine in held.items() if mine}


def _per_class(rows: Sequence[Row]) -> dict[str, list[Row]]:
    seen: dict[str, list[Row]] = {}
    for row in rows:
        seen.setdefault(str(row["true_label"]), []).append(row)
    return dict(sorted(seen.items()))


def profiles_in(path: Path) -> list[InferenceServiceProfile]:
    return [
        InferenceServiceProfile.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
