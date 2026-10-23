from __future__ import annotations

from collections.abc import Mapping

import httpx

from eemls.domain.cost import DeviceRate
from eemls.domain.types import Condition, RuntimeContext
from eemls.evaluation_engine.window import Window
from eemls.metrics_database.database import MetricsDatabase
from eemls.observation_service.metrics_storage import store_metrics
from eemls.observation_service.observation_service_api import (
    ProfileUpdateAnswer,
    ProfileUpdateRequest,
    RowBatch,
    Stored,
)
from eemls.observation_service.operating_conditions import Conditioning
from eemls.observation_service.profile_updates import (
    ProfileSettings,
    ProfileUpdatesClient,
    update_profiles,
)
from eemls.testing.records import context


class FixedContext:
    def __init__(self, held: RuntimeContext | None = None) -> None:
        self.held = held if held is not None else context()

    def __call__(self, consumer_id: str) -> RuntimeContext:
        return self.held

    def read_from(self, consumer_id: str, window: Window) -> None: ...


def conditioned(condition: Condition = "day") -> Conditioning:
    async def of(image_id: str) -> Condition:
        return condition

    return of


class InProcessObservationService(httpx.MockTransport):
    def __init__(
        self,
        database: MetricsDatabase,
        *,
        run_id: str,
        rates: Mapping[str, DeviceRate],
        window_s: float,
        min_labeled_per_class: int,
        min_labeled_overall: int,
    ) -> None:
        super().__init__(self._answered)
        self.database = database
        self.run_id = run_id
        self._rates = rates
        self._window_s = window_s
        self._min_labeled_per_class = min_labeled_per_class
        self._min_labeled_overall = min_labeled_overall

    def _answered(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rows":
            batch = RowBatch.model_validate_json(request.content)
            if batch.run_id != self.run_id:
                return httpx.Response(409, text=f"this service stores run {self.run_id}")
            stored = store_metrics(batch, self.database)
            return httpx.Response(200, json=Stored(stored=stored).model_dump(mode="json"))
        if request.url.path == "/profiles/update":
            at = ProfileUpdateRequest.model_validate_json(request.content).at
            updated = update_profiles(
                self.database,
                ProfileSettings.ending_at(
                    at,
                    window_s=self._window_s,
                    rates=self._rates,
                    min_labeled_per_class=self._min_labeled_per_class,
                    min_labeled_overall=self._min_labeled_overall,
                ),
            )
            return httpx.Response(
                200, json=ProfileUpdateAnswer(profiles=updated).model_dump(mode="json")
            )
        return httpx.Response(404)


def in_process_profile_updates(
    database: MetricsDatabase,
    *,
    rates: Mapping[str, DeviceRate],
    window_s: float,
    interval_s: float,
    min_labeled_per_class: int,
    min_labeled_overall: int,
) -> ProfileUpdatesClient:
    return ProfileUpdatesClient(
        database=database,
        base_url="http://observation-service",
        interval_s=interval_s,
        request_timeout_ms=1000.0,
        transport=InProcessObservationService(
            database,
            run_id="in-process",
            rates=rates,
            window_s=window_s,
            min_labeled_per_class=min_labeled_per_class,
            min_labeled_overall=min_labeled_overall,
        ),
    )
