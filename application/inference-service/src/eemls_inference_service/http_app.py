from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict

from eemls.domain.measuring import Measuring
from eemls.interfaces.inference_service_api import PredictionAnswer
from eemls.monitoring.metrics import Monitoring
from eemls.monitoring.metrics_endpoint import service_app
from eemls_inference_service.backends.backend import Backend, BackendError
from eemls_inference_service.data_hub_client import DataHubClient, ImageUnavailableError

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    query_id: str | None = None


class InferenceEndpoints:
    def __init__(
        self,
        *,
        instance_id: str,
        service_name: str,
        model: str,
        device: str,
        node: str,
        replica: int,
        number_format: str,
        backend: Backend,
        images: DataHubClient,
        max_concurrent_requests: int,
        monitoring: Measuring,
    ) -> None:
        self.instance_id = instance_id
        self.service_name = service_name
        self.model = model
        self.device = device
        self.node = node
        self.replica = replica
        self.number_format = number_format
        self.backend = backend
        self.images = images
        self.monitoring = monitoring
        self._free = threading.BoundedSemaphore(max_concurrent_requests)

    @contextmanager
    def within_capacity(self) -> Iterator[None]:
        if not self._free.acquire(blocking=False):
            raise CapacityError(f"{self.instance_id} is already serving all it accepts")
        try:
            yield
        finally:
            self._free.release()


class CapacityError(BackendError): ...


def build_app(serving: InferenceEndpoints, monitoring: Monitoring) -> FastAPI:
    app = service_app(f"eemls inference {serving.service_name}", monitoring, lambda: True)

    @app.post("/predict", response_model=PredictionAnswer)
    def predict(asked: PredictRequest) -> PredictionAnswer:
        try:
            with serving.within_capacity():
                return _answer(serving, asked)
        except CapacityError as refused:
            _counted(serving, "refused")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(refused)) from refused
        except ImageUnavailableError as refused:
            _counted(serving, "failed")
            logger.warning(
                f"error_type=image_unavailable image_id={asked.image_id} "
                f"recovery_action=the_ensemble_answers_without_this_member detail={refused}"
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(refused)) from refused
        except BackendError as refused:
            _counted(serving, "failed")
            logger.error(f"error_type=backend_refused recovery_action=none detail={refused}")
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(refused)) from refused

    return app


def _counted(serving: InferenceEndpoints, outcome: str) -> None:
    serving.monitoring.count(
        "eemls_inference_requests_total",
        model=serving.model,
        device=serving.device,
        outcome=outcome,
    )


def _answer(serving: InferenceEndpoints, asked: PredictRequest) -> PredictionAnswer:
    started = time.perf_counter()
    held = serving.images.image(asked.image_id)
    serving.monitoring.observe(
        "eemls_image_fetch_seconds",
        seconds=held.fetch_ms / 1000.0,
        model=serving.model,
        cache="hit" if held.cache_hit else "miss",
    )
    inference_started = time.perf_counter()
    with serving.monitoring.timer(
        "eemls_model_inference_seconds", model=serving.model, device=serving.device
    ):
        prediction = serving.backend.predict(held.body)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    _counted(serving, "accepted")
    return PredictionAnswer(
        query_id=asked.query_id,
        instance_id=serving.instance_id,
        service_name=serving.service_name,
        model=serving.model,
        device=serving.device,
        node=serving.node,
        replica=serving.replica,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        fetch_ms=held.fetch_ms,
        inference_ms=inference_ms,
        total_ms=(time.perf_counter() - started) * 1000.0,
        execution_provider=serving.backend.provider(),
        number_format=serving.number_format,
        cache_hit=held.cache_hit,
        status="ok",
    )
