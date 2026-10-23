from __future__ import annotations

import asyncio
import io
import time
from collections.abc import AsyncIterator

import httpx
import pytest
from PIL import Image

from eemls.monitoring.metrics import Monitoring
from eemls.testing.trees import TREE
from eemls_inference_service.backends.backend import BackendError, Prediction
from eemls_inference_service.config import load_inference_config
from eemls_inference_service.data_hub_client import DataHubClient
from eemls_inference_service.http_app import InferenceEndpoints, build_app

IMAGE_ID = "img-7"
Service = tuple[httpx.AsyncClient, InferenceEndpoints, list[str]]


def picture() -> bytes:
    held = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 40)).save(held, format="JPEG")
    return held.getvalue()


class StandingIn:
    def __init__(self, label: str = "car", delay_s: float = 0.0) -> None:
        self.label = label
        self.delay_s = delay_s
        self.calls = 0

    def predict(self, image: bytes) -> Prediction:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if not image:
            raise BackendError("no image to read")
        return Prediction(label=self.label, confidence=0.91, ranked=((self.label, 0.91),))

    def provider(self) -> str:
        return "CPUExecutionProvider"

    def close(self) -> None: ...


def built(
    *,
    backend: StandingIn | None = None,
    hub_status: int = 200,
    max_concurrent_requests: int = 2,
    instance_id: str = "i-1",
) -> Service:
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(hub_status, content=picture())

    settings = load_inference_config(TREE).services
    serving = InferenceEndpoints(
        instance_id=instance_id,
        service_name="infer-b5-orin",
        model="b5",
        device="jetson_orin_nano",
        node="orin-1",
        replica=1,
        number_format=settings.inference_service.number_format,
        backend=backend or StandingIn(),
        images=DataHubClient(
            base_url=str(settings.inference_service.data_hub_url),
            timeout_ms=settings.inference_service.image_fetch_timeout_ms,
            cache_images=settings.inference_service.image_cache_images,
            max_image_bytes=settings.inference_service.max_image_bytes,
            transport=httpx.MockTransport(answer),
        ),
        max_concurrent_requests=max_concurrent_requests,
        monitoring=Monitoring.from_config(settings.monitoring),
    )
    app = build_app(serving, Monitoring.from_config(settings.monitoring))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://inference")
    return client, serving, asked


@pytest.fixture
async def service() -> AsyncIterator[Service]:
    held = built()
    yield held
    await held[0].aclose()


class TestWhatItAnswers:
    async def test_an_image_id_is_answered_with_a_label(self, service: Service) -> None:
        client, _, asked = service

        answer = await client.post("/predict", json={"image_id": IMAGE_ID, "query_id": "q-1"})

        assert answer.status_code == 200
        body = answer.json()
        assert body["predicted_label"] == "car"
        assert body["confidence"] == pytest.approx(0.91)
        assert asked == [f"/images/{IMAGE_ID}"]

    async def test_it_answers_everything_its_stored_row_holds(self, service: Service) -> None:
        client, _, _ = service

        answer = await client.post("/predict", json={"image_id": IMAGE_ID, "query_id": "q-1"})
        body = answer.json()

        assert set(body) == {
            "query_id",
            "instance_id",
            "service_name",
            "model",
            "device",
            "node",
            "replica",
            "predicted_label",
            "confidence",
            "fetch_ms",
            "inference_ms",
            "total_ms",
            "execution_provider",
            "number_format",
            "cache_hit",
            "status",
        }
        assert body["instance_id"] == "i-1"
        assert body["execution_provider"] == "CPUExecutionProvider"
        assert body["total_ms"] >= body["inference_ms"]

    async def test_a_second_request_for_one_image_does_not_fetch_it_again(
        self, service: Service
    ) -> None:
        client, _, asked = service
        await client.post("/predict", json={"image_id": IMAGE_ID})

        again = await client.post("/predict", json={"image_id": IMAGE_ID})

        assert again.json()["cache_hit"] is True
        assert asked == [f"/images/{IMAGE_ID}"]

    async def test_two_copies_answer_alike_and_each_names_itself(self) -> None:
        one, _, _ = built(instance_id="i-1")
        other, _, _ = built(instance_id="i-2")

        first = (await one.post("/predict", json={"image_id": IMAGE_ID})).json()
        second = (await other.post("/predict", json={"image_id": IMAGE_ID})).json()

        assert first["predicted_label"] == second["predicted_label"]
        assert (first["instance_id"], second["instance_id"]) == ("i-1", "i-2")
        await one.aclose()
        await other.aclose()


class TestWhenItWillNotAnswer:
    async def test_a_data_hub_that_is_down_is_a_typed_error_and_not_a_500(self) -> None:
        client, _, _ = built(hub_status=503)

        answer = await client.post("/predict", json={"image_id": IMAGE_ID})

        assert answer.status_code == 502
        assert IMAGE_ID in answer.json()["detail"]
        await client.aclose()

    async def test_a_copy_over_its_limit_answers_503_rather_than_queueing(self) -> None:
        client, _, _ = built(backend=StandingIn(delay_s=0.3), max_concurrent_requests=1)

        answers = await asyncio.gather(
            *(client.post("/predict", json={"image_id": IMAGE_ID}) for _ in range(3))
        )

        codes = [one.status_code for one in answers]
        assert 503 in codes
        assert 200 in codes
        await client.aclose()

    async def test_a_request_naming_something_it_does_not_take_is_refused(
        self, service: Service
    ) -> None:
        client, _, _ = service

        answer = await client.post("/predict", json={"image_id": IMAGE_ID, "nonsense": 1})

        assert answer.status_code == 422


class TestWhatItSaysAboutItself:
    async def test_a_built_copy_is_healthy_and_ready(self, service: Service) -> None:
        client, _, _ = service

        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200

    async def test_it_publishes_what_it_did_on_its_metrics_path(self, service: Service) -> None:
        client, _, _ = service
        await client.post("/predict", json={"image_id": IMAGE_ID})

        body = (await client.get("/metrics")).text

        assert "eemls_inference_requests_total" in body
        assert "eemls_model_inference_seconds" in body
        assert "eemls_image_fetch_seconds" in body
