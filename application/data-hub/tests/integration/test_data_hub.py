from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from eemls.monitoring.metrics import Monitoring
from eemls.testing.images import prepared
from eemls.testing.trees import TREE
from eemls_data_hub.config import load_data_hub_config
from eemls_data_hub.http_app import build_app
from eemls_data_hub.image_store import ImageStore
from eemls_data_hub.main import main


def serving(root: Path) -> httpx.AsyncClient:
    settings = load_data_hub_config(TREE).services
    store = ImageStore.opened(prepared(root), max_image_bytes=settings.data_hub.max_image_bytes)
    monitoring = Monitoring.from_config(settings.monitoring)
    app = build_app(store, monitoring)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://data-hub")


@pytest.fixture
async def hub(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    client = serving(tmp_path)
    yield client
    await client.aclose()


class TestServingAnImage:
    async def test_an_image_is_answered_as_bytes(self, hub: httpx.AsyncClient) -> None:
        answer = await hub.get("/images/car-day-clear-0")

        assert answer.status_code == 200
        assert answer.headers["content-type"] == "image/jpeg"
        assert answer.content[:2] == b"\xff\xd8"

    async def test_what_the_hub_knows_about_an_image(self, hub: httpx.AsyncClient) -> None:
        body = (await hub.get("/images/car-night-clear-0/meta")).json()

        assert body == {
            "image_id": "car-night-clear-0",
            "label": "car",
            "dataset": "open_images",
            "original_label": "/m/car",
            "condition": "night",
            "image_quality": "clear",
        }

    async def test_three_services_fetching_at_once_each_get_the_image(
        self, hub: httpx.AsyncClient
    ) -> None:
        answers = await asyncio.gather(*(hub.get("/images/car-day-clear-0") for _ in range(3)))

        assert [one.status_code for one in answers] == [200, 200, 200]
        assert len({one.content for one in answers}) == 1


class TestListingThem:
    async def test_it_lists_one_class_in_one_quality(self, hub: httpx.AsyncClient) -> None:
        body = (await hub.get("/images", params={"label": "car"})).json()

        assert body["image_ids"] == ["car-day-clear-0", "car-day-clear-1"]
        assert (body["condition"], body["image_quality"]) == ("day", "clear")

    async def test_it_lists_the_blurred_copies_when_asked(self, hub: httpx.AsyncClient) -> None:
        body = (
            await hub.get("/images", params={"label": "car", "image_quality": "blurred"})
        ).json()

        assert body["image_ids"] == ["car-day-blurred-0", "car-day-blurred-1"]

    async def test_it_lists_the_night_copies_apart_from_the_day_ones(
        self, hub: httpx.AsyncClient
    ) -> None:
        day = (await hub.get("/images", params={"label": "car", "condition": "day"})).json()
        night = (await hub.get("/images", params={"label": "car", "condition": "night"})).json()

        assert night["image_ids"] == ["car-night-clear-0", "car-night-clear-1"]
        assert set(day["image_ids"]) & set(night["image_ids"]) == set()

    async def test_it_lists_the_conditions_it_holds(self, hub: httpx.AsyncClient) -> None:
        body = (await hub.get("/conditions")).json()

        assert body == {"conditions": ["day", "night"]}

    async def test_a_class_it_does_not_hold_lists_nothing(self, hub: httpx.AsyncClient) -> None:
        body = (await hub.get("/images", params={"label": "unicorn"})).json()

        assert body["image_ids"] == []

    async def test_it_lists_the_classes_it_holds(self, hub: httpx.AsyncClient) -> None:
        body = (await hub.get("/labels")).json()

        assert body["labels"] == ["car", "pedestrian"]


class TestWhenItCannotAnswer:
    async def test_an_unknown_id_is_a_404_naming_it_and_never_a_500(
        self, hub: httpx.AsyncClient
    ) -> None:
        answer = await hub.get("/images/img-nowhere")

        assert answer.status_code == 404
        assert "img-nowhere" in answer.json()["detail"]

    async def test_an_unknown_id_has_no_metadata_either(self, hub: httpx.AsyncClient) -> None:
        assert (await hub.get("/images/img-nowhere/meta")).status_code == 404

    async def test_a_listing_without_a_class_is_refused(self, hub: httpx.AsyncClient) -> None:
        assert (await hub.get("/images")).status_code == 422


class TestWhatItSaysAboutItself:
    async def test_it_is_healthy_and_ready_once_it_holds_images(
        self, hub: httpx.AsyncClient
    ) -> None:
        assert (await hub.get("/healthz")).status_code == 200
        assert (await hub.get("/readyz")).status_code == 200

    async def test_it_publishes_what_it_served(self, hub: httpx.AsyncClient) -> None:
        await hub.get("/images/car-day-clear-0")
        await hub.get("/images/img-nowhere")

        body = (await hub.get("/metrics")).text

        assert 'eemls_images_served_total{outcome="accepted"} 1.0' in body
        assert 'eemls_images_served_total{outcome="refused"} 1.0' in body

    def test_a_root_with_no_images_is_refused_by_the_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(TREE), "--images", str(tmp_path)]) == 1
        assert "manifest" in capsys.readouterr().err
