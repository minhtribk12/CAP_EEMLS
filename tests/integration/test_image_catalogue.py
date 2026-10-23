from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from eemls.image_catalogue import Catalogue, CatalogueError
from eemls.monitoring.metrics import Monitoring
from eemls.testing.images import prepared
from eemls.testing.trees import TREE
from eemls_data_hub.config import load_data_hub_config
from eemls_data_hub.http_app import build_app
from eemls_data_hub.image_store import ImageStore


def hub_of(root: Path) -> httpx.Client:
    settings = load_data_hub_config(TREE).services
    store = ImageStore.opened(prepared(root), max_image_bytes=settings.data_hub.max_image_bytes)
    monitoring = Monitoring.from_config(settings.monitoring)
    app = build_app(store, monitoring)

    def answer(request: httpx.Request) -> httpx.Response:
        async def ask() -> tuple[int, bytes]:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://data-hub:8081"
            ) as client:
                held = await client.get(request.url.path, params=request.url.params)
                return held.status_code, held.content

        status, body = asyncio.run(ask())
        return httpx.Response(status, content=body, headers={"content-type": "application/json"})

    return httpx.Client(transport=httpx.MockTransport(answer), base_url="http://data-hub:8081")


class TestTheCatalogue:
    def test_it_reads_the_ids_it_may_ask_about_from_the_data_hub(self, tmp_path: Path) -> None:
        catalogue = Catalogue.read(hub_of(tmp_path))

        assert (
            catalogue.pick(label="car", condition="day", image_quality="clear", draw=0.0)
            == "car-day-clear-0"
        )
        assert (
            catalogue.pick(label="car", condition="night", image_quality="blurred", draw=0.99)
            == "car-night-blurred-1"
        )

    def test_a_data_hub_that_answers_an_error_is_a_typed_refusal(self) -> None:
        hub = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
            base_url="http://data-hub:8081",
        )

        with pytest.raises(CatalogueError, match="503"):
            Catalogue.read(hub)

    def test_a_data_hub_that_is_not_there_is_a_typed_refusal(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nothing is listening for the data hub")

        hub = httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://data-hub:8081")

        with pytest.raises(CatalogueError, match="did not answer"):
            Catalogue.read(hub)
