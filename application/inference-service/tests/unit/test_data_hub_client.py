from __future__ import annotations

import httpx
import pytest

from eemls_inference_service.data_hub_client import DataHubClient, ImageUnavailableError

BODY = b"the bytes of one image"


def serving(status: int = 200, body: bytes = BODY) -> tuple[DataHubClient, list[str]]:
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(status, content=body)

    client = DataHubClient(
        base_url="http://data-hub:8081",
        timeout_ms=100,
        cache_images=2,
        max_image_bytes=1_000_000,
        transport=httpx.MockTransport(answer),
    )
    return client, asked


class TestReadingOne:
    def test_it_reads_the_image_by_its_id(self) -> None:
        client, asked = serving()

        held = client.image("img-7")

        assert held.body == BODY
        assert held.cache_hit is False
        assert asked == ["/images/img-7"]

    def test_it_reports_how_long_the_fetch_took(self) -> None:
        client, _ = serving()

        assert client.image("img-7").fetch_ms >= 0.0


class TestTheCache:
    def test_an_image_it_has_just_read_is_not_read_again(self) -> None:
        client, asked = serving()
        client.image("img-7")

        again = client.image("img-7")

        assert again.cache_hit is True
        assert asked == ["/images/img-7"]

    def test_it_forgets_the_least_recently_asked_for(self) -> None:
        client, asked = serving()
        for image_id in ("a", "b", "c"):
            client.image(image_id)

        assert client.image("a").cache_hit is False
        assert asked == ["/images/a", "/images/b", "/images/c", "/images/a"]

    def test_a_cache_of_nothing_holds_nothing(self) -> None:
        client = DataHubClient(
            base_url="http://data-hub:8081",
            timeout_ms=100,
            cache_images=0,
            max_image_bytes=1_000_000,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=BODY)),
        )
        client.image("img-7")

        assert client.image("img-7").cache_hit is False


class TestWhenTheHubWillNotAnswer:
    def test_a_hub_that_does_not_know_the_id_is_a_typed_refusal(self) -> None:
        client, _ = serving(status=404)

        with pytest.raises(ImageUnavailableError, match="img-9"):
            client.image("img-9")

    def test_a_hub_that_is_down_is_a_typed_refusal(self) -> None:
        def refuses(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        client = DataHubClient(
            base_url="http://data-hub:8081",
            timeout_ms=100,
            cache_images=2,
            max_image_bytes=1_000_000,
            transport=httpx.MockTransport(refuses),
        )

        with pytest.raises(ImageUnavailableError, match="the data hub did not answer"):
            client.image("img-7")

    def test_an_image_larger_than_allowed_is_refused(self) -> None:
        client = DataHubClient(
            base_url="http://data-hub:8081",
            timeout_ms=100,
            cache_images=2,
            max_image_bytes=4,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=BODY)),
        )

        with pytest.raises(ImageUnavailableError, match="over the 4"):
            client.image("img-7")

    def test_a_refused_image_is_not_cached(self) -> None:
        client, asked = serving(status=500)

        for _ in range(2):
            with pytest.raises(ImageUnavailableError):
                client.image("img-7")

        assert asked == ["/images/img-7", "/images/img-7"]
