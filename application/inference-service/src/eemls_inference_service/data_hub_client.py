from __future__ import annotations

import time
from collections import OrderedDict

import httpx

from eemls.domain.errors import EemlsError


class ImageUnavailableError(EemlsError): ...


class FetchedImage:
    __slots__ = ("body", "cache_hit", "fetch_ms")

    def __init__(self, body: bytes, *, fetch_ms: float, cache_hit: bool) -> None:
        self.body = body
        self.fetch_ms = fetch_ms
        self.cache_hit = cache_hit


class DataHubClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_ms: int,
        cache_images: int,
        max_image_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_ms / 1000.0, transport=transport
        )
        self._cache_images = cache_images
        self._max_image_bytes = max_image_bytes
        self._held: OrderedDict[str, bytes] = OrderedDict()

    def image(self, image_id: str) -> FetchedImage:
        start = time.perf_counter()
        held = self._held.get(image_id)
        if held is not None:
            self._held.move_to_end(image_id)
            return FetchedImage(
                held, fetch_ms=(time.perf_counter() - start) * 1000.0, cache_hit=True
            )
        body = self._read(image_id)
        self._keep(image_id, body)
        return FetchedImage(body, fetch_ms=(time.perf_counter() - start) * 1000.0, cache_hit=False)

    def close(self) -> None:
        self._client.close()

    def _read(self, image_id: str) -> bytes:
        try:
            answer = self._client.get(f"/images/{image_id}")
        except httpx.HTTPError as refused:
            raise ImageUnavailableError(
                f"{image_id}: the data hub did not answer: {refused}"
            ) from refused
        if answer.status_code != httpx.codes.OK:
            raise ImageUnavailableError(f"{image_id}: the data hub answered {answer.status_code}")
        body = answer.content
        if len(body) > self._max_image_bytes:
            raise ImageUnavailableError(
                f"{image_id}: {len(body)} bytes is over the {self._max_image_bytes} allowed"
            )
        return bytes(body)

    def _keep(self, image_id: str, body: bytes) -> None:
        if self._cache_images <= 0:
            return
        self._held[image_id] = body
        self._held.move_to_end(image_id)
        while len(self._held) > self._cache_images:
            self._held.popitem(last=False)
