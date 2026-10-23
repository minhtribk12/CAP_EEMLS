from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import get_args

import httpx

from eemls.domain.types import Condition
from eemls.interfaces.data_hub_api import ImageMetadata

logger = logging.getLogger(__name__)

Conditioning = Callable[[str], Awaitable[Condition]]

CONDITIONS: tuple[Condition, ...] = get_args(Condition)


class ConditionLookup:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_ms: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout_ms / 1000.0, transport=transport
        )
        self._known: dict[str, Condition] = {}
        self._last: Condition = "day"

    async def __call__(self, image_id: str) -> Condition:
        """The condition of one image; the last known one if the hub does not answer."""
        held = self._known.get(image_id)
        if held is not None:
            return held
        named = await self._asked(image_id)
        if named is None:
            return self._last
        self._known[image_id] = named
        self._last = named
        return named

    async def close(self) -> None:
        await self._client.aclose()

    async def _asked(self, image_id: str) -> Condition | None:
        try:
            answer = await self._client.get(f"/images/{image_id}/meta")
            if answer.status_code != httpx.codes.OK:
                self._unanswered(image_id, "refused", answer.status_code)
                return None
            held = ImageMetadata.model_validate(answer.json()).condition
        except (httpx.HTTPError, ValueError) as refused:
            self._unanswered(image_id, type(refused).__name__, refused)
            return None
        named = next((one for one in CONDITIONS if one == held), None)
        if named is None:
            self._unanswered(image_id, "unknown_condition", held)
        return named

    def _unanswered(self, image_id: str, error_type: str, detail: object) -> None:
        logger.warning(
            f"stage_tag=condition_lookup process_tag=control_plane error_type={error_type} "
            f"image_id={image_id} recovery_action=the_last_condition_is_kept detail={detail}"
        )


ConditionLookupFactory = Callable[..., ConditionLookup]
