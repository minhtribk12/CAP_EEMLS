from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import get_args

import httpx

from eemls.domain.errors import EemlsError
from eemls.domain.types import DEFAULT_CONDITION, DEFAULT_IMAGE_QUALITY, ImageQuality

QUALITIES: tuple[ImageQuality, ...] = get_args(ImageQuality)


class CatalogueError(EemlsError): ...


class Catalogue:
    def __init__(self, ids: Mapping[tuple[str, str, str], Sequence[str]]) -> None:
        self._ids = {key: tuple(value) for key, value in ids.items()}

    @classmethod
    def read(cls, hub: httpx.Client) -> Catalogue:
        held: dict[tuple[str, str, str], tuple[str, ...]] = {}
        conditions = _names(_read(hub, "/conditions", None), "conditions", "/conditions")
        for label in _names(_read(hub, "/labels", None), "labels", "/labels"):
            for condition in conditions:
                for quality in QUALITIES:
                    answer = _read(
                        hub,
                        "/images",
                        {"label": label, "condition": condition, "image_quality": quality},
                    )
                    held[(label, condition, quality)] = _names(answer, "image_ids", "/images")
        return cls(held)

    def pick(self, *, label: str, condition: str, image_quality: str, draw: float) -> str:
        held = self._ids.get((label, condition, image_quality), ())
        if not held:
            raise CatalogueError(
                f"the data hub holds no {image_quality} {condition} image of {label}"
            )
        return held[min(int(draw * len(held)), len(held) - 1)]

    def draw(self, *, dominant_classes: Sequence[str], draws: random.Random) -> str:
        return self.pick(
            label=draws.choice(dominant_classes),
            condition=DEFAULT_CONDITION,
            image_quality=DEFAULT_IMAGE_QUALITY,
            draw=draws.random(),
        )


def _names(held: Mapping[str, object], key: str, path: str) -> tuple[str, ...]:
    found = held.get(key)
    if not isinstance(found, list) or any(not isinstance(one, str) for one in found):
        raise CatalogueError(f"the data hub answered {path} with no list of {key}")
    return tuple(str(one) for one in found)


def _read(hub: httpx.Client, path: str, params: Mapping[str, str] | None) -> dict[str, object]:
    try:
        answer = hub.get(path, params=params)
    except httpx.HTTPError as refused:
        raise CatalogueError(f"the data hub did not answer {path}: {refused}") from refused
    if answer.status_code != httpx.codes.OK:
        raise CatalogueError(f"the data hub answered {answer.status_code} for {path}")
    try:
        held = answer.json()
    except ValueError as refused:
        raise CatalogueError(f"the data hub answered {path} with no listing") from refused
    if not isinstance(held, dict):
        raise CatalogueError(f"the data hub answered {path} with no listing")
    return held
