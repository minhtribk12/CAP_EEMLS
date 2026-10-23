from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from pydantic import ValidationError, computed_field, model_validator

from eemls.config.reading import ConfigError
from eemls.config.schema import (
    FrozenMap,
    FrozenModel,
    PositiveCount,
    PositiveNumber,
    Ratio,
)

QUALITY = 92
MANIFEST = "manifest.jsonl"
TRAINED = "trained.txt"


class ImageQuality(StrEnum):
    clear = "clear"
    blurred = "blurred"


class DatasetSource(FrozenModel):
    citation: str
    url: str
    license: str
    contributes: tuple[str, ...]
    conditions: tuple[str, ...]
    label_map: FrozenMap[str]

    @model_validator(mode="after")
    def _the_label_map_covers_what_it_contributes(self) -> DatasetSource:
        if not self.contributes:
            raise ValueError("contributes no class")
        mapped = set(self.label_map.values())
        missing = [name for name in self.contributes if name not in mapped]
        if missing:
            raise ValueError(f"maps no original label to {missing}")
        if not self.conditions:
            raise ValueError("supplies no condition")
        return self


class BlurConfig(FrozenModel):
    sigma: PositiveNumber
    kernel: PositiveCount


class DatasetsConfig(FrozenModel):
    dominant_classes: tuple[str, ...]
    tail_classes: tuple[str, ...]
    conditions: tuple[str, ...]
    image_size: PositiveCount
    images_per_dominant_class: PositiveCount
    images_per_tail_class: PositiveCount
    train_fraction: Ratio
    blur: BlurConfig
    sources: FrozenMap[DatasetSource]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classes(self) -> tuple[str, ...]:
        return self.dominant_classes + self.tail_classes

    @property
    def images_per_class(self) -> dict[str, int]:
        return {
            **dict.fromkeys(self.dominant_classes, self.images_per_dominant_class),
            **dict.fromkeys(self.tail_classes, self.images_per_tail_class),
        }

    @model_validator(mode="after")
    def _a_class_is_dominant_or_tail_and_not_both(self) -> DatasetsConfig:
        both = sorted(set(self.dominant_classes) & set(self.tail_classes))
        if both:
            raise ValueError(f"{both} are named as dominant and as tail")
        if not self.dominant_classes:
            raise ValueError("dominant_classes is empty")
        if self.images_per_tail_class >= self.images_per_dominant_class:
            raise ValueError(
                f"a tail class drawing {self.images_per_tail_class} images does not sit behind a "
                f"dominant one drawing {self.images_per_dominant_class}"
            )
        return self

    @model_validator(mode="after")
    def _every_condition_is_supplied(self) -> DatasetsConfig:
        if not self.conditions:
            raise ValueError("conditions is empty")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions repeats a name")
        declared = set(self.conditions)
        for name, source in self.sources.items():
            unknown = sorted(set(source.conditions) - declared)
            if unknown:
                raise ValueError(f"source {name} supplies conditions {unknown}, which are not ours")
        supplied = {one for source in self.sources.values() for one in source.conditions}
        missing = sorted(declared - supplied)
        if missing:
            raise ValueError(f"no source supplies {missing}")
        return self

    @model_validator(mode="after")
    def _every_class_comes_from_a_source(self) -> DatasetsConfig:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("classes repeats a name")
        contributed = {name for source in self.sources.values() for name in source.contributes}
        missing = [name for name in self.classes if name not in contributed]
        if missing:
            raise ValueError(f"no source contributes {missing}")
        unknown = [name for name in sorted(contributed) if name not in self.classes]
        if unknown:
            raise ValueError(f"a source contributes unknown classes {unknown}")
        return self


def datasets_config_of(path: Path) -> DatasetsConfig:
    try:
        return DatasetsConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (ValidationError, yaml.YAMLError, OSError) as refused:
        raise ConfigError(f"{path}: {refused}") from refused


def center_crop(image: Image.Image, size: int) -> Image.Image:
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    held = image.crop((left, top, left + side, top + side))
    return held.resize((size, size), Image.Resampling.BICUBIC)


def sample_ids(ids: Sequence[str], wanted: int, *, seed: int) -> list[str]:
    if len(ids) <= wanted:
        return list(ids)
    return sorted(random.Random(seed).sample(list(ids), wanted))  # noqa: S311


def pick_ids(image_ids: Sequence[str], *, count: int, seed: int) -> tuple[str, ...]:
    held = sorted(
        set(image_ids),
        key=lambda one: hashlib.sha256(f"pick:{seed}:{one}".encode()).hexdigest(),
    )
    return tuple(held[:count])


def image_id_of(dataset: str, condition: str, quality: str, name: str) -> str:
    return f"{dataset}-{condition}-{quality}-{name}"


def manifest_rows(hub: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (hub / MANIFEST).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def trained_ids(hub: Path) -> set[str]:
    path = hub / TRAINED
    return set(path.read_text(encoding="utf-8").split()) if path.is_file() else set()
