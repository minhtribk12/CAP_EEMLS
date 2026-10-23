from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

from eemls.domain.types import FrozenMap, FrozenModel, NonNegativeCount, PositiveCount
from eemls_tools.config_tree import ConfigError

REPOSITORY = "models/model_repository.yaml"

Outputs = tuple[NonNegativeCount | tuple[NonNegativeCount, NonNegativeCount], ...]


class ClassMapping(FrozenModel):
    outputs: PositiveCount
    classes: FrozenMap[Outputs]


def classes_of(path: Path) -> list[str]:
    if not path.is_file():
        raise ConfigError(f"{path}: no such file")
    try:
        held = ClassMapping.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (ValidationError, yaml.YAMLError) as refused:
        raise ConfigError(f"{path}: {refused}") from refused
    names: dict[int, str] = {}
    for name, outputs in held.classes.items():
        for output in outputs:
            first, last = (output, output) if isinstance(output, int) else output
            if first > last:
                raise ConfigError(f"{path}: {name} gives the empty range {first}-{last}")
            for index in range(first, last + 1):
                if index >= held.outputs:
                    raise ConfigError(f"{path}: output {index} is past the {held.outputs} outputs")
                if index in names:
                    raise ConfigError(f"{path}: output {index} maps to more than one class")
                names[index] = name
    return [names.get(index, f"output_{index}") for index in range(held.outputs)]


def with_class_mappings(body: Mapping[str, object], folder: Path) -> dict[str, object]:
    keys = body.get(REPOSITORY)
    models = keys.get("models") if isinstance(keys, dict) else None
    if not isinstance(keys, dict) or not isinstance(models, dict):
        return dict(body)
    return {**body, REPOSITORY: keys | {"models": _expanded(models, folder)}}


def _expanded(models: Mapping[str, object], folder: Path) -> dict[str, object]:
    expanded: dict[str, object] = {}
    for name, entry in models.items():
        if isinstance(entry, dict) and "class_mapping" in entry:
            if "classes" in entry:
                raise ConfigError(f"{REPOSITORY}: {name}: both classes and a class mapping")
            classes = classes_of(folder / str(entry["class_mapping"]))
            entry = {
                ("classes" if key == "class_mapping" else key): (
                    classes if key == "class_mapping" else value
                )
                for key, value in entry.items()
            }
        expanded[name] = entry
    return expanded
