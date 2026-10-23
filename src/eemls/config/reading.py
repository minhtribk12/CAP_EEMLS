from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from eemls.config.models import InferenceServicesConfig, ModelSource, ModelSourcesConfig
from eemls.config.schema import FrozenModel
from eemls.domain.errors import EemlsError

ENV_VAR = "EEMLS_CONFIG_DIR"
MODEL_SOURCES = "local/models.yaml"
SOURCES_VAR = "EEMLS_MODEL_SOURCES"

_T = TypeVar("_T", bound=FrozenModel)


class ConfigError(EemlsError): ...


def root_of(directory: Path | str | None) -> Path:
    named = directory if directory is not None else os.environ.get(ENV_VAR)
    if named is None:
        raise ConfigError(f"no configuration directory given and {ENV_VAR} is not set")
    root = Path(named)
    if not root.is_dir():
        raise ConfigError(f"{root}: configuration directory is missing")
    return root


def _refuse_repeated_keys(node: yaml.Node) -> None:
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = str(getattr(key_node, "value", ""))
            if key in seen:
                raise yaml.YAMLError(f"key {key!r} appears twice")
            seen.add(key)
            _refuse_repeated_keys(value_node)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _refuse_repeated_keys(item)


class Reader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.read: set[Path] = set()

    def read_file(self, name: str) -> dict[str, object]:
        path = self.root / name
        self.read.add(path)
        try:
            if not path.is_file():
                raise ConfigError(f"{name}: file is missing")
            loader = yaml.SafeLoader(path.read_text(encoding="utf-8"))
            try:
                node = loader.get_single_node()
                if node is not None:
                    _refuse_repeated_keys(node)
                body = loader.construct_document(node) if node is not None else None
            finally:
                loader.dispose()
        except yaml.YAMLError as error:
            raise ConfigError(f"{name}: {error}") from error
        except (OSError, UnicodeDecodeError) as error:
            raise ConfigError(f"{name}: could not be read: {error}") from error
        if body is None:
            raise ConfigError(f"{name}: file is empty")
        if not isinstance(body, dict):
            raise ConfigError(f"{name}: file is not a mapping of settings")
        return body

    def build(self, model: type[_T], name: str) -> _T:
        return validate(model, self.read_file(name), name)

    def read_with_model_sources(self, sections: Mapping[str, str]) -> dict[str, dict[str, object]]:
        bodies = {name: self.read_file(name) for name in sections}
        named = os.environ.get(SOURCES_VAR) or MODEL_SOURCES
        if named != MODEL_SOURCES:
            self.read.add(self.root / MODEL_SOURCES)
        if not (self.root / named).is_file():
            return bodies
        sources = validate(ModelSourcesConfig, self.read_file(named), named)
        for name, section in sections.items():
            given: Mapping[str, ModelSource] = getattr(sources, section)
            held = bodies[name].get(section)
            entries = held if isinstance(held, dict) else {}
            unknown = sorted(set(given) - set(entries))
            if unknown:
                raise ConfigError(f"{named}: {section} {unknown}, which {name} does not hold")
            for key, values in given.items():
                entry = entries[key]
                if isinstance(entry, dict):
                    entries[key] = entry | values.model_dump()
        return bodies

    def refuse_unread_files(self) -> None:
        found = {
            path
            for path in (*self.root.rglob("*.yaml"), *self.root.rglob("*.yml"))
            if not any(part.startswith(".") for part in path.relative_to(self.root).parts)
        }
        stray = sorted(str(path.relative_to(self.root)) for path in found - self.read)
        if stray:
            raise ConfigError(f"files that nothing reads: {stray}")


def validate(model: type[_T], body: object, name: str) -> _T:
    try:
        return model.model_validate(body)
    except ValidationError as error:
        raise _refusal(name, error) from error


def _refusal(name: str, error: ValidationError) -> ConfigError:
    details = "; ".join(_one_refusal(item) for item in error.errors())
    return ConfigError(f"{name}: {details}")


def _one_refusal(item: ErrorDetails) -> str:
    key = ".".join(str(part) for part in item["loc"]) or "<root>"
    given = item.get("input")
    if isinstance(given, str | int | float | bool | None) and item["type"] != "missing":
        return f"{key}: {item['msg']}, and was given {given!r}"
    return f"{key}: {item['msg']}"


def read_services(reader: Reader, model: type[_T]) -> _T:
    bodies = {name: reader.read_file(f"services/{name}.yaml") for name in model.model_fields}
    return validate(model, bodies, "services/")


def check_the_device_types_are_priced(
    inference_services: InferenceServicesConfig, devices: Mapping[str, object], source: str
) -> None:
    for name, service in inference_services.services.items():
        if service.device_type not in devices:
            raise ConfigError(
                f"models/inference_services.yaml: {name} names device type "
                f"{service.device_type}, which {source} does not hold"
            )
