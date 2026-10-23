from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from typing import Any

from eemls.config.reading import ConfigError, Reader, validate
from eemls.domain.errors import EemlsError
from eemls.ensemble_optimization.approach import API_VERSION, Approach, ApproachSettings, Policy

GROUP = "eemls.approaches"


class StartupError(EemlsError): ...


def approaches() -> Mapping[str, Approach[Any]]:
    found: dict[str, Approach[Any]] = {}
    for point in entry_points(group=GROUP):
        try:
            loaded = point.load()
        except Exception as error:
            raise ConfigError(f"approach {point.name} could not be loaded: {error}") from error
        if not isinstance(loaded, Approach):
            raise ConfigError(f"approach {point.name} is not an Approach")
        if loaded.name != point.name:
            raise ConfigError(f"approach {point.name} is installed under the name {loaded.name}")
        if not (
            isinstance(loaded.Settings, type) and issubclass(loaded.Settings, ApproachSettings)
        ):
            raise ConfigError(
                f"approach {loaded.name} states Settings that are not ApproachSettings"
            )
        if loaded.name in found:
            raise ConfigError(f"approach {loaded.name} is installed twice")
        if loaded.api_version != API_VERSION:
            raise ConfigError(
                f"approach {loaded.name} states api_version {loaded.api_version}, "
                f"and this eemls serves {API_VERSION}"
            )
        found[loaded.name] = loaded
    if not found:
        raise ConfigError("no approach is installed; run `poetry install`")
    return found


def settings_of(
    body: dict[str, object], name: str, held: Mapping[str, Approach[Any]]
) -> ApproachSettings:
    kind = body.get("kind")
    if not isinstance(kind, str) or kind not in held:
        raise ConfigError(f"{name}: kind {kind!r} is not one of {sorted(held)}")
    settings: ApproachSettings = validate(held[kind].Settings, body, f"{name}: approach {kind}")
    return settings


def load_approaches(reader: Reader) -> dict[str, ApproachSettings]:
    directory = reader.root / "approaches"
    if not directory.is_dir():
        raise ConfigError("approaches: directory is missing")
    held = approaches()
    found: dict[str, ApproachSettings] = {}
    for path in sorted(directory.glob("*.yaml")):
        name = f"approaches/{path.name}"
        found[path.stem] = settings_of(reader.read_file(name), name, held)
    if not found:
        raise ConfigError("approaches: no approach file found")
    return found


def policy_of(settings: ApproachSettings, *, classes: Sequence[str], seed: int = 0) -> Policy:
    built = approaches()[settings.kind].build(settings, classes=classes, seed=seed)
    if not isinstance(built, Policy):
        raise StartupError(f"approach {settings.kind} builds no policy")
    if built.name != settings.kind:
        raise StartupError(f"approach {settings.kind} builds a policy named {built.name}")
    return built
