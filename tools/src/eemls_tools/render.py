from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import yaml

from eemls_tools.class_mapping import with_class_mappings
from eemls_tools.config_tree import ConfigError, ConfigTree, load_config


def rendered(source: Path, overrides: Path, out: Path) -> Path:
    body = yaml.safe_load(overrides.read_text(encoding="utf-8")) or {}
    if not isinstance(body, dict):
        raise ConfigError(f"{overrides} does not map files to the keys they replace")
    rendered_with(source, with_class_mappings(body, overrides.parent), out)
    return out


def rendered_with(source: Path, body: Mapping[str, object], out: Path) -> ConfigTree:
    replaced = _replaced(source, body)
    if out.exists() and any(out.iterdir()):
        raise ConfigError(f"{out} already holds a tree, and a rendered tree is never merged")
    shutil.copytree(source, out, dirs_exist_ok=True)
    for name, held in replaced.items():
        (out / name).write_text(yaml.safe_dump(held, sort_keys=False), encoding="utf-8")
    try:
        return load_config(out)
    except ConfigError:
        shutil.rmtree(out)
        raise


def _replaced(source: Path, body: Mapping[str, object]) -> dict[str, dict[str, object]]:
    replaced: dict[str, dict[str, object]] = {}
    for name, keys in body.items():
        path = source / str(name)
        if not path.is_file():
            raise ConfigError(f"{name}: the tree holds no such file")
        held = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(keys, dict) or not isinstance(held, dict):
            raise ConfigError(f"{name}: an override replaces keys of a mapping")
        unknown = sorted(str(key) for key in keys if key not in held)
        if unknown:
            raise ConfigError(f"{name}: {', '.join(unknown)} is not a key of this file")
        replaced[str(name)] = held | keys
    return replaced
