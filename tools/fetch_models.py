"""Download the backbones and the pool's models, and check each SHA-256 (config/local/models.yaml).

Usage:
    python3 tools/fetch_models.py                 every backbone, then the pool
    python3 tools/fetch_models.py b1              one backbone, into models/base/b1.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import sys
import urllib.request
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path

from eemls.config.models import ModelEntry, ModelRepositoryConfig
from eemls_tools.config_tree import ConfigError, load_config
from eemls_tools.model_config import BaseModelEntry

ROOT = Path(__file__).resolve().parents[1]
INTO = ROOT / "models"
TIMEOUT_S = 60

Sourced = ModelEntry | BaseModelEntry


class FetchError(Exception): ...


class Fetched(Enum):
    downloaded = "downloaded"
    already_there = "already there"
    built_here = "built here, not fetched"


def url_of(entry: Sourced) -> str:
    return f"https://huggingface.co/{entry.hugging_face_id}/resolve/main/{entry.onnx_file}"


def over_https(url: str) -> bytes:
    if not url.startswith("https://"):
        raise FetchError(f"{url} is not an https URL")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as answer:  # noqa: S310
        return bytes(answer.read())


def fetch_one(
    name: str,
    entry: ModelEntry,
    *,
    into: Path,
    download: Callable[[str], bytes] = over_https,
) -> Fetched:
    if entry.variant_of is not None:
        return Fetched.built_here
    return _fetched(name, entry, into / name / str(entry.onnx_file), download)


def fetch_backbone(
    name: str,
    entry: BaseModelEntry,
    *,
    into: Path,
    download: Callable[[str], bytes] = over_https,
) -> Fetched:
    return _fetched(name, entry, into / "base" / f"{name}.onnx", download)


def _fetched(name: str, entry: Sourced, path: Path, download: Callable[[str], bytes]) -> Fetched:
    wanted = str(entry.sha256)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == wanted:
        return Fetched.already_there
    try:
        body = download(url_of(entry))
    except (FetchError, OSError, http.client.HTTPException) as refused:
        raise FetchError(f"{name}: {url_of(entry)} could not be read: {refused}") from refused
    found = hashlib.sha256(body).hexdigest()
    if found != wanted:
        raise FetchError(f"{name}: the file is {found}, and the pool declares {wanted}")
    path.parent.mkdir(parents=True, exist_ok=True)
    held = path.with_suffix(path.suffix + ".part")
    held.write_bytes(body)
    held.replace(path)
    return Fetched.downloaded


def _has_source(entry: Sourced) -> bool:
    return None not in (entry.hugging_face_id, entry.onnx_file, entry.sha256)


def declared_models(repository: ModelRepositoryConfig) -> dict[str, ModelEntry]:
    missing = sorted(
        name
        for name, entry in repository.models.items()
        if entry.variant_of is None and not _has_source(entry)
    )
    if missing:
        raise FetchError(f"{missing} have no source and digest in local/models.yaml")
    return dict(repository.models)


def declared_backbones(base_models: Mapping[str, BaseModelEntry]) -> dict[str, BaseModelEntry]:
    return {name: entry for name, entry in base_models.items() if _has_source(entry)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch_models")
    parser.add_argument("names", nargs="*", help="the models to fetch, or all of them")
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--into", type=Path, default=INTO)
    args = parser.parse_args(argv)

    try:
        models = load_config(args.config).models
        backbones = declared_backbones(models.variations.base_models)
        declared = declared_models(models.model_repository)
    except (ConfigError, FetchError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    wanted = args.names or [*sorted(backbones), *sorted(declared)]
    unknown = [name for name in wanted if name not in backbones and name not in declared]
    if unknown:
        print(f"refused: no backbone or pool model is named {unknown}", file=sys.stderr)
        return 1
    for name in wanted:
        try:
            if name in backbones:
                fetched = fetch_backbone(name, backbones[name], into=args.into)
            else:
                fetched = fetch_one(name, declared[name], into=args.into)
        except FetchError as refused:
            print(f"refused: {refused}", file=sys.stderr)
            return 1
        print(f"{name}: {fetched.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
