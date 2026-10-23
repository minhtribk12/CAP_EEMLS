from __future__ import annotations

import argparse
import io
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from eemls.config.reading import ConfigError
from eemls_tools.datasets import QUALITY, DatasetsConfig, datasets_config_of, sample_ids

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "datasets" / "sources.yaml"
IMAGENET = ROOT / "data" / "ilsvrc"
INTO = ROOT / "data" / "imagenet"


class Copied(Enum):
    written = "written"
    already_there = "already there"
    unreadable = "unreadable"


def wanted_synsets(declared: DatasetsConfig) -> dict[str, str]:
    held = declared.sources["imagenet"]
    return {synset: ours for synset, ours in held.label_map.items() if ours in held.contributes}


def available(source: Path, synsets: Mapping[str, str]) -> dict[str, dict[str, Path]]:
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for synset, ours in synsets.items():
        folder = source / synset
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"}:
                found[ours][path.stem] = path
    return found


def copy_image(path: Path, into: Path) -> Copied:
    target = into / f"{path.stem}.jpg"
    if target.exists():
        return Copied.already_there
    body = path.read_bytes()
    try:
        with Image.open(io.BytesIO(body)) as opened:
            if opened.format != "JPEG":
                held = io.BytesIO()
                opened.convert("RGB").save(held, format="JPEG", quality=QUALITY)
                body = held.getvalue()
    except (OSError, UnidentifiedImageError):
        return Copied.unreadable
    part = target.with_suffix(".part")
    part.write_bytes(body)
    part.replace(target)
    return Copied.written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare_imagenet")
    parser.add_argument("--count", action="store_true", help="only say how many are available")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--imagenet", type=Path, default=IMAGENET)
    parser.add_argument("--into", type=Path, default=INTO)
    args = parser.parse_args(argv)

    if not args.imagenet.is_dir():
        print(
            f"refused: no images under {args.imagenet}; download the dataset first", file=sys.stderr
        )
        return 1
    try:
        declared = datasets_config_of(args.sources)
    except ConfigError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    asked_of = declared.images_per_class
    found = available(args.imagenet, wanted_synsets(declared))

    names = args.classes or sorted(name for name in asked_of if name in found)
    print(f"{'class':<16}{'asked':>8}{'available':>11}{'taken':>8}")
    for name in names:
        held = found.get(name, {})
        asked = asked_of.get(name, 0)
        take = sample_ids(sorted(held), asked, seed=args.seed)
        print(f"{name:<16}{asked:>8}{len(held):>11}{len(take):>8}", flush=True)
        if args.count or not take:
            continue
        folder = args.into / name
        folder.mkdir(parents=True, exist_ok=True)
        outcomes = Counter(copy_image(held[one], folder) for one in take)
        print(
            f"  {outcomes[Copied.written]} written, {outcomes[Copied.already_there]} already "
            f"there, {outcomes[Copied.unreadable]} unreadable",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
