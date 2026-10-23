"""Download the day images from Open Images, by class.

Usage:
    python3 tools/fetch_dataset.py --count            how many each class can have
    python3 tools/fetch_dataset.py                    fetch what sources.yaml asks for
    python3 tools/fetch_dataset.py --classes car bus  only those
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from eemls.config.reading import ConfigError
from eemls_tools.datasets import DatasetsConfig, datasets_config_of, sample_ids

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "datasets" / "sources.yaml"
INTO = ROOT / "data" / "open_images"
DESCRIPTIONS = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
LABELS = "https://storage.googleapis.com/openimages/v5/{}-annotations-human-imagelabels-boxable.csv"
IMAGE = "https://open-images-dataset.s3.amazonaws.com/{}/{}.jpg"
SPLITS = ("train", "validation", "test")
TIMEOUT_S = 120


class FetchError(Exception): ...


class FetchedImage(Enum):
    downloaded = "downloaded"
    already_there = "already there"
    failed = "failed"


def streamed(url: str) -> Iterator[str]:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as answer:  # noqa: S310
            yield from io.TextIOWrapper(answer, encoding="utf-8")
    except urllib.error.URLError as refused:
        raise FetchError(f"{url} could not be read: {refused}") from refused


def wanted_classes(declared: DatasetsConfig) -> dict[str, str]:
    by_name = dict(declared.sources["open_images"].label_map)
    found: dict[str, str] = {}
    for row in csv.reader(streamed(DESCRIPTIONS)):
        if len(row) == 2 and row[1] in by_name:
            found[row[0]] = by_name[row[1]]
    missing = sorted(set(by_name.values()) - set(found.values()))
    if missing:
        raise FetchError(f"Open Images names no label for {missing}")
    return found


def wanted_rows(split: str, ours: Mapping[str, str]) -> dict[str, list[str]]:
    marked: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(streamed(LABELS.format(split))):
        name = ours.get(row["LabelName"])
        if name is not None and row["Confidence"] == "1":
            marked[row["ImageID"]].add(name)
    found: dict[str, list[str]] = defaultdict(list)
    for image_id, names in marked.items():
        if len(names) == 1:
            found[next(iter(names))].append(f"{split}/{image_id}")
    for ids in found.values():
        ids.sort()
    return found


def available(ours: Mapping[str, str], splits: Sequence[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = defaultdict(list)
    for split in splits:
        print(f"  reading the {split} labels", file=sys.stderr, flush=True)
        for name, ids in wanted_rows(split, ours).items():
            found[name].extend(ids)
    for ids in found.values():
        ids.sort()
    return found


def fetch_one(named: str, into: Path) -> FetchedImage:
    split, image_id = named.split("/", 1)
    path = into / f"{image_id}.jpg"
    if path.exists():
        return FetchedImage.already_there
    try:
        with urllib.request.urlopen(IMAGE.format(split, image_id), timeout=60) as answer:  # noqa: S310
            body = answer.read()
        Image.open(io.BytesIO(body)).verify()
    except (urllib.error.URLError, OSError, UnidentifiedImageError, ValueError):
        return FetchedImage.failed
    held = path.with_suffix(".part")
    held.write_bytes(body)
    held.replace(path)
    return FetchedImage.downloaded


def set_aside(folder: Path, picked: Sequence[str], into: Path) -> int:
    kept = {f"{named.split('/', 1)[1]}.jpg" for named in picked}
    unpicked = [path for path in sorted(folder.glob("*.jpg")) if path.name not in kept]
    if unpicked:
        into.mkdir(parents=True, exist_ok=True)
    for path in unpicked:
        path.replace(into / path.name)
    return len(unpicked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch_dataset")
    parser.add_argument("--count", action="store_true", help="only say how many are available")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--into", type=Path, default=INTO)
    args = parser.parse_args(argv)

    try:
        declared = datasets_config_of(args.sources)
        ours = wanted_classes(declared)
        found = available(ours, args.splits)
    except (ConfigError, FetchError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1

    asked_of = declared.images_per_class
    failed = 0
    names = args.classes or sorted(name for name in asked_of if name in found)
    print(f"{'class':<16}{'asked':>8}{'available':>11}{'taken':>8}")
    for name in names:
        asked = asked_of.get(name, 0)
        take = sample_ids(found.get(name, ()), asked, seed=args.seed)
        print(f"{name:<16}{asked:>8}{len(found.get(name, ())):>11}{len(take):>8}", flush=True)
        if args.count:
            continue
        folder = args.into / name
        folder.mkdir(parents=True, exist_ok=True)
        aside = args.into / "unpicked" / name
        moved = set_aside(folder, take, aside)
        if moved:
            print(f"  {moved} not picked, moved to {aside}", flush=True)
        outcomes = Counter(fetch_one(one, folder) for one in take)
        failed += outcomes[FetchedImage.failed]
        print(
            f"  {outcomes[FetchedImage.downloaded]} downloaded, "
            f"{outcomes[FetchedImage.already_there]} already there, "
            f"{outcomes[FetchedImage.failed]} failed",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
