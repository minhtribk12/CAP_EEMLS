"""Build the data hub store from the downloaded images.

Usage:
    python3 tools/prepare_dataset.py            build data/hub from what is downloaded
    python3 tools/prepare_dataset.py --check    say what would be built, write nothing
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from eemls.config.reading import ConfigError
from eemls_tools.datasets import (
    MANIFEST,
    QUALITY,
    DatasetsConfig,
    ImageQuality,
    center_crop,
    datasets_config_of,
    image_id_of,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "datasets" / "sources.yaml"
OPEN_IMAGES = ROOT / "data" / "open_images"
IMAGENET = ROOT / "data" / "imagenet"
MULTISPECTRAL = ROOT / "data" / "mfnet_224"
INTO = ROOT / "data" / "hub"


class PrepareError(Exception): ...


def gaussian_kernel(sigma: float, size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise PrepareError(f"a kernel of {size} is not an odd width")
    reach = (size - 1) // 2
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    weights = np.exp(-(offsets**2) / (2.0 * sigma**2))
    normalised: np.ndarray = weights / weights.sum()
    return normalised


def blurred(image: Image.Image, *, sigma: float, kernel: int) -> Image.Image:
    weights = gaussian_kernel(sigma, kernel)
    pixels = np.asarray(image, dtype=np.float64)
    reach = (kernel - 1) // 2
    for axis in (0, 1):
        padded = np.pad(
            pixels, [(reach, reach) if one == axis else (0, 0) for one in range(3)], mode="edge"
        )
        stacked = np.stack(
            [
                np.take(padded, np.arange(index, index + pixels.shape[axis]), axis=axis)
                for index in range(kernel)
            ]
        )
        pixels = np.tensordot(weights, stacked, axes=(0, 0))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def write_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    held = io.BytesIO()
    image.save(held, format="JPEG", quality=QUALITY)
    path.write_bytes(held.getvalue())


def found(
    declared: DatasetsConfig, folders: Mapping[str, tuple[Path, bool]]
) -> Iterator[tuple[str, str, str, str, Path]]:
    for dataset, source in declared.sources.items():
        if dataset not in folders:
            raise PrepareError(f"no folder holds the images of source {dataset}")
        root, by_condition = folders[dataset]
        if not by_condition and len(source.conditions) != 1:
            raise PrepareError(f"{root} holds images of one condition, {dataset} supplies more")
        for condition in source.conditions:
            held = root / condition if by_condition else root
            if not held.is_dir():
                continue
            for folder in sorted(one for one in held.glob("*") if one.is_dir()):
                if folder.name not in declared.classes:
                    continue
                for path in sorted(folder.glob("*.jpg")):
                    yield dataset, condition, folder.name, path.stem, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare_dataset")
    parser.add_argument("--check", action="store_true", help="count what would be built")
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--into", type=Path, default=INTO)
    parser.add_argument("--open-images", type=Path, default=OPEN_IMAGES)
    parser.add_argument("--imagenet", type=Path, default=IMAGENET)
    parser.add_argument("--multispectral", type=Path, default=MULTISPECTRAL)
    args = parser.parse_args(argv)

    try:
        declared = datasets_config_of(args.sources)
    except ConfigError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    counted: dict[tuple[str, str], int] = defaultdict(int)
    rows: list[dict[str, str]] = []
    folders = {
        "open_images": (args.open_images, False),
        "imagenet": (args.imagenet, False),
        "multispectral": (args.multispectral, True),
    }
    try:
        sourced = list(found(declared, folders))
    except PrepareError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    for dataset, condition, label, name, path in sourced:
        counted[(condition, label)] += 1
        if args.check:
            continue
        try:
            opened = Image.open(path).convert("RGB")
        except OSError as refused:
            print(f"refused: {path} is not an image: {refused}", file=sys.stderr)
            return 1
        clear = center_crop(opened, declared.image_size)
        for quality, held in (
            (ImageQuality.clear, clear),
            (
                ImageQuality.blurred,
                blurred(clear, sigma=declared.blur.sigma, kernel=declared.blur.kernel),
            ),
        ):
            image_id = image_id_of(dataset, condition, quality, name)
            relative = f"{condition}/{quality}/{label}/{image_id}.jpg"
            write_image(held, args.into / relative)
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "dataset": dataset,
                    "original_label": name,
                    "condition": condition,
                    "image_quality": quality,
                    "path": relative,
                }
            )

    if not counted:
        print("nothing is downloaded yet", file=sys.stderr)
        return 1
    if not args.check:
        args.into.mkdir(parents=True, exist_ok=True)
        (args.into / MANIFEST).write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in sorted(rows, key=lambda one: one["image_id"])
            ),
            encoding="utf-8",
        )
    print(f"{'class':<16}{'day':>8}{'night':>8}")
    for label in sorted({one for _, one in counted}):
        print(f"{label:<16}{counted[('day', label)]:>8}{counted[('night', label)]:>8}")
    wanted = {
        (condition, label)
        for source in declared.sources.values()
        for condition in source.conditions
        for label in source.contributes
    }
    missing = sorted(one for one in wanted if not counted[one])
    if missing:
        print("missing: " + ", ".join(f"{condition} {label}" for condition, label in missing))
    print(f"\n{sum(counted.values())} source images, {len(rows)} copies written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
