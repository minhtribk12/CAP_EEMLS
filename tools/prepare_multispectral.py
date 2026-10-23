"""Build the multispectral crops from the downloaded dataset.

Usage:
    python3 tools/prepare_multispectral.py    build data/mfnet_224 from data/multispectral
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eemls.config.reading import ConfigError
from eemls_tools.datasets import QUALITY, center_crop, datasets_config_of

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "datasets" / "sources.yaml"
MULTISPECTRAL = ROOT / "data" / "multispectral"
INTO = ROOT / "data" / "mfnet_224"
LEAST_SHARE = 0.005
# Label ids of the dataset's masks, as its README names them.
MASK_LABELS = {
    1: "car",
    2: "person",
    3: "bike",
    4: "curve",
    5: "car_stop",
    6: "guardrail",
    7: "color_cone",
    8: "bump",
}
CONDITIONS = {"D": "day", "N": "night"}
SPLITS = ("train", "val", "test")


def _splits(source: Path) -> dict[str, str]:
    held: dict[str, str] = {}
    for split in SPLITS:
        for name in (source / f"{split}.txt").read_text(encoding="utf-8").split():
            if not name.endswith("_flip"):
                held[name] = split
    return held


def build_crops(
    source: Path, into: Path, *, size: int, label_map: Mapping[str, str]
) -> list[dict[str, Any]]:
    splits = _splits(source)
    black = set((source / "black_list.txt").read_text(encoding="utf-8").split())
    index: list[dict[str, Any]] = []
    for path in sorted((source / "images").glob("*.png")):
        name = path.stem
        labels = source / "labels" / f"{name}.png"
        if name in black or name not in splits or not labels.is_file():
            continue
        mask = np.asarray(Image.open(labels))
        covered = Counter(mask.flatten().tolist())
        covered.pop(0, None)
        if not covered:
            continue
        label_id, pixels = max(covered.items(), key=lambda one: one[1])
        if pixels / mask.size < LEAST_SHARE:
            continue
        label = label_map[MASK_LABELS[label_id]]
        condition = CONDITIONS[name[-1]]
        target = into / condition / label / f"{name}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(path).convert("RGB")
        center_crop(image, size).save(target, "JPEG", quality=QUALITY)
        index.append(
            {
                "name": name,
                "label": label,
                "condition": condition,
                "split": splits[name],
                "path": str(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target),
                "share": round(pixels / mask.size, 4),
            }
        )
    (into / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare_multispectral")
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--multispectral", type=Path, default=MULTISPECTRAL)
    parser.add_argument("--into", type=Path, default=INTO)
    parser.add_argument("--size", type=int, help="side of a crop; default the image_size declared")
    args = parser.parse_args(argv)
    try:
        declared = datasets_config_of(args.sources)
    except ConfigError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    if not (args.multispectral / "images").is_dir():
        print(
            f"refused: no images under {args.multispectral}; download the dataset first",
            file=sys.stderr,
        )
        return 1
    index = build_crops(
        args.multispectral,
        args.into,
        size=args.size or declared.image_size,
        label_map=declared.sources["multispectral"].label_map,
    )
    counted = Counter((one["condition"], one["label"]) for one in index)
    for (condition, label), count in sorted(counted.items()):
        print(f"{condition:<6}{label:<16}{count:>6}")
    print(f"{len(index)} crops written to {args.into}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
