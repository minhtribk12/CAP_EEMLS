from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

LABELS = ("pedestrian", "car")


def picture(colour: tuple[int, int, int] = (90, 90, 90)) -> bytes:
    held = io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(held, format="JPEG")
    return held.getvalue()


def prepared(root: Path, *, rows: list[dict[str, object]] | None = None) -> Path:
    written = rows if rows is not None else _default_rows()
    for row in written:
        path = root / str(row["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(picture())
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in written), encoding="utf-8"
    )
    return root


def _default_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        for condition in ("day", "night"):
            for quality in ("clear", "blurred"):
                for index in range(2):
                    image_id = f"{label}-{condition}-{quality}-{index}"
                    rows.append(
                        {
                            "image_id": image_id,
                            "label": label,
                            "dataset": "open_images",
                            "original_label": f"/m/{label}",
                            "condition": condition,
                            "image_quality": quality,
                            "path": f"{condition}/{quality}/{label}/{image_id}.jpg",
                        }
                    )
    return rows
