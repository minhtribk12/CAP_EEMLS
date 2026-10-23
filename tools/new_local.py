"""Write empty config/local/models.yaml and images.yaml to fill in.

Usage:
    python3 tools/new_local.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {"hugging_face_id": "", "onnx_file": "", "sha256": ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="new_local")
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    args = parser.parse_args(argv)
    local = args.config / "local"
    there = sorted(str(path) for path in local.glob("*.yaml"))
    if there:
        print(f"refused: {there} already there, and is never overwritten", file=sys.stderr)
        return 1
    repository = yaml.safe_load((args.config / "models" / "model_repository.yaml").read_text())
    recipe = yaml.safe_load((args.config / "models" / "variations.yaml").read_text())
    models = {
        "models": {name: dict(SOURCE) for name in sorted(repository["models"])},
        "base_models": {name: {"name": ""} | SOURCE for name in sorted(recipe["base_models"])},
    }
    local.mkdir(parents=True, exist_ok=True)
    (local / "models.yaml").write_text(
        yaml.safe_dump(models, sort_keys=False),
        encoding="utf-8",
    )
    (local / "images.yaml").write_text(
        yaml.safe_dump({"registry": "", "version": "latest"}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"{local}: fill in models.yaml and images.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
