from __future__ import annotations

from pathlib import Path

ROOT = next(
    folder
    for folder in Path(__file__).resolve().parents
    if (folder / "deploy").is_dir() and (folder / "tools").is_dir() and (folder / "config").is_dir()
)

PROJECTS = ("src", "application/*/src", "tools/src")


def sources() -> list[Path]:
    return sorted(
        folder
        for pattern in PROJECTS
        for folder in ROOT.glob(f"{pattern}/eemls*")
        if folder.is_dir()
    )
