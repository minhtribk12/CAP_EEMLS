from __future__ import annotations

from pathlib import Path

from eemls.domain.errors import EemlsError


class NoConfigurationTreeError(EemlsError): ...


def tree_above(start: Path) -> Path:
    for folder in (start, *start.parents):
        if (folder / "config" / "services").is_dir() and (folder / "pyproject.toml").is_file():
            return folder / "config"
    raise NoConfigurationTreeError(
        f"no folder above {start} holds config/services and a pyproject.toml"
    )


TREE = tree_above(Path(__file__).resolve())


def committed_tree(target: Path) -> Path:
    for source in TREE.rglob("*.yaml"):
        relative = source.relative_to(TREE)
        if relative.parts[0] == "local":
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return target
