from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

FAKE = """
from eemls.ensemble_optimization.approach import API_VERSION, Approach, ApproachSettings
from eemls.domain.types import Action
from eemls.ensemble_optimization.approach import PolicyDecision


class FakeSettings(ApproachSettings):
    patience: int


class FakePolicy:
    name = "{name}"

    def decide(self, observed):
        return PolicyDecision(action=Action(kind="keep"))


def build(settings, *, classes, seed):
    return FakePolicy()


approach = Approach(name="{name}", api_version={api_version}, Settings=FakeSettings, build=build)
"""


def installed(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "fake",
    api_version: str = "API_VERSION",
    source: str = FAKE,
) -> None:
    module = f"approach_{root.parent.name}_{root.name}".replace("-", "_")
    (root / f"{module}.py").write_text(
        source.format(name=name, api_version=api_version), encoding="utf-8"
    )
    info = root / f"{module}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module}\nVersion: 1.0\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(
        f"[eemls.approaches]\n{name} = {module}:approach\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(root))
