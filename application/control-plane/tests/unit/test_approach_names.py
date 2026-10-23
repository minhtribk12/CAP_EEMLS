from __future__ import annotations

import re
from importlib.metadata import entry_points
from pathlib import Path

from eemls.ensemble_optimization.approach_loading import GROUP

APPLICATION = Path(__file__).resolve().parents[2]
CONTROL_PLANE = APPLICATION / "src" / "eemls_control_plane"


def test_no_approach_is_named_in_the_control_plane_code() -> None:
    installed = sorted(point.name for point in entry_points(group=GROUP))
    assert installed
    names = [re.escape(one) for one in installed]
    named = re.compile(
        rf"\b({'|'.join(names + [one.replace('_', '') for one in names])})\b", re.IGNORECASE
    )

    found = [
        f"{path.relative_to(APPLICATION)}:{line}: {match.group(0)}"
        for path in sorted(CONTROL_PLANE.rglob("*.py"))
        for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for match in named.finditer(text)
    ]

    assert not found
