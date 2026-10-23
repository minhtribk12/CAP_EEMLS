from __future__ import annotations

import pytest

from eemls.testing.environment import no_ambient_environment
from support.repository import ROOT

__all__ = ["no_ambient_environment"]


@pytest.fixture(autouse=True)
def at_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
