from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def no_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [key for key in os.environ if key.startswith("EEMLS_")]:
        monkeypatch.delenv(name, raising=False)
