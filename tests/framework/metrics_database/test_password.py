from __future__ import annotations

import pytest

from eemls.metrics_database import MissingPasswordError, password_from_environment


def test_the_password_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "held")

    assert password_from_environment() == "held"


def test_a_missing_password_is_refused_by_the_name_of_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EEMLS_DATABASE_PASSWORD", raising=False)

    with pytest.raises(MissingPasswordError, match="EEMLS_DATABASE_PASSWORD"):
        password_from_environment()


def test_an_empty_password_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "")

    with pytest.raises(MissingPasswordError):
        password_from_environment()
