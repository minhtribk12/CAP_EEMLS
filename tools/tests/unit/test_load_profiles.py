from __future__ import annotations

from pathlib import Path

import pytest

from eemls.metrics_database.postgres import DatabaseUnavailableError
from eemls.testing.records import profile
from load_profiles import main
from support.repository import ROOT

TREE = ROOT / "config"


def unreachable(*_: object, **__: object) -> None:
    raise DatabaseUnavailableError("the metrics database refused the connection: no one listens")


def test_a_database_that_refuses_the_connection_is_refused_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(profile().model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
    monkeypatch.setattr("load_profiles.PostgresDatabase.from_config", unreachable)

    code = main(["--run", "r1", "--profiles", str(profiles), "--config", str(TREE)])

    assert code == 1
    assert "refused: the metrics database refused the connection" in capsys.readouterr().err


class Recording:
    def __init__(self, **_: object) -> None:
        self.written: list[str] = []
        Recording.last = self

    last: Recording

    @classmethod
    def from_config(cls, *_: object, **__: object) -> Recording:
        return cls()

    def apply_schema(self) -> None:
        return None

    def write_profile(self, one: object) -> None:
        self.written.append(one.service_name)  # type: ignore[attr-defined]

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_only_the_services_of_the_run_s_pool_are_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(
        profile(service_name="infer-b2-general-pi4").model_dump_json()
        + "\n"
        + profile(service_name="infer-not-deployed").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
    monkeypatch.setattr("load_profiles.PostgresDatabase", Recording)

    code = main(["--run", "r1", "--profiles", str(profiles), "--config", str(TREE)])

    assert code == 0
    assert Recording.last.written == ["infer-b2-general-pi4"]
