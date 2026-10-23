from __future__ import annotations

from pathlib import Path

import pytest

from eemls_consumer.load_test_command import main, rates_of

TREE = Path(__file__).resolve().parents[4] / "config"


def test_the_rates_are_read_from_one_list() -> None:
    assert rates_of("10,20,50,100") == [10.0, 20.0, 50.0, 100.0]


@pytest.mark.parametrize("written", ["", "10,,20", "ten"])
def test_a_list_that_is_not_rates_is_refused(written: str) -> None:
    with pytest.raises(ValueError, match="rates"):
        rates_of(written)


def test_the_run_id_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--config", str(TREE), "--consumer", "c3_surveillance", "--rates", "10"])


def test_a_missing_database_password_is_refused_by_the_name_of_the_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("EEMLS_DATABASE_PASSWORD", raising=False)

    code = main(
        ["--config", str(TREE), "--consumer", "c3_surveillance", "--rates", "10", "--run-id", "r1"]
    )

    assert code == 1
    assert "EEMLS_DATABASE_PASSWORD" in capsys.readouterr().err


def test_a_consumer_the_configuration_does_not_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")

    code = main(
        ["--config", str(TREE), "--consumer", "c9_nobody", "--rates", "10", "--run-id", "r1"]
    )

    assert code == 1
    assert "c9_nobody" in capsys.readouterr().err


def test_the_configuration_is_read_without_any_approach_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
    monkeypatch.setattr(
        "eemls.ensemble_optimization.approach_loading.entry_points", lambda group: []
    )

    code = main(
        ["--config", str(TREE), "--consumer", "c9_nobody", "--rates", "10", "--run-id", "r1"]
    )

    assert code == 1
    assert "c9_nobody is not a consumer" in capsys.readouterr().err
