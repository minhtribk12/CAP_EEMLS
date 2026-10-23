from __future__ import annotations

from pathlib import Path

import pytest

from eemls.testing.trees import committed_tree
from eemls_tools.config_command import check, main
from support.repository import ROOT


def test_it_accepts_the_committed_tree(capsys: pytest.CaptureFixture[str]) -> None:
    assert check(ROOT / "config") == 0
    assert "3 consumers" in capsys.readouterr().out


def test_it_refuses_a_tree_that_is_not_there(capsys: pytest.CaptureFixture[str]) -> None:
    assert check(ROOT / "nowhere") == 1
    assert "refused" in capsys.readouterr().err


def test_it_refuses_a_tree_with_a_wrong_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    target = committed_tree(tmp_path / "config")
    path = target / "approaches" / "cap.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["evaluation_window"] = -1
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    assert check(target) == 1
    assert "evaluation_window" in capsys.readouterr().err


def test_the_command_takes_the_directory_to_check() -> None:
    assert main(["check", str(ROOT / "config")]) == 0


def test_the_command_refuses_anything_but_check_and_render() -> None:
    with pytest.raises(SystemExit):
        main(["explode", str(ROOT / "config")])


def test_render_writes_the_tree_the_overrides_describe(tmp_path: Path) -> None:
    out = tmp_path / "config"

    code = main(
        [
            "render",
            str(ROOT / "config"),
            "--overrides",
            str(ROOT / "deploy" / "one_host" / "overrides.yaml"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert main(["check", str(out)]) == 0


def test_render_refuses_a_key_the_tree_does_not_hold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text("services/control_plane.yaml:\n  nowhere: 1\n", encoding="utf-8")

    code = main(
        [
            "render",
            str(ROOT / "config"),
            "--overrides",
            str(overrides),
            "--out",
            str(tmp_path / "c"),
        ]
    )

    assert code == 1
    assert "refused: services/control_plane.yaml: nowhere" in capsys.readouterr().err


def test_render_needs_overrides_and_a_place_to_write() -> None:
    with pytest.raises(SystemExit):
        main(["render", str(ROOT / "config")])
