from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = sorted(
    path.parent.relative_to(ROOT).as_posix()
    for pattern in ("application/*", "tools")
    for path in ROOT.glob(f"{pattern}/pyproject.toml")
)


def run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=ROOT, capture_output=True, text=True, check=False)


def sample(tmp_path: Path, body: str, name: str = "sample.py") -> Path:
    target = tmp_path / name
    target.write_text(body, encoding="utf-8")
    return target


def read_pyproject(project: str = ".") -> dict[str, Any]:
    with (ROOT / project / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.parametrize("project", [".", *PROJECTS])
def test_the_installed_distribution_states_the_version_of_the_project(project: str) -> None:
    held = read_pyproject(project)["project"]

    assert metadata.version(held["name"]) == held["version"]


class TestTheTypeGate:
    def test_an_untyped_signature_is_refused(self, tmp_path: Path) -> None:
        target = sample(tmp_path, "def widths(rows):\n    return len(rows)\n")

        result = run(sys.executable, "-m", "mypy", str(target))

        assert result.returncode == 1
        assert "no-untyped-def" in result.stdout

    def test_a_bare_generic_is_refused(self, tmp_path: Path) -> None:
        target = sample(tmp_path, "def widths(rows: list) -> int:\n    return len(rows)\n")

        result = run(sys.executable, "-m", "mypy", str(target))

        assert result.returncode == 1
        assert "type-arg" in result.stdout

    @pytest.mark.parametrize("project", PROJECTS)
    def test_every_package_ships_its_typing_marker(self, project: str) -> None:
        shipped = {one["path"] for one in read_pyproject(project)["tool"]["poetry"]["include"]}
        markers = {
            path.relative_to(ROOT / project).as_posix()
            for path in (ROOT / project / "src").rglob("py.typed")
        }

        assert markers
        assert shipped == markers


class TestTheTestGate:
    def test_an_undeclared_marker_is_refused(self, tmp_path: Path) -> None:
        target = sample(
            tmp_path,
            "import pytest\n\n\n@pytest.mark.not_declared\ndef test_x() -> None:\n    pass\n",
            name="test_marker.py",
        )

        result = run(sys.executable, "-m", "pytest", "-c", "pyproject.toml", str(target), "-q")

        assert result.returncode != 0
        assert "not_declared" in result.stdout

    @pytest.mark.parametrize("marker", ["slow", "integration", "requires_gpu", "requires_docker"])
    def test_the_hardware_markers_are_declared(self, marker: str) -> None:
        declared = read_pyproject()["tool"]["pytest"]["ini_options"]["markers"]

        assert any(entry.startswith(f"{marker}:") for entry in declared)


class TestWhatMakeCheckRuns:
    @pytest.mark.parametrize(
        "gate",
        ["ruff check", "mypy", "codespell", "pytest"],
    )
    def test_every_gate_is_in_the_check_target(self, gate: str) -> None:
        result = run("make", "-n", "check")

        assert result.returncode == 0, result.stderr
        assert gate in result.stdout


@pytest.mark.skipif(not (ROOT / ".git").is_dir(), reason="needs the checkout this tree lives in")
class TestWhatStaysLocal:
    @pytest.mark.parametrize(
        "name",
        [
            ".env.prod",
            "secrets.env",
            "kubeconfig",
            "k3s.yaml",
            "gcp-sa.json",
            "server.key",
            "client.pem",
            "id_rsa",
            "terraform.tfstate",
        ],
    )
    def test_a_credential_file_stays_local(self, name: str) -> None:
        assert run("git", "check-ignore", "-q", name).returncode == 0, name

    @pytest.mark.parametrize(
        "path",
        [
            ".env.example",
            "config/models/model_repository.yaml",
            "tests/fixtures/models/m.onnx",
        ],
    )
    def test_a_versioned_path_is_shared(self, path: str) -> None:
        assert run("git", "check-ignore", "-q", path).returncode == 1, path

    def test_the_example_environment_file_leaves_no_credential_empty(self) -> None:
        body = (ROOT / ".env.example").read_text(encoding="utf-8")

        for line in body.splitlines():
            assert not line.endswith(("PASSWORD=", "TOKEN=", "SECRET=")), line


class TestTheFrozenTreesStayOut:
    @pytest.mark.parametrize("banned", ["simulation", "rohe_rl", "experiment"])
    def test_an_import_of_a_frozen_tree_is_refused(self, tmp_path: Path, banned: str) -> None:
        target = tmp_path / "probe.py"
        target.write_text(f"import {banned}\n", encoding="utf-8")

        result = run(
            sys.executable, "-m", "ruff", "check", "--config", "pyproject.toml", str(target)
        )

        assert result.returncode == 1
        assert "TID251" in result.stdout


def test_the_linter_is_the_same_version_in_both_gates() -> None:
    pinned = read_pyproject()["tool"]["poetry"]["group"]["dev"]["dependencies"]["ruff"]
    hooks = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert pinned == "0.12.12"
    assert f"rev: v{pinned}" in hooks
