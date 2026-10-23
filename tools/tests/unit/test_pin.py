from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from pin import (
    ROOT,
    built_wheel,
    framework_contexts,
    framework_line,
    framework_version,
    main,
    pinned_project,
)

PROJECT = """\
[project]
name = "eemls-app"
version = "1.0.0"

dependencies = [
    "eemls{extras} (>=1.0,<2)",
    "pydantic (>=2.11.7,<3.0.0)",
]

[tool.poetry]
requires-poetry = ">=2.0"
packages = [{{ include = "eemls_app", from = "src" }}]

[tool.poetry.group.dev.dependencies]
pytest = "^9.0.3"
"""


def project(extras: str = "") -> str:
    return PROJECT.format(extras=extras)


def parsed(text: str) -> dict[str, object]:
    return tomllib.loads(text)


def poetry_dependencies(text: str) -> dict[str, object]:
    found: dict[str, object] = tomllib.loads(text)["tool"]["poetry"]["dependencies"]
    return found


def test_the_framework_requirement_becomes_an_exact_pin_keeping_its_extras() -> None:
    pinned = pinned_project(project("[database,approaches]"), "1.2.0")

    assert parsed(pinned)["project"]["dependencies"] == [  # type: ignore[index]
        "eemls[database,approaches]==1.2.0",
        "pydantic (>=2.11.7,<3.0.0)",
    ]


def test_a_requirement_without_extras_becomes_a_plain_exact_pin() -> None:
    pinned = pinned_project(project(), "1.2.0")

    assert parsed(pinned)["project"]["dependencies"][0] == "eemls==1.2.0"  # type: ignore[index]


def test_poetry_takes_the_framework_from_the_repository_root_with_the_same_extras() -> None:
    pinned = pinned_project(project("[database,approaches]"), "1.2.0")

    assert poetry_dependencies(pinned) == {
        "eemls": {"path": "../..", "develop": False, "extras": ["database", "approaches"]}
    }


def test_a_requirement_without_extras_takes_the_repository_root_alone() -> None:
    pinned = pinned_project(project(), "1.2.0")

    assert poetry_dependencies(pinned) == {"eemls": {"path": "../..", "develop": False}}


def test_a_new_pin_replaces_the_old_one() -> None:
    pinned = pinned_project(pinned_project(project("[database]"), "1.2.0"), "1.3.0")

    assert parsed(pinned)["project"]["dependencies"][0] == "eemls[database]==1.3.0"  # type: ignore[index]
    assert poetry_dependencies(pinned) == {
        "eemls": {"path": "../..", "develop": False, "extras": ["database"]}
    }


def test_pinning_leaves_the_rest_of_the_project_as_it_was() -> None:
    pinned = pinned_project(project(), "1.2.0")

    assert parsed(pinned)["tool"]["poetry"]["group"] == parsed(project())["tool"]["poetry"]["group"]  # type: ignore[index]
    assert pinned.count("[tool.poetry.dependencies]") == 1


def test_the_framework_file_names_the_pin_and_the_hash_of_its_wheel() -> None:
    digest = "ab" * 32

    assert framework_line("1.2.0", ["database", "approaches"], digest) == (
        f"eemls[database,approaches]==1.2.0 --hash=sha256:{digest}\n"
    )
    assert framework_line("1.2.0", [], digest) == f"eemls==1.2.0 --hash=sha256:{digest}\n"


FRAMEWORK = """\
[project]
name = "eemls"
version = "{version}"
requires-python = ">=3.11"

[tool.poetry]
packages = [{{ include = "eemls", from = "src" }}]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
"""

APPLICATION = """\
[project]
name = "eemls-app"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = ["eemls (>=0.0,<1)"]

[tool.poetry]
packages = [{ include = "eemls_app", from = "src" }]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
"""


def framework_at(root: Path, body: str = "", version: str = "0.0.1") -> Path:
    (root / "src" / "eemls").mkdir(parents=True, exist_ok=True)
    (root / "src" / "eemls" / "__init__.py").write_text(body, encoding="utf-8")
    (root / "pyproject.toml").write_text(FRAMEWORK.format(version=version), encoding="utf-8")
    return root


def application_at(root: Path) -> Path:
    folder = root / "application" / "app"
    (folder / "src" / "eemls_app").mkdir(parents=True)
    (folder / "src" / "eemls_app" / "__init__.py").write_text("", encoding="utf-8")
    (folder / "pyproject.toml").write_text(APPLICATION, encoding="utf-8")
    return folder


def git(root: Path, *arguments: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    signing = ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
    subprocess.run(
        ["git", *identity, *signing, *arguments], cwd=root, check=True, capture_output=True
    )


def tagged(root: Path, version: str) -> None:
    git(root, "init", "--quiet")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", "framework")
    git(root, "tag", f"eemls-v{version}")


def digest_of(wheel: Path) -> str:
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


@pytest.fixture
def no_virtual_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POETRY_VIRTUALENVS_CREATE", "false")


def test_it_refuses_an_application_that_does_not_exist(tmp_path: Path) -> None:
    framework_at(tmp_path)

    assert main(["missing", "0.0.1", "--root", str(tmp_path)]) == 1


def test_the_version_of_the_root_project_is_built_from_the_working_tree(tmp_path: Path) -> None:
    root = framework_at(tmp_path / "root")
    before = built_wheel(root, "0.0.1", tmp_path / "before")
    framework_at(root, "changed = True\n")

    after = built_wheel(root, "0.0.1", tmp_path / "after")

    assert after.name == "eemls-0.0.1-py3-none-any.whl"
    assert digest_of(after) != digest_of(before)


def test_a_readme_change_does_not_change_the_framework_wheel(tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil.copytree(
        ROOT / "src" / "eemls", root / "src" / "eemls", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(ROOT / name, root / name)
    version = framework_version(root)
    before = built_wheel(root, version, tmp_path / "before")
    (root / "README.md").write_text("changed\n", encoding="utf-8")

    after = built_wheel(root, version, tmp_path / "after")

    assert digest_of(after) == digest_of(before)


def test_another_version_is_built_from_its_tag(tmp_path: Path) -> None:
    root = framework_at(tmp_path / "root")
    tagged(root, "0.0.1")
    released = built_wheel(root, "0.0.1", tmp_path / "released")
    framework_at(root, "changed = True\n", version="0.0.2")

    rebuilt = built_wheel(root, "0.0.1", tmp_path / "rebuilt")

    assert rebuilt.name == "eemls-0.0.1-py3-none-any.whl"
    assert digest_of(rebuilt) == digest_of(released)


def test_it_refuses_a_version_without_its_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = framework_at(tmp_path)
    tagged(root, "0.0.1")
    folder = application_at(root)

    assert main(["app", "0.0.2", "--root", str(root)]) == 1

    assert "eemls-v0.0.2" in capsys.readouterr().err
    assert not (folder / "framework.txt").exists()


def pinned_digest(folder: Path) -> str:
    line = (folder / "framework.txt").read_text(encoding="utf-8")
    found = re.fullmatch(r"eemls==\S+ --hash=sha256:([0-9a-f]{64})\n", line)
    assert found is not None, line
    return found.group(1)


def lock_check(folder: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["poetry", "check", "--lock"], cwd=folder, capture_output=True, text=True, check=False
    )


def locked_framework(folder: Path) -> str:
    lock = (folder / "poetry.lock").read_text(encoding="utf-8")
    found = re.search(r'^name = "eemls"\nversion = "([^"]+)"', lock, flags=re.MULTILINE)
    assert found is not None
    return found.group(1)


@pytest.mark.usefixtures("no_virtual_environment")
def test_the_framework_file_names_the_hash_of_a_fresh_build_of_the_root_version(
    tmp_path: Path,
) -> None:
    root = framework_at(tmp_path / "root")
    folder = application_at(root)

    assert main(["app", "0.0.1", "--root", str(root)]) == 0

    assert pinned_digest(folder) == digest_of(built_wheel(root, "0.0.1", tmp_path / "fresh"))


@pytest.mark.usefixtures("no_virtual_environment")
def test_pinning_leaves_no_wheel_in_the_application(tmp_path: Path) -> None:
    framework_at(tmp_path)
    folder = application_at(tmp_path)

    assert main(["app", "0.0.1", "--root", str(tmp_path)]) == 0

    assert not list(tmp_path.rglob("*.whl"))
    assert not (folder / "pinned").exists()


@pytest.mark.usefixtures("no_virtual_environment")
def test_the_lock_takes_the_framework_from_the_repository_root(tmp_path: Path) -> None:
    framework_at(tmp_path)
    folder = application_at(tmp_path)

    assert main(["app", "0.0.1", "--root", str(tmp_path)]) == 0

    assert 'type = "directory"\nurl = "../.."' in (folder / "poetry.lock").read_text(
        encoding="utf-8"
    )
    assert lock_check(folder).returncode == 0


@pytest.mark.usefixtures("no_virtual_environment")
def test_an_older_version_is_locked_from_its_tag(tmp_path: Path) -> None:
    root = framework_at(tmp_path / "root")
    folder = application_at(root)
    tagged(root, "0.0.1")
    released = built_wheel(root, "0.0.1", tmp_path / "released")
    framework_at(root, "changed = True\n", version="0.0.2")

    assert main(["app", "0.0.1", "--root", str(root)]) == 0

    assert pinned_digest(folder) == digest_of(released)
    assert locked_framework(folder) == "0.0.1"
    assert lock_check(folder).returncode == 0


def pinned_by_hand(folder: Path, version: str, digest: str = "0" * 64) -> None:
    line = f"eemls=={version} --hash=sha256:{digest}\n"
    (folder / "framework.txt").write_text(line, encoding="utf-8")


def pinned_to_build(root: Path, folder: Path, version: str, output: Path) -> None:
    pinned_by_hand(folder, version, digest_of(built_wheel(root, version, output)))


def test_an_application_pinning_the_root_version_takes_the_root_as_its_framework(
    tmp_path: Path,
) -> None:
    root = framework_at(tmp_path / "root")
    pinned_to_build(root, application_at(root), "0.0.1", tmp_path / "wheel")

    assert framework_contexts(root, tmp_path / "contexts") == {"FRAMEWORK_APP": "."}


def test_an_application_pinning_an_older_version_takes_a_checkout_of_its_tag(
    tmp_path: Path,
) -> None:
    root = framework_at(tmp_path / "root")
    pinned_to_build(root, application_at(root), "0.0.1", tmp_path / "wheel")
    tagged(root, "0.0.1")
    framework_at(root, "changed = True\n", version="0.0.2")

    contexts = framework_contexts(root, tmp_path / "contexts")

    source = Path(contexts["FRAMEWORK_APP"])
    assert 'version = "0.0.1"' in (source / "pyproject.toml").read_text(encoding="utf-8")
    assert (source / "src" / "eemls" / "__init__.py").read_text(encoding="utf-8") == ""


def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "bin"
    folder.mkdir()
    seen = tmp_path / "seen"
    docker = folder / "docker"
    docker.write_text(
        f'#!/bin/sh\necho "$* $FRAMEWORK_APP" > {seen}\n[ -f "$FRAMEWORK_APP/pyproject.toml" ] '
        f"&& echo found >> {seen}\nexit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")
    return seen


def test_the_bake_runs_with_the_checkout_of_each_older_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = framework_at(tmp_path / "root")
    pinned_to_build(root, application_at(root), "0.0.1", tmp_path / "wheel")
    tagged(root, "0.0.1")
    framework_at(root, version="0.0.2")
    seen = fake_docker(tmp_path, monkeypatch)

    assert main(["--root", str(root), "--bake", "--load", "local"]) == 0

    called, found = seen.read_text(encoding="utf-8").splitlines()
    assert called.startswith("buildx bake --load local /")
    assert found == "found"
    assert not Path(called.split()[-1]).exists()


def test_the_bake_refuses_in_one_line_an_application_whose_tag_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = framework_at(tmp_path)
    pinned_by_hand(application_at(root), "0.0.3")
    tagged(root, "0.0.1")
    seen = fake_docker(tmp_path, monkeypatch)

    assert main(["--root", str(root), "--bake", "--load", "local"]) == 1

    refused = capsys.readouterr().err.splitlines()
    assert len(refused) == 1
    assert "eemls-v0.0.3" in refused[0]
    assert not seen.exists()


def test_the_bake_refuses_in_one_line_an_application_whose_hash_its_source_does_not_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = framework_at(tmp_path / "root")
    pinned_by_hand(application_at(root), "0.0.1")
    built = digest_of(built_wheel(root, "0.0.1", tmp_path / "wheel"))
    seen = fake_docker(tmp_path, monkeypatch)

    assert main(["--root", str(root), "--bake", "--load", "local"]) == 1

    assert capsys.readouterr().err.splitlines() == [
        f"refused: app pins eemls 0.0.1 with hash {'0' * 12}, but the source builds "
        f"{built[:12]}: run make pin APP=app VERSION=0.0.1 (or a new version)"
    ]
    assert not seen.exists()


def test_the_bake_refuses_an_application_whose_tag_builds_another_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = framework_at(tmp_path / "root")
    folder = application_at(root)
    pinned_to_build(root, folder, "0.0.1", tmp_path / "wheel")
    framework_at(root, "changed = True\n")
    tagged(root, "0.0.1")
    framework_at(root, version="0.0.2")
    seen = fake_docker(tmp_path, monkeypatch)

    assert main(["--root", str(root), "--bake", "--load", "local"]) == 1

    assert "run make pin APP=app VERSION=0.0.1" in capsys.readouterr().err
    assert not seen.exists()
