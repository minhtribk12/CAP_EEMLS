from __future__ import annotations

import subprocess
from pathlib import Path

from release import main

PROJECT = """\
[project]
name = "eemls"
version = "0.0.1"
requires-python = ">=3.11"

[tool.poetry]
packages = [{ include = "eemls", from = "src" }]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
"""


def git(root: Path, *arguments: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    signing = ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
    done = subprocess.run(
        ["git", *identity, *signing, *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


def repository_at(root: Path) -> Path:
    (root / "src" / "eemls").mkdir(parents=True)
    (root / "src" / "eemls" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(PROJECT, encoding="utf-8")
    (root / "README.md").write_text("framework\n", encoding="utf-8")
    git(root, "init", "--quiet")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", "framework")
    return root


def tags(root: Path) -> list[str]:
    return git(root, "tag", "--list").split()


def test_it_tags_the_version_of_the_root_project(tmp_path: Path) -> None:
    root = repository_at(tmp_path)

    assert main(["0.0.1", "--root", str(root)]) == 0

    assert tags(root) == ["eemls-v0.0.1"]


def test_it_refuses_a_version_other_than_the_root_project_one(tmp_path: Path) -> None:
    root = repository_at(tmp_path)

    assert main(["0.0.2", "--root", str(root)]) == 1

    assert tags(root) == []


def test_it_refuses_a_changed_framework_source(tmp_path: Path) -> None:
    root = repository_at(tmp_path)
    (root / "src" / "eemls" / "__init__.py").write_text("changed = True\n", encoding="utf-8")

    assert main(["0.0.1", "--root", str(root)]) == 1

    assert tags(root) == []


def test_it_refuses_a_new_framework_file(tmp_path: Path) -> None:
    root = repository_at(tmp_path)
    (root / "src" / "eemls" / "new.py").write_text("", encoding="utf-8")

    assert main(["0.0.1", "--root", str(root)]) == 1

    assert tags(root) == []


def test_it_refuses_a_changed_root_project(tmp_path: Path) -> None:
    root = repository_at(tmp_path)
    (root / "pyproject.toml").write_text(PROJECT + "\n", encoding="utf-8")

    assert main(["0.0.1", "--root", str(root)]) == 1

    assert tags(root) == []


def test_a_change_outside_the_framework_does_not_block_it(tmp_path: Path) -> None:
    root = repository_at(tmp_path)
    (root / "README.md").write_text("changed\n", encoding="utf-8")

    assert main(["0.0.1", "--root", str(root)]) == 0

    assert tags(root) == ["eemls-v0.0.1"]


def test_it_refuses_a_version_already_tagged(tmp_path: Path) -> None:
    root = repository_at(tmp_path)
    assert main(["0.0.1", "--root", str(root)]) == 0

    assert main(["0.0.1", "--root", str(root)]) == 1
