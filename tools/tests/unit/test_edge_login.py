from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from support.repository import ROOT

TOKEN = "dckr_pat_test0000000000000000000000000"  # noqa: S105

FAKE_SSH = """#!/bin/sh
printf '%s\\n' "$@" > "$(dirname "$0")/args"
cat > "$(dirname "$0")/stdin"
"""
# poetry runs python3 in this interpreter.
POETRY = f'#!/bin/sh\nshift 2\nexec {sys.executable} "$@"\n'


def logged_in(root: Path, env_line: str) -> subprocess.CompletedProcess[str]:
    (root / "deploy" / "edge").mkdir(parents=True)
    shutil.copy(ROOT / "deploy" / "edge" / "login.sh", root / "deploy" / "edge")
    (root / "config" / "local").mkdir(parents=True)
    (root / "config" / "local" / "images.yaml").write_text(
        "registry: docker.io/someone\nversion: latest\n", encoding="utf-8"
    )
    (root / ".env").write_text(f"EEMLS_RUN_ID=r1\n{env_line}\n", encoding="utf-8")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(FAKE_SSH, encoding="utf-8")
    (bin_dir / "poetry").write_text(POETRY, encoding="utf-8")
    for tool in ("ssh", "poetry"):
        (bin_dir / tool).chmod(0o755)
    return subprocess.run(
        ["sh", "deploy/edge/login.sh", "user@device"],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_token_reaches_the_device_on_its_input_only(tmp_path: Path) -> None:
    done = logged_in(tmp_path, f"DOCKERHUB_READ_ONLY_TOKEN={TOKEN}")
    args = (tmp_path / "bin" / "args").read_text(encoding="utf-8")

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "bin" / "stdin").read_text(encoding="utf-8").strip() == TOKEN
    assert TOKEN not in args
    assert TOKEN not in done.stdout + done.stderr


def test_the_device_logs_in_as_the_owner_of_the_registry(tmp_path: Path) -> None:
    logged_in(tmp_path, f"DOCKERHUB_READ_ONLY_TOKEN={TOKEN}")
    args = (tmp_path / "bin" / "args").read_text(encoding="utf-8").splitlines()

    assert args[0] == "user@device"
    assert args[1] == "docker login -u 'someone' --password-stdin 'docker.io'"


def test_no_token_is_refused_before_the_device_is_reached(tmp_path: Path) -> None:
    done = logged_in(tmp_path, "DOCKERHUB_TOKEN=other")

    assert done.returncode == 1
    assert "no DOCKERHUB_READ_ONLY_TOKEN in .env" in done.stdout
    assert not (tmp_path / "bin" / "args").exists()
