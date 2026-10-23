from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def forget_the_run(run_id: str) -> None:
    subprocess.run(
        ["docker", "volume", "rm", f"eemls_metrics-database-{run_id}"],
        capture_output=True,
        check=False,
    )


def make(target: str, environment: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", target],
        cwd=ROOT,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
