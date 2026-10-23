from __future__ import annotations

import json
import os
import secrets
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.docker import docker_is_available

pytestmark = [pytest.mark.requires_docker, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
APPROACHES = ("cap", "cap_star")
RUN_IDS = ("recurring-ten-days-cap-s0", "recurring-ten-days-cap-star-s0")


def rows(folder: Path, table: str) -> list[dict[str, object]]:
    lines = (folder / "rows" / f"{table}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    if not docker_is_available():
        pytest.skip("no Docker daemon to run the system on")
    folder = tmp_path_factory.mktemp("experiment")
    done = subprocess.run(
        ["make", "experiment"],
        cwd=ROOT,
        env=os.environ
        | {
            "EEMLS_DATABASE_PASSWORD": secrets.token_hex(16),
            "PLAN": "experiments/runs/recurring_ten_days.yaml",
            "APPROACHES": ",".join(APPROACHES),
            "SEEDS": "0",
            "DURATION_S": "180",
            "WARM_IMAGES": "10",
            "OUT": str(folder),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=3600,
    )
    try:
        assert done.returncode == 0, done.stdout[-6000:]
        yield folder
    finally:
        subprocess.run(["make", "down"], cwd=ROOT, capture_output=True, check=False)


def test_every_approach_has_its_own_folder_manifest_and_rows(out: Path) -> None:
    for approach in APPROACHES:
        folder = out / "recurring_ten_days" / approach / "seed-0"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["approach"] == approach
        assert json.loads((folder / "rows" / "summary.json").read_text())["rows"]["queries"] > 0


def test_no_query_of_one_run_is_written_out_with_another(out: Path) -> None:
    (first, second) = (
        {
            row["query_id"]
            for row in rows(out / "recurring_ten_days" / approach / "seed-0", "queries")
        }
        for approach in APPROACHES
    )

    assert first
    assert second
    assert not first & second


def test_every_consumer_decided_on_cdm_values_in_every_run(out: Path) -> None:
    for approach in APPROACHES:
        decisions = rows(out / "recurring_ten_days" / approach / "seed-0", "decisions")

        assert len({row["consumer_id"] for row in decisions if row["cdm_values"]}) == 3


def test_the_two_approaches_score_different_contracts(out: Path) -> None:
    scored = [
        {
            name
            for row in rows(out / "recurring_ten_days" / approach / "seed-0", "decisions")
            for name in row["cdm_values"]  # type: ignore[attr-defined]
        }
        for approach in APPROACHES
    ]

    assert scored[0] != scored[1]


def test_no_run_volume_is_left_behind(out: Path) -> None:
    volumes = subprocess.run(
        ["docker", "volume", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert not {f"eemls_metrics-database-{run_id}" for run_id in RUN_IDS} & set(volumes)
