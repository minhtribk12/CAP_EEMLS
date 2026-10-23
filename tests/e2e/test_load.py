from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.docker import docker_is_available
from tests.support.one_host import forget_the_run, make

pytestmark = [pytest.mark.requires_docker, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "load-check"
REPORT = ROOT / "runs" / "one_host" / "load.json"
ENVIRONMENT = os.environ | {
    "EEMLS_DATABASE_PASSWORD": secrets.token_hex(16),
    "RUN_ID": RUN_ID,
    "RATES": "5,10",
    "STEP_S": "20",
    "CONSUMER": "c3_surveillance",
}


@pytest.fixture(scope="module")
def reports() -> Iterator[list[dict[str, object]]]:
    if not docker_is_available():
        pytest.skip("no Docker daemon to run the system on")
    forget_the_run(RUN_ID)
    started = make("up", ENVIRONMENT)
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        loaded = make("load", ENVIRONMENT)
        assert loaded.returncode == 0, loaded.stdout + loaded.stderr
        yield json.loads(REPORT.read_text(encoding="utf-8"))
    finally:
        make("down", ENVIRONMENT)
        forget_the_run(RUN_ID)


def test_there_is_one_report_per_rate(reports: list[dict[str, object]]) -> None:
    assert [one["rate_per_s"] for one in reports] == [5.0, 10.0]


def test_every_query_of_a_rate_the_system_carries_is_sent_and_accepted(
    reports: list[dict[str, object]],
) -> None:
    for one in reports:
        assert one["not_sent"] == 0
        assert one["accepted"] == int(float(str(one["rate_per_s"])) * 20)


def test_each_rate_reports_its_response_time_and_where_it_went(
    reports: list[dict[str, object]],
) -> None:
    for one in reports:
        assert one["answered"] > 0  # type: ignore[operator]
        for field in ("p50_ms", "p95_ms", "p99_ms", "fetch_ms", "inference_ms", "wait_ms"):
            assert one[field] is not None, field
