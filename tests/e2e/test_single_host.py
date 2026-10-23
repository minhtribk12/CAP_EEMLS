from __future__ import annotations

import os
import secrets
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.docker import docker_is_available
from tests.support.one_host import forget_the_run, make

pytestmark = [pytest.mark.requires_docker, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "one-host-check"
STARTING_MEMBERS = (
    "infer-b1-general-pi4",
    "infer-b1-pedestrian-bt3",
    "infer-b2-cyclist-bt3",
)
ENVIRONMENT = os.environ | {
    "EEMLS_DATABASE_PASSWORD": secrets.token_hex(16),
    "RUN_ID": RUN_ID,
    "SCENARIO": "recurring",
    "DURATION_S": "180",
}


def stored(sql: str) -> str:
    container = subprocess.run(
        ["docker", "ps", "--filter", "name=eemls_postgres", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]
    return subprocess.run(
        ["docker", "exec", container, "psql", "-U", "eemls", "-d", "eemls", "-t", "-A", "-c", sql],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture(scope="module", autouse=True)
def run() -> Iterator[None]:
    if not docker_is_available():
        pytest.skip("no Docker daemon to run the system on")
    forget_the_run(RUN_ID)
    started = make("up", ENVIRONMENT)
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        ran = make("run", ENVIRONMENT)
        assert ran.returncode == 0, ran.stdout + ran.stderr
        yield
    finally:
        make("down", ENVIRONMENT)
        forget_the_run(RUN_ID)


def test_the_run_stored_the_members_answers() -> None:
    answered = stored(
        f"select count(*) from member_predictions where run_id = '{RUN_ID}' and status = 'ok'"
    )

    assert int(answered) > 0


def test_no_member_call_failed_to_reach_its_service() -> None:
    unreached = stored(
        f"select count(*) from member_predictions where run_id = '{RUN_ID}' and status = 'error'"
    )

    assert int(unreached) == 0


def test_every_member_the_consumers_start_on_answered() -> None:
    answering = stored(
        "select count(distinct service_name) from member_predictions "
        f"where run_id = '{RUN_ID}' and status = 'ok'"
    )

    assert int(answering) >= len(STARTING_MEMBERS)


def test_almost_every_query_was_answered() -> None:
    unanswered = stored(
        f"select count(*) filter (where outcome = 'failed') * 100 / count(*) from queries "
        f"where run_id = '{RUN_ID}'"
    )

    assert int(unanswered) <= 1


def test_the_run_stored_at_least_one_decision_for_every_consumer() -> None:
    consumers = stored(
        f"select count(distinct consumer_id) from decisions where run_id = '{RUN_ID}'"
    )

    assert int(consumers) == 3


def test_every_consumer_decided_on_cdm_values_once_its_labels_returned() -> None:
    deciding = stored(
        "select count(distinct consumer_id) from decisions "
        f"where run_id = '{RUN_ID}' and cdm_values <> '{{}}'::jsonb"
    )

    assert int(deciding) == 3
