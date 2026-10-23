from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Iterator
from typing import TypedDict

import pytest

from eemls_tools.services import DATABASE_IMAGE
from tests.support.docker import a_free_port, docker_is_available

PASSWORD = "eemls-test"
READY_SECONDS = 30.0


class Writing(TypedDict):
    buffer_rows: int
    flush_interval_ms: float
    statement_timeout_ms: int
    max_connections: int


WRITING: Writing = {
    "buffer_rows": 200,
    "flush_interval_ms": 500.0,
    "statement_timeout_ms": 2000,
    "max_connections": 8,
}


def wait_until_ready(name: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        done = subprocess.run(
            ["docker", "exec", name, "pg_isready", "-U", "postgres", "-q"],
            capture_output=True,
            check=False,
        )
        if done.returncode == 0:
            return
        time.sleep(0.2)
    raise AssertionError(f"{name} did not become ready in {READY_SECONDS} s")


@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    if not docker_is_available():
        pytest.skip("no Docker daemon, so no PostgreSQL to test against")
    port = a_free_port()
    name = f"eemls-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--name",
            name,
            "--env",
            f"POSTGRES_PASSWORD={PASSWORD}",
            "--publish",
            f"127.0.0.1:{port}:5432",
            DATABASE_IMAGE,
        ],
        capture_output=True,
        check=True,
    )
    try:
        wait_until_ready(name, time.monotonic() + READY_SECONDS)
        yield f"postgresql://postgres:{PASSWORD}@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)


@pytest.fixture
def run_id() -> str:
    return f"run-{uuid.uuid4().hex[:8]}"
