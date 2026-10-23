from __future__ import annotations

import socket
import subprocess
import time
import uuid
from collections.abc import Iterator

import pytest

from eemls_tools.services import BROKER_IMAGE
from tests.support.docker import a_free_port, docker_is_available

READY_SECONDS = 30.0


def wait_until_listening(host: str, port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"{host}:{port} did not start listening in {READY_SECONDS} s")


@pytest.fixture(scope="session")
def broker() -> Iterator[tuple[str, int]]:
    if not docker_is_available():
        pytest.skip("no Docker daemon, so no broker to test against")
    port = a_free_port()
    name = f"eemls-broker-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--name",
            name,
            "--publish",
            f"127.0.0.1:{port}:1883",
            BROKER_IMAGE,
            "mosquitto",
            "-c",
            "/mosquitto-no-auth.conf",
        ],
        capture_output=True,
        check=True,
    )
    try:
        wait_until_listening("127.0.0.1", port, time.monotonic() + READY_SECONDS)
        yield "127.0.0.1", port
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
