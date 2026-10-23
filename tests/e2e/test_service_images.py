from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.docker import a_free_port, docker_is_available

pytestmark = [pytest.mark.requires_docker, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "eemls.local"
READY_SECONDS = 60.0
CPU_SERVICE = "infer-b1-pedestrian-bt3"


@pytest.fixture(scope="module", autouse=True)
def built() -> None:
    if not docker_is_available():
        pytest.skip("no Docker daemon, so no image to build")
    done = subprocess.run(
        ["docker", "buildx", "bake", "--load", "local"],
        cwd=ROOT,
        env=os.environ | {"REGISTRY": REGISTRY},
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if done.returncode != 0:
        raise AssertionError(f"the bake file did not build:\n{done.stderr}")


def started(image: str, *arguments: str, port: int) -> Iterator[str]:
    name = f"eemls-image-test-{uuid.uuid4().hex[:8]}"
    host_port = a_free_port()
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{host_port}:{port}",
            *arguments,
            image,
        ],
        capture_output=True,
        check=True,
    )
    try:
        yield f"http://127.0.0.1:{host_port}"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def status_of(url: str) -> int:
    deadline = time.monotonic() + READY_SECONDS
    status = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as answer:
                return int(answer.status)
        except urllib.error.HTTPError as refused:
            status = int(refused.code)
        except (urllib.error.URLError, ConnectionError):
            status = 0
        time.sleep(0.5)
    return status


@pytest.fixture(scope="module")
def inference() -> Iterator[str]:
    sources = ROOT / "config" / "local" / "models.yaml"
    if not sources.is_file():
        pytest.skip("config/local/models.yaml is not here, so no model can be proved")
    yield from started(
        f"{REGISTRY}/eemls-inference-service:cpu-amd64",
        "-v",
        f"{ROOT / 'models'}:/opt/eemls/models:ro",
        "-v",
        f"{ROOT / 'config'}:/opt/eemls/config:ro",
        "-e",
        f"EEMLS_SERVICE_NAME={CPU_SERVICE}",
        "-e",
        "EEMLS_INSTANCE_ID=i-1",
        "-e",
        "EEMLS_NODE=local",
        port=8082,
    )


@pytest.fixture(scope="module")
def data_hub(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    store = tmp_path_factory.mktemp("hub")
    (store / "manifest.jsonl").write_text("", encoding="utf-8")
    store.chmod(0o755)
    (store / "manifest.jsonl").chmod(0o644)
    yield from started(
        f"{REGISTRY}/eemls-data-hub:latest",
        "-v",
        f"{store}:/var/lib/eemls/images:ro",
        "-v",
        f"{ROOT / 'config'}:/opt/eemls/config:ro",
        port=8081,
    )


def test_the_inference_service_is_alive(inference: str) -> None:
    assert status_of(f"{inference}/healthz") == 200


def test_the_inference_service_is_ready_after_its_warm_up(inference: str) -> None:
    assert status_of(f"{inference}/readyz") == 200


def test_the_data_hub_is_alive(data_hub: str) -> None:
    assert status_of(f"{data_hub}/healthz") == 200


@pytest.mark.parametrize(
    ("image", "program"),
    [
        ("eemls-control-plane:latest", "eemls-control-plane"),
        ("eemls-consumer:latest", "eemls-consumer"),
    ],
)
def test_the_image_runs_its_entry_point(image: str, program: str) -> None:
    done = subprocess.run(
        ["docker", "run", "--rm", f"{REGISTRY}/{image}", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0
    assert program in done.stdout
