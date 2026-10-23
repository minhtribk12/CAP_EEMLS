from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import load_config
from eemls_tools.deploy_command import main
from tests.support.docker import a_free_port, docker_is_available

pytestmark = [pytest.mark.requires_docker, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
K3S_IMAGE = "rancher/k3s:v1.33.5-k3s1"
READY_SECONDS = 120.0


@pytest.fixture(scope="module", autouse=True)
def docker_here() -> None:
    if not docker_is_available():
        pytest.skip("no Docker daemon to check the files against")


def written(root: Path, target: str, config: Path = ROOT / "config") -> Path:
    profiles = root / "profiles.jsonl"
    rows = []
    for name, service in load_config(config).models.inference_services.services.items():
        rows.append(
            json.dumps(
                {
                    "model": service.model,
                    "device": service.device_type,
                    "node": "n-1",
                    "instance_id": f"{name}-1",
                    "service_name": name,
                    "window_start": "2026-10-17T10:00:00Z",
                    "window_end": "2026-10-17T10:05:00Z",
                    "mean_ms": 10.0,
                    "p50_ms": 10.0,
                    "p95_ms": 12.0,
                    "max_ms": 15.0,
                    "throughput_rps": 100.0,
                    "n_queries": 500,
                    "energy_wh_per_1k": 1.0,
                    "cost_per_1k": 1.0,
                    "general_accuracy": 0.8,
                    "accuracy_by_class": {"car": 0.8},
                    "confidence_by_class": {"car": 0.8},
                    "source": "measured",
                }
            )
        )
    profiles.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = root / (target if target == "kube" else f"{target}.yaml")
    code = main(
        [
            target,
            "--config",
            str(config),
            "--profiles",
            str(profiles),
            "--images",
            "/srv/eemls/images",
            "--out",
            str(out),
            "--run-id",
            "r1",
            "--seed",
            "0",
            "--scenario",
            "smoke",
            "--scenarios",
            str(ROOT / "experiments" / "scenarios"),
            "--duration-s",
            "60",
        ]
    )
    assert code == 0
    return out


@pytest.fixture(scope="module")
def kubeconfig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is not installed")
    name = f"eemls-k3s-{uuid.uuid4().hex[:8]}"
    port = a_free_port()
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--privileged",
            "-p",
            f"127.0.0.1:{port}:6443",
            K3S_IMAGE,
            "server",
            "--disable=traefik,metrics-server",
        ],
        capture_output=True,
        check=True,
    )
    try:
        deadline = time.monotonic() + READY_SECONDS
        while time.monotonic() < deadline:
            ready = subprocess.run(
                ["docker", "exec", name, "kubectl", "get", "--raw", "/readyz"],
                capture_output=True,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(1.0)
        else:
            raise AssertionError(f"k3s was not ready in {READY_SECONDS} s")
        held = subprocess.run(
            ["docker", "exec", name, "cat", "/etc/rancher/k3s/k3s.yaml"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        path = tmp_path_factory.mktemp("k3s") / "kubeconfig"
        path.write_text(held.replace("127.0.0.1:6443", f"127.0.0.1:{port}"), encoding="utf-8")
        yield path
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def test_docker_swarm_accepts_the_stack(tmp_path: Path) -> None:
    done = subprocess.run(
        ["docker", "stack", "config", "-c", str(written(tmp_path, "swarm"))],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"EEMLS_DATABASE_PASSWORD": "checked-not-used"},
    )

    assert done.returncode == 0, done.stderr
    assert "observation-service" in yaml.safe_load(done.stdout)["services"]


def autoscaled_tree(root: Path, *, enabled: bool) -> Path:
    tree = committed_tree(root / "config")
    path = tree / "models" / "inference_services.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["autoscale"]["enabled"] = enabled
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return tree


def accepted(folder: Path, kubeconfig: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"KUBECONFIG": str(kubeconfig)}
    subprocess.run(
        ["kubectl", "apply", "-f", str(folder / "run.yaml")],
        capture_output=True,
        check=True,
        env=env,
    )
    return subprocess.run(
        ["kubectl", "apply", "--dry-run=server", "-k", str(folder)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_a_k3s_server_accepts_the_folder(tmp_path: Path, kubeconfig: Path) -> None:
    done = accepted(written(tmp_path, "kube", autoscaled_tree(tmp_path, enabled=False)), kubeconfig)

    assert done.returncode == 0, done.stderr
    assert done.stdout.count("(server dry run)") == done.stdout.count("\n")
    assert "horizontalpodautoscaler" not in done.stdout
    assert "deployment.apps/observation-service" in done.stdout
    assert "service/observation-service" in done.stdout


def test_a_k3s_server_accepts_the_folder_with_its_autoscalers(
    tmp_path: Path, kubeconfig: Path
) -> None:
    folder = written(tmp_path, "kube", autoscaled_tree(tmp_path, enabled=True))
    pool = load_config(ROOT / "config").models.inference_services.services

    done = accepted(folder, kubeconfig)

    assert done.returncode == 0, done.stderr
    assert done.stdout.count("(server dry run)") == done.stdout.count("\n")
    assert done.stdout.count("horizontalpodautoscaler.autoscaling/") == len(pool)
