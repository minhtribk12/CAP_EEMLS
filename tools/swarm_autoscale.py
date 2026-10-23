"""Scale the inference services of a Swarm stack by CPU use.

Usage:
    python3 tools/swarm_autoscale.py --config config --stack eemls
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

from eemls.config.models import Autoscale, InferenceServiceEntry
from eemls_tools.config_tree import load_config

ROOT = Path(__file__).resolve().parents[1]

STACK = "eemls"
INTERVAL_S = 15.0


@dataclass(frozen=True)
class Scaling:
    replicas: int = 0
    low_since: float | None = None


def scaled(
    current: int,
    use: Sequence[float],
    *,
    minimum: int,
    settings: Autoscale,
    previous: Scaling,
    now: float,
) -> Scaling:
    """Copies one service should run, from each copy's use in %."""
    mean = sum(use) / len(use)
    wanted = math.ceil(current * mean / settings.cpu_percent)
    wanted = min(max(wanted, minimum), max(settings.max_replicas, minimum))
    if wanted >= current:
        return Scaling(replicas=wanted)
    low_since = previous.low_since if previous.low_since is not None else now
    if now - low_since >= settings.cooldown_s:
        return Scaling(replicas=wanted)
    return Scaling(replicas=current, low_since=low_since)


def cpu_use_of(
    stats: str, stack: str, services: Set[str], cpu_cores: float
) -> dict[str, list[float]]:
    """Parse `docker stats` into each service's copy use, in %."""
    use: dict[str, list[float]] = {}
    for line in stats.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row["Name"].removeprefix(f"{stack}_")
        service = task.split(".")[0]
        if task == row["Name"] or service not in services:
            continue
        percent = float(row["CPUPerc"].rstrip("%"))
        use.setdefault(service, []).append(percent / cpu_cores)
    return use


def replicas_of(listing: str, stack: str) -> dict[str, int]:
    replicas: dict[str, int] = {}
    for line in listing.splitlines():
        name, counts = line.split()[:2]
        if name.startswith(f"{stack}_"):
            replicas[name.removeprefix(f"{stack}_")] = int(counts.split("/")[1])
    return replicas


def docker(*arguments: str) -> str:
    return subprocess.run(["docker", *arguments], check=True, capture_output=True, text=True).stdout


def watch(
    services: Mapping[str, InferenceServiceEntry],
    settings: Autoscale,
    cpu_cores: float,
    stack: str,
    interval_s: float,
) -> None:
    scalings = {name: Scaling() for name in services}
    while True:
        listing = docker(
            "service", "ls", "--filter", f"label=com.docker.stack.namespace={stack}",
            "--format", "{{.Name}} {{.Replicas}}",
        )  # fmt: skip
        replicas = replicas_of(listing, stack)
        stats = docker("stats", "--no-stream", "--format", "{{json .}}")
        use = cpu_use_of(stats, stack, set(services), cpu_cores)
        for name in sorted(set(use) & set(replicas)):
            scalings[name] = scaled(
                replicas[name],
                use[name],
                minimum=services[name].replicas,
                settings=settings,
                previous=scalings[name],
                now=time.monotonic(),
            )
            if scalings[name].replicas != replicas[name]:
                docker("service", "scale", "--detach", f"{stack}_{name}={scalings[name].replicas}")
                print(f"{name}: {replicas[name]} -> {scalings[name].replicas} copies", flush=True)
        time.sleep(interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swarm_autoscale")
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--stack", default=STACK)
    parser.add_argument("--interval-s", type=float, default=INTERVAL_S)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    inference = config.models.inference_services
    if not inference.autoscale.enabled:
        print(
            "refused: autoscale.enabled is false in models/inference_services.yaml",
            file=sys.stderr,
        )
        return 1
    cpu_cores = config.services.deployment.inference_cpu_cores
    watch(inference.services, inference.autoscale, cpu_cores, args.stack, args.interval_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
