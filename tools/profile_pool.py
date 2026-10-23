"""Measure every inference service of the pool on this machine; write start profiles.

Usage:
    python3 tools/profile_pool.py                 every service
    python3 tools/profile_pool.py --images 200    faster, for a smoke run
    python3 tools/profile_pool.py --at infer-a=http://127.0.0.1:18001   a running service
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from eemls_inference_service.model_loading import load

from eemls.domain.cost import DeviceRate, cost_per_1k, energy_wh_per_1k
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.statistics import percentile
from eemls_tools.config_tree import ConfigError, ConfigTree, load_config
from eemls_tools.datasets import ImageQuality, manifest_rows, pick_ids, trained_ids
from eemls_tools.profiles import PROFILES

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "data" / "hub"
MODELS = ROOT / "models"
INTO = ROOT / PROFILES

Row = dict[str, Any]


class ProfileError(Exception): ...


class Answer(NamedTuple):
    predicted_label: str
    label: str
    confidence: float
    total_ms: float


class Answers(NamedTuple):
    answers: list[Answer]
    node: str
    instance_id: str


def service_urls(given: Sequence[str]) -> dict[str, str]:
    held = {}
    for one in given:
        name, _, url = one.partition("=")
        if not url:
            raise ProfileError(f"{one}: give it as <service>=<url>")
        held[name] = url
    return held


def answered_by(
    url: str,
    images: Sequence[tuple[str, str]],
    *,
    transport: httpx.BaseTransport | None = None,
) -> Answers:
    answers = []
    node = instance_id = ""
    with httpx.Client(base_url=url, timeout=30.0, transport=transport) as client:
        for image_id, label in images:
            response = client.post("/predict", json={"image_id": image_id})
            if response.status_code != httpx.codes.OK:
                raise ProfileError(f"{url} answered {response.status_code}")
            body = response.json()
            answers.append(
                Answer(
                    predicted_label=body["predicted_label"],
                    label=label,
                    confidence=float(body["confidence"]),
                    total_ms=float(body["total_ms"]),
                )
            )
            node, instance_id = body["node"], body["instance_id"]
    return Answers(answers=answers, node=node, instance_id=instance_id)


def sampled(hub: Path, *, classes: Sequence[str], images: int, seed: int) -> list[Row]:
    held: dict[str, list[Row]] = {name: [] for name in classes}
    trained = trained_ids(hub)
    for row in manifest_rows(hub):
        if (
            row["image_quality"] == ImageQuality.clear
            and row["label"] in held
            and row["image_id"] not in trained
        ):
            held[row["label"]].append(row)
    queues = []
    for name in classes:
        by_id = {row["image_id"]: row for row in held[name]}
        queues.append([by_id[one] for one in pick_ids(list(by_id), count=len(by_id), seed=seed)])
    chosen: list[Row] = []
    depth = 0
    while len(chosen) < images and any(depth < len(one) for one in queues):
        chosen.extend(one[depth] for one in queues if depth < len(one))
        depth += 1
    return chosen[:images]


def measured(
    answers: Sequence[Answer],
    *,
    service_name: str,
    model: str,
    device: str,
    rate: DeviceRate,
    window_start: datetime,
    window_end: datetime,
    instance_id: str,
    node: str | None = None,
) -> InferenceServiceProfile:
    if not answers:
        raise ProfileError(f"{service_name} answered nothing, so nothing can be profiled")
    times = sorted(one.total_ms for one in answers)
    mean_ms = sum(times) / len(times)
    seconds = sum(times) / 1000.0
    return InferenceServiceProfile(
        model=model,
        device=device,
        node=node or device,
        instance_id=instance_id,
        service_name=service_name,
        window_start=window_start,
        window_end=window_end,
        mean_ms=mean_ms,
        p50_ms=percentile(times, 50.0),
        p95_ms=percentile(times, 95.0),
        max_ms=times[-1],
        throughput_rps=len(answers) / seconds if seconds else 0.0,
        n_queries=len(answers),
        energy_wh_per_1k=energy_wh_per_1k(rate, mean_response_time_ms=mean_ms),
        cost_per_1k=cost_per_1k(rate, mean_response_time_ms=mean_ms),
        general_accuracy=_accuracy(answers),
        accuracy_by_class=_accuracy_by_class(answers),
        confidence_by_class=_confidence_by_class(answers),
        source="measured",
    )


def _accuracy(answers: Sequence[Answer]) -> float:
    return sum(1 for one in answers if one.predicted_label == one.label) / len(answers)


def _accuracy_by_class(answers: Sequence[Answer]) -> dict[str, float]:
    return {name: _accuracy(mine) for name, mine in sorted(_per_class(answers).items())}


def _confidence_by_class(answers: Sequence[Answer]) -> dict[str, float]:
    return {
        name: sum(one.confidence for one in mine) / len(mine)
        for name, mine in sorted(_per_class(answers).items())
    }


def _per_class(answers: Sequence[Answer]) -> Mapping[str, list[Answer]]:
    held: dict[str, list[Answer]] = {}
    for one in answers:
        held.setdefault(one.label, []).append(one)
    return held


def _remote(
    service_name: str, url: str, rows: Sequence[Row], *, model: str, device: str, rate: DeviceRate
) -> InferenceServiceProfile:
    started = _now()
    remote = answered_by(url, [(row["image_id"], row["label"]) for row in rows])
    return measured(
        remote.answers,
        service_name=service_name,
        model=model,
        device=device,
        rate=rate,
        window_start=started,
        window_end=_now(),
        instance_id=remote.instance_id,
        node=remote.node,
    )


def _local(
    service_name: str,
    config: ConfigTree,
    images: Sequence[tuple[bytes, str]],
    *,
    rate: DeviceRate,
) -> InferenceServiceProfile:
    service = config.models.inference_services.services[service_name]
    backend = load(
        service.model,
        config.models.model_repository.models[service.model],
        under=MODELS,
        providers=("CPUExecutionProvider",),
        session_threads=1,
        warmup_requests=1,
    )
    started = _now()
    answers = []
    for image, label in images:
        at = time.perf_counter()
        answer = backend.predict(image)
        answers.append(
            Answer(
                predicted_label=answer.label,
                label=label,
                confidence=answer.confidence,
                total_ms=(time.perf_counter() - at) * 1000.0,
            )
        )
    backend.close()
    return measured(
        answers,
        service_name=service_name,
        model=service.model,
        device=service.device_type,
        rate=rate,
        window_start=started,
        window_end=_now(),
        instance_id=f"{service_name}-0",
    )


def _profiled(
    config: ConfigTree, rows: Sequence[Row], running: Mapping[str, str]
) -> list[InferenceServiceProfile]:
    rates = config.cost.rates()
    images = [((HUB / row["path"]).read_bytes(), row["label"]) for row in rows]
    held = []
    for service_name, service in sorted(config.models.inference_services.services.items()):
        rate = rates.get(service.device_type)
        if rate is None:
            raise ProfileError(f"{service.device_type} has no rate")
        if service_name in running:
            try:
                one = _remote(
                    service_name,
                    running[service_name],
                    rows,
                    model=service.model,
                    device=service.device_type,
                    rate=rate,
                )
            except (ProfileError, httpx.HTTPError) as refused:
                raise ProfileError(f"{service_name}: {refused}") from refused
        else:
            one = _local(service_name, config, images, rate=rate)
        held.append(one)
        _report(one)
    return held


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="profile_pool")
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--into", type=Path, default=INTO)
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--at", action="append", default=[], help="<service>=<url> of a copy")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        classes = config.datasets.classes
        running = service_urls(args.at)
        rows = sampled(HUB, classes=classes, images=args.images, seed=args.seed)
        if not rows:
            raise ProfileError(f"{HUB} holds no prepared image")
        print(f"{len(rows)} images over {len(classes)} classes")
        held = _profiled(config, rows, running)
    except (ConfigError, ProfileError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    args.into.parent.mkdir(parents=True, exist_ok=True)
    args.into.write_text(
        "".join(f"{json.dumps(one.model_dump(mode='json'), sort_keys=True)}\n" for one in held),
        encoding="utf-8",
    )
    print(f"{len(held)} profiles written to {args.into}")
    return 0


def _report(one: InferenceServiceProfile) -> None:
    print(
        f"{one.service_name:38s} {one.mean_ms:7.1f} ms  "
        f"{one.general_accuracy:.3f} accurate  {one.cost_per_1k:.4f} per 1k"
    )


def _now() -> datetime:
    return datetime.now(UTC)


if __name__ == "__main__":
    sys.exit(main())
