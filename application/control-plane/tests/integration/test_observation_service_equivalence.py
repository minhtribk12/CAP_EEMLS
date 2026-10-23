from __future__ import annotations

import itertools
import json
import time
import uuid
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from eemls.domain.records import Decision
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.metrics_reporting import ReportingDatabase
from eemls.observation_service.profile_updates import ProfileUpdatesClient, ProfileUpdating
from eemls.observation_service.testing import FixedContext, InProcessObservationService, conditioned
from eemls.testing.records import AT, profile
from eemls.testing.trees import committed_tree
from eemls_control_plane.config import ControlPlaneView, load_control_plane_config
from eemls_control_plane.http_app import ControlPlaneEndpoints, build_app
from eemls_control_plane.main import metrics_reporter_of, profile_updates_of
from eemls_control_plane.startup import build_control_plane

RUN = "run-equivalence"
SEED = 11
ROUNDS = 80
TABLES = ("queries", "member_predictions", "feedback", "decisions", "inference_service_profiles")


@pytest.fixture
def config(tmp_path: Path) -> ControlPlaneView:
    tree = committed_tree(tmp_path / "config")
    service = tree / "services" / "observation_service.yaml"
    body = yaml.safe_load(service.read_text(encoding="utf-8"))
    body |= {"profile_min_labeled_per_class": 1, "profile_min_labeled_overall": 2}
    service.write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_control_plane_config(tree)


def picked(key: str, choices: tuple[str, ...]) -> str:
    return choices[zlib.crc32(key.encode()) % len(choices)]


def truth(config: ControlPlaneView, image_id: str) -> str:
    return picked(image_id, config.datasets.dominant_classes)


def members(config: ControlPlaneView) -> httpx.MockTransport:
    classes = config.datasets.dominant_classes

    def answered(request: httpx.Request) -> httpx.Response:
        name = request.url.host
        asked = json.loads(request.content)
        image_id = str(asked["image_id"])
        right = zlib.crc32(f"{name}/{image_id}".encode()) % 3 != 0
        service = config.models.inference_services.services[name]
        return httpx.Response(
            200,
            json={
                "query_id": asked["query_id"],
                "instance_id": f"{name}-1",
                "service_name": name,
                "model": service.model,
                "device": service.device_type,
                "node": "node-1",
                "replica": 1,
                "predicted_label": truth(config, image_id)
                if right
                else picked(f"{name}/{image_id}", classes),
                "confidence": 0.5 + (zlib.crc32(f"{image_id}/{name}".encode()) % 50) / 100.0,
                "fetch_ms": 1.0,
                "inference_ms": 5.0 + len(name) % 7,
                "total_ms": 8.0 + len(name) % 7,
                "execution_provider": "CPUExecutionProvider",
                "number_format": "float32",
                "cache_hit": True,
                "status": "ok",
            },
        )

    return httpx.MockTransport(answered)


def profiled(config: ControlPlaneView) -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase(RUN)
    services = sorted(config.models.inference_services.services.items())
    for index, (name, service) in enumerate(services):
        database.write_profile(
            profile(
                service_name=name,
                model=service.model,
                device=service.device_type,
                instance_id=f"i-{index}",
                general_accuracy=0.5 + index / (2 * len(services)),
                window_start=AT - timedelta(minutes=10),
                window_end=AT - timedelta(minutes=5),
            )
        )
    return database


def ticking() -> Callable[[], datetime]:
    moments = (AT + timedelta(milliseconds=500 * step) for step in itertools.count())
    return lambda: next(moments)


@dataclass
class Wiring:
    serving: ControlPlaneEndpoints
    stored: MemoryMetricsDatabase
    closing: Callable[[], None]


def observation_service(
    config: ControlPlaneView, stored: MemoryMetricsDatabase
) -> InProcessObservationService:
    settings = config.services.observation_service
    return InProcessObservationService(
        stored,
        run_id=RUN,
        rates=config.cost.rates(),
        window_s=settings.profile_window_s,
        min_labeled_per_class=settings.profile_min_labeled_per_class,
        min_labeled_overall=settings.profile_min_labeled_overall,
    )


def direct(config: ControlPlaneView, monitoring: Monitoring) -> Wiring:
    stored = profiled(config)
    profiles = ProfileUpdatesClient(
        database=stored,
        base_url=str(config.services.observation_service.url),
        interval_s=config.services.control_plane.profile_update_interval_s,
        request_timeout_ms=float(config.services.observation_service.request_timeout_ms),
        transport=observation_service(config, stored),
    )
    serving = served(config, stored, monitoring, profiles=profiles)

    def closing() -> None:
        profiles.close()
        stored.flush()

    return Wiring(serving, stored, closing)


def observed(config: ControlPlaneView, monitoring: Monitoring) -> Wiring:
    stored = profiled(config)
    service = observation_service(config, stored)
    database = ReportingDatabase(
        stored,
        metrics_reporter_of(
            config,
            instance_id="control-plane-1",
            run_id=RUN,
            monitoring=monitoring,
            transport=service,
        ),
    )
    profiles = profile_updates_of(config, database, transport=service)
    serving = served(config, database, monitoring, profiles=profiles)

    def closing() -> None:
        profiles.close()
        database.close()

    return Wiring(serving, stored, closing)


def served(
    config: ControlPlaneView,
    database: MemoryMetricsDatabase | ReportingDatabase,
    monitoring: Monitoring,
    profiles: ProfileUpdating,
) -> ControlPlaneEndpoints:
    return build_control_plane(
        config,
        database=database,
        instance_id="control-plane-1",
        monitoring=monitoring,
        publishing=lambda query, result: None,
        runtime_context=FixedContext(),
        condition_of=conditioned(),
        now=ticking(),
        seed=SEED,
        transport=members(config),
        profiles=profiles,
    )


def counted() -> Iterator[uuid.UUID]:
    return (uuid.UUID(int=index) for index in itertools.count(1))


async def run(
    config: ControlPlaneView, held: Wiring, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, list[dict[str, object]]], list[Decision]]:
    ids = counted()
    monkeypatch.setattr(uuid, "uuid4", lambda: next(ids))
    monitoring = Monitoring.from_config(config.services.monitoring)
    app = build_app(held.serving, monitoring)
    consumers = sorted(config.consumers)
    waiting: list[tuple[str, str, str]] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://control-plane"
    ) as client:
        for index in range(ROUNDS):
            for consumer_id in consumers:
                image_id = f"img-{consumer_id}-{index}"
                answer = await client.post(
                    "/classify",
                    json={
                        "consumer_id": consumer_id,
                        "image_id": image_id,
                        "reply_topic": f"eemls/results/{consumer_id}/client-1",
                        "correlation_id": f"corr-{index}",
                    },
                )
                assert answer.status_code == 202
                await held.serving.drain()
                if index % 2 == 0:
                    waiting.append((consumer_id, answer.json()["query_id"], image_id))
            if index % 4 == 3:
                for consumer_id in consumers:
                    labels = [
                        {"query_id": query_id, "true_label": truth(config, image_id)}
                        for one, query_id, image_id in waiting
                        if one == consumer_id
                    ]
                    placed = await client.post(
                        "/feedback", json={"consumer_id": consumer_id, "labels": labels}
                    )
                    assert placed.status_code == 200
                waiting.clear()
    held.closing()
    rows = {table: [dict(row) for row in held.stored.read_rows(table)] for table in TABLES}
    decisions = sorted(held.stored.decisions.values(), key=lambda one: one.decision_id)
    return rows, decisions


def sequence(decisions: list[Decision]) -> list[tuple[object, ...]]:
    return [
        (
            one.decision_id,
            one.consumer_id,
            one.t,
            one.action,
            one.accepted,
            one.deployment_before,
            one.deployment_after,
            one.score,
            one.cdm_values,
        )
        for one in decisions
    ]


async def test_the_observation_service_stores_what_the_direct_writes_stored(
    config: ControlPlaneView, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)
    monitoring = Monitoring.from_config(config.services.monitoring)

    direct_rows, direct_decisions = await run(config, direct(config, monitoring), monkeypatch)
    observed_rows, observed_decisions = await run(config, observed(config, monitoring), monkeypatch)

    assert len(direct_decisions) >= len(config.consumers) * ROUNDS // 5
    assert {one.action.kind for one in direct_decisions} - {"keep"}
    assert len(direct_rows["inference_service_profiles"]) > len(
        config.models.inference_services.services
    )
    assert direct_rows["feedback"]
    assert sequence(observed_decisions) == sequence(direct_decisions)
    for table in TABLES:
        assert observed_rows[table] == direct_rows[table], table
