from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

from eemls.domain.cost import DeviceRate
from eemls.domain.measuring import NoMeasuring
from eemls.domain.types import AggregatedResult, Query
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.observation_service.testing import conditioned
from eemls.orchestration.aggregation import AggregationError
from eemls.orchestration.ensemble_deployments import EnsembleDeployments
from eemls.orchestration.query_forwarding import QueryForwarding
from eemls.orchestration.testing import HOST, member
from eemls.testing.records import context, query

RATES = {
    "raspberry_pi5": DeviceRate(
        power_watts=7.0,
        hardware_price=80.0,
        lifetime_years=5,
        energy_price_per_kwh=0.25,
        eur_per_cost_unit=0.00001,
    )
}


def listening(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex((HOST, port)) == 0


@pytest.fixture
def port() -> Iterator[int]:
    with socket.socket() as held:
        held.bind((HOST, 0))
        free = int(held.getsockname()[1])
    server = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "eemls.orchestration.testing:sometimes_slow",
            "--host",
            HOST,
            "--port",
            str(free),
            "--log-level",
            "warning",
        ]
    )
    while not listening(free):
        time.sleep(0.05)
    yield free
    server.terminate()
    server.wait(timeout=10)


def query_forwarding_on(port: int) -> QueryForwarding:
    deployments = EnsembleDeployments(
        available_inference_services=(member(HOST),), max_ensemble_size=3
    )
    deployments.register(consumer_id="c1_navigation", service_names=(HOST,))
    return QueryForwarding(
        deployments=deployments,
        database=MemoryMetricsDatabase(),
        monitoring=NoMeasuring(),
        rates=RATES,
        runtime_context=lambda _: context(),
        condition_of=conditioned("day"),
        publishing=published,
        member_port=port,
        member_timeout_ms=300.0,
        probe_fraction=0.0,
        max_connections=200,
    )


def published(asked: Query, result: AggregatedResult) -> None: ...


def open_files() -> int:
    return len(os.listdir("/proc/self/fd"))


async def lagging() -> None:
    while True:
        time.sleep(0.2)
        await asyncio.sleep(0.001)


async def asked(query_forwarding: QueryForwarding, number: int) -> None:
    with contextlib.suppress(AggregationError):
        await query_forwarding.answer(
            query(query_id=f"q-{number}", consumer_id="c1_navigation"),
            arrived_at=time.perf_counter(),
        )


class TestAMemberCallThatRunsOutOfTime:
    @pytest.mark.filterwarnings("ignore::ResourceWarning")
    async def test_it_leaves_no_connection_open_when_the_loop_lags(self, port: int) -> None:
        query_forwarding = query_forwarding_on(port)
        started = open_files()
        for wave in range(10):
            lag = asyncio.create_task(lagging())
            await asyncio.gather(*(asked(query_forwarding, wave * 60 + one) for one in range(60)))
            lag.cancel()
        await asyncio.sleep(2.5)
        gc.collect()
        await asyncio.sleep(0.5)
        pool = query_forwarding._client._transport._pool  # type: ignore[attr-defined]

        stuck = [one for one in pool.connections if not one.is_idle() and not one.is_closed()]
        idle = [one for one in pool.connections if one.is_idle()]

        files = open_files()
        await query_forwarding.close()

        assert stuck == []
        assert files <= started + len(idle) + 5
