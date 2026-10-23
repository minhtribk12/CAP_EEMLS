from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime

import httpx
import pytest

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.cost import DeviceRate
from eemls.domain.records import Feedback
from eemls.domain.types import AggregatedResult, Query, Row
from eemls.ensemble_optimization.approach import ConsumerSpecification
from eemls.ensemble_optimization.approaches.baselines.keep_the_ensemble import KeepTheEnsemble
from eemls.ensemble_optimization.decision_loop import DecisionLoop
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.testing import (
    FixedContext,
    conditioned,
    in_process_profile_updates,
)
from eemls.orchestration.ensemble_deployments import EnsembleDeployments
from eemls.orchestration.query_forwarding import QueryForwarding, member_connections
from eemls.orchestration.testing import PORT, Answers, Members, member
from eemls.testing.metrics import series
from eemls.testing.trees import TREE
from eemls_control_plane.config import load_control_plane_config
from eemls_control_plane.http_app import ControlPlaneEndpoints, build_app

CONFIG = load_control_plane_config(TREE)
CONSUMER = "c1_navigation"
TOPIC = "eemls/results/c1_navigation/client-1"
AT = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)

RATES: Mapping[str, DeviceRate] = {
    "raspberry_pi5": DeviceRate(
        power_watts=7.0,
        hardware_price=80.0,
        lifetime_years=5,
        energy_price_per_kwh=0.25,
        eur_per_cost_unit=0.00001,
    )
}

ANSWERING: Mapping[str, Answers] = {
    "infer-a": Answers(label="car", confidence=0.9),
    "infer-b": Answers(label="car", confidence=0.7),
    "infer-c": Answers(label="bus", confidence=0.6),
}


class Holding(httpx.AsyncBaseTransport):
    def __init__(self, inner: Members) -> None:
        self._inner = inner
        self.release = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self.release.wait()
        return await self._inner.handle_async_request(request)


class Published:
    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable
        self.sent: list[tuple[Query, AggregatedResult]] = []

    def __call__(self, asked: Query, result: AggregatedResult) -> None:
        if not self.reachable:
            raise OSError("the broker is not there")
        self.sent.append((asked, result))

    def topics(self) -> list[str]:
        return [one.reply_topic for one, _ in self.sent]


def consumers(
    served: Sequence[str] = (CONSUMER,), contract: str | None = None
) -> dict[str, ConsumerSpecification]:
    return {
        consumer_id: ConsumerSpecification(
            contract=CONFIG.consumers[consumer_id].scored_under(contract),
            cdms=CdmSet(definitions=CONFIG.consumers[consumer_id].cdms),
        )
        for consumer_id in served
    }


class Instance:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        serving: ControlPlaneEndpoints,
        query_forwarding: QueryForwarding,
        database: MemoryMetricsDatabase,
        broker: Published,
        monitoring: Monitoring,
        hold: Holding | None,
    ) -> None:
        self.client = client
        self.serving = serving
        self.query_forwarding = query_forwarding
        self.database = database
        self.broker = broker
        self.monitoring = monitoring
        self.hold = hold

    async def classify(self, **changes: object) -> httpx.Response:
        body: dict[str, object] = {
            "consumer_id": CONSUMER,
            "image_id": "img-7",
            "reply_topic": TOPIC,
            "correlation_id": "corr-1",
            "image_quality": "clear",
        }
        return await self.client.post("/classify", json=body | changes)

    async def answered(self) -> None:
        await self.serving.drain()


OPENED: list[Instance] = []


def built(
    *,
    deployments: EnsembleDeployments | None = None,
    database: MemoryMetricsDatabase | None = None,
    reachable: bool = True,
    cap: int = 64,
    decide_every: int = 5,
    holding: bool = False,
    ids: str = "q",
    instance_id: str = "control-plane-1",
    probe_fraction: float = 0.0,
) -> Instance:
    settings = CONFIG.services
    held = deployments or registered()
    store = database if database is not None else MemoryMetricsDatabase()
    broker = Published(reachable=reachable)
    monitoring = Monitoring.from_config(settings.monitoring)
    hold = Holding(Members(ANSWERING)) if holding else None
    query_forwarding = QueryForwarding(
        deployments=held,
        database=store,
        monitoring=monitoring,
        rates=RATES,
        runtime_context=FixedContext(),
        condition_of=conditioned(),
        publishing=broker,
        member_port=PORT,
        member_timeout_ms=1000.0,
        transport=hold if hold is not None else Members(ANSWERING),
        probe_fraction=probe_fraction,
        max_connections=member_connections(
            max_concurrent_queries_per_consumer=cap,
            consumers=len(held.consumers()),
            max_ensemble_size=3,
            probe_fraction=probe_fraction,
            available=len(held.available_inference_services),
        ),
    )
    serving = ControlPlaneEndpoints(
        instance_id=instance_id,
        query_forwarding=query_forwarding,
        deployments=held,
        database=store,
        orchestrator=DecisionLoop(
            deployments=held,
            database=store,
            policy=KeepTheEnsemble(),
            consumers=consumers(),
            monitoring=monitoring,
            runtime_context=FixedContext(),
            now=lambda: AT,
            policy_window=0,
            profiles=in_process_profile_updates(
                store,
                rates=RATES,
                window_s=120.0,
                interval_s=30.0,
                min_labeled_per_class=1,
                min_labeled_overall=1,
            ),
        ),
        monitoring=monitoring,
        max_concurrent_queries_per_consumer=cap,
        decision_interval_queries=decide_every,
        remember_served=100,
        now=lambda: AT,
        identifiers=_counting(ids),
    )
    app = build_app(serving, monitoring)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://control-plane"
    )
    instance = Instance(
        client=client,
        serving=serving,
        query_forwarding=query_forwarding,
        database=store,
        broker=broker,
        monitoring=monitoring,
        hold=hold,
    )
    OPENED.append(instance)
    return instance


def registered(serving: Sequence[str] | None = None) -> EnsembleDeployments:
    deployments = EnsembleDeployments(
        available_inference_services=tuple(member(name) for name in ANSWERING),
        max_ensemble_size=3,
    )
    deployments.register(consumer_id=CONSUMER, service_names=tuple(serving or ANSWERING))
    return deployments


def _counting(prefix: str) -> Callable[[], str]:
    held = {"n": 0}

    def next_id() -> str:
        held["n"] += 1
        return f"{prefix}-{held['n']}"

    return next_id


@pytest.fixture(autouse=True)
async def _let_the_instances_go() -> AsyncIterator[None]:
    yield
    while OPENED:
        held = OPENED.pop()
        if held.hold is not None:
            held.hold.release.set()
        await held.serving.drain()
        await held.query_forwarding.close()
        await held.client.aclose()


@pytest.fixture
def one() -> Iterator[Instance]:
    held = built()
    yield held


class TestStopping:
    async def test_the_answers_of_the_services_outside_the_ensemble_are_waited_for(
        self,
    ) -> None:
        held = built(deployments=registered(("infer-a", "infer-b")), probe_fraction=1.0)

        await held.classify()
        await held.serving.drain()

        assert ("q-1", "infer-c", False) in held.database.predictions


class TestAcceptingAQuery:
    async def test_a_query_is_accepted_at_once_with_the_id_it_was_given(
        self, one: Instance
    ) -> None:
        answer = await one.classify()

        assert answer.status_code == 202
        assert answer.json() == {"query_id": "q-1"}

    async def test_the_result_reaches_the_client_on_the_topic_the_request_named(
        self, one: Instance
    ) -> None:
        await one.classify()
        await one.answered()

        assert one.broker.topics() == [TOPIC]
        asked, result = one.broker.sent[0]
        assert (asked.correlation_id, result.predicted_label) == ("corr-1", "car")

    async def test_the_query_is_stored_with_the_time_it_arrived(self, one: Instance) -> None:
        await one.classify()
        await one.answered()

        stored = one.database.queries["q-1"]
        assert (stored.consumer_id, stored.t_arrive) == (CONSUMER, AT)

    async def test_the_query_is_stored_with_its_place_in_the_simulated_day(
        self, one: Instance
    ) -> None:
        await one.classify(time_of_day=0.25)
        await one.answered()

        assert one.database.queries["q-1"].time_of_day == 0.25

    async def test_a_query_sent_with_no_simulated_day_is_stored_without_one(
        self, one: Instance
    ) -> None:
        await one.classify()
        await one.answered()

        assert one.database.queries["q-1"].time_of_day is None

    async def test_a_place_in_the_day_outside_it_is_refused(self, one: Instance) -> None:
        answer = await one.classify(time_of_day=1.0)

        assert answer.status_code == 422

    async def test_a_consumer_this_instance_does_not_serve_is_not_accepted(
        self, one: Instance
    ) -> None:
        answer = await one.classify(consumer_id="c3_surveillance")

        assert answer.status_code == 404
        assert "c3_surveillance" in answer.text

    async def test_a_request_missing_what_a_query_needs_is_refused(self, one: Instance) -> None:
        answer = await one.client.post("/classify", json={"consumer_id": CONSUMER})

        assert answer.status_code == 422


class TestTheInFlightCap:
    async def test_a_query_over_the_cap_is_refused_at_once(self) -> None:
        instance = built(cap=1, holding=True)

        first = await instance.classify(correlation_id="corr-1")
        second = await instance.classify(correlation_id="corr-2")

        assert (first.status_code, second.status_code) == (202, 503)

    async def test_a_refusal_is_counted_and_never_queued(self) -> None:
        instance = built(cap=1, holding=True)

        await instance.classify(correlation_id="corr-1")
        await instance.classify(correlation_id="corr-2")

        body = instance.monitoring.exposition()[0]
        assert series(body, "eemls_queries_total", consumer=CONSUMER, outcome="refused") == 1.0
        assert instance.broker.sent == []

    async def test_an_answered_query_gives_its_place_back(self, one: Instance) -> None:
        for index in range(3):
            answer = await one.classify(correlation_id=f"corr-{index}")
            assert answer.status_code == 202
            await one.answered()

        assert len(one.broker.sent) == 3


class TestAnyInstanceCanAnswer:
    async def test_a_second_instance_answers_a_request_the_first_never_saw(self) -> None:
        deployments, database = registered(), MemoryMetricsDatabase()
        first = built(deployments=deployments, database=database, ids="first", instance_id="cp-1")
        second = built(deployments=deployments, database=database, ids="second", instance_id="cp-2")

        await first.classify(correlation_id="corr-1")
        await second.classify(correlation_id="corr-2")
        await first.answered()
        await second.answered()

        assert first.broker.topics() == [TOPIC]
        assert second.broker.topics() == [TOPIC]
        assert sorted(database.queries) == ["first-1", "second-1"]


class TestWhenTheBrokerIsNotThere:
    async def test_a_dead_broker_still_leaves_the_row_in_the_store(self) -> None:
        instance = built(reachable=False)

        answer = await instance.classify()
        await instance.answered()

        assert answer.status_code == 202
        stored = instance.database.queries["q-1"]
        assert (stored.predicted_label, stored.t_publish) == ("car", None)

    async def test_the_next_query_is_still_accepted(self) -> None:
        instance = built(reachable=False)

        await instance.classify(correlation_id="corr-1")
        await instance.answered()
        answer = await instance.classify(correlation_id="corr-2")
        await instance.answered()

        assert answer.status_code == 202
        assert sorted(instance.database.queries) == ["q-1", "q-2"]


class TestReturnedLabels:
    async def test_a_label_of_a_query_it_served_is_kept(self, one: Instance) -> None:
        await one.classify()
        await one.answered()

        answer = await one.client.post(
            "/feedback",
            json={
                "consumer_id": CONSUMER,
                "labels": [{"query_id": "q-1", "true_label": "car"}],
            },
        )

        assert answer.status_code == 200
        assert answer.json() == {"kept": 1, "ignored": 0}
        assert one.database.feedback["q-1"].true_label == "car"

    async def test_a_label_of_a_query_it_never_served_is_ignored_and_counted(
        self, one: Instance
    ) -> None:
        answer = await one.client.post(
            "/feedback",
            json={
                "consumer_id": CONSUMER,
                "labels": [{"query_id": "q-never", "true_label": "car"}],
            },
        )

        assert answer.json() == {"kept": 0, "ignored": 1}
        assert one.database.feedback == {}

    async def test_a_label_for_a_consumer_this_instance_does_not_serve_is_not_taken(
        self, one: Instance
    ) -> None:
        answer = await one.client.post(
            "/feedback",
            json={
                "consumer_id": "c3_surveillance",
                "labels": [{"query_id": "q-1", "true_label": "car"}],
            },
        )

        assert answer.status_code == 404


class SlowFeedback(MemoryMetricsDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.released_while_writing = False

    def write_feedback(self, feedback: Feedback) -> None:
        self.started.set()
        self.released_while_writing = self.release.wait(2.0)
        super().write_feedback(feedback)


class TestASlowLabelHandOff:
    async def test_a_query_is_still_accepted_while_a_label_is_being_written(self) -> None:
        store = SlowFeedback()
        instance = built(database=store)
        await instance.classify()
        await instance.answered()
        handing = asyncio.create_task(
            instance.client.post(
                "/feedback",
                json={
                    "consumer_id": CONSUMER,
                    "labels": [{"query_id": "q-1", "true_label": "car"}],
                },
            )
        )
        assert await asyncio.to_thread(store.started.wait, 5.0)

        answer = await instance.classify(correlation_id="corr-2")
        store.release.set()
        handed = await handing

        assert answer.status_code == 202
        assert store.released_while_writing
        assert handed.json() == {"kept": 1, "ignored": 0}


class TestWhatARunReadsBack:
    async def test_the_ensemble_serving_a_consumer_is_answered(self, one: Instance) -> None:
        answer = await one.client.get(f"/ensembles/{CONSUMER}")

        assert answer.status_code == 200
        body = answer.json()
        assert body["deployment_id"] == f"{CONSUMER}-0001"
        assert [held["service_name"] for held in body["members"]] == [
            "infer-a",
            "infer-b",
            "infer-c",
        ]

    async def test_a_consumer_nothing_serves_is_a_404(self, one: Instance) -> None:
        answer = await one.client.get("/ensembles/c9_nobody")

        assert answer.status_code == 404

    async def test_the_decisions_taken_are_answered_newest_first(self, one: Instance) -> None:
        one.serving.decide(CONSUMER)
        one.serving.decide(CONSUMER)

        answer = await one.client.get("/decisions", params={"limit": 2})

        assert answer.status_code == 200
        held = answer.json()["decisions"]
        assert len(held) == 2
        assert {one["consumer_id"] for one in held} == {CONSUMER}


class TestDecidingWhileItServes:
    async def test_a_decision_is_taken_every_so_many_queries_of_that_consumer(self) -> None:
        instance = built(decide_every=2)

        for index in range(4):
            await instance.classify(correlation_id=f"corr-{index}")
            await instance.answered()

        assert len(instance.database.decisions) == 2

    async def test_no_decision_before_the_interval_is_reached(self) -> None:
        instance = built(decide_every=5)

        for index in range(4):
            await instance.classify(correlation_id=f"corr-{index}")
            await instance.answered()

        assert instance.database.decisions == {}


class TestADecisionStillBeingTaken:
    async def test_an_interval_due_while_its_consumer_decides_is_skipped_and_counted(
        self,
    ) -> None:
        instance = built(decide_every=1)
        started, release = threading.Event(), threading.Event()
        taken: list[str] = []

        def slow(consumer_id: str) -> None:
            taken.append(consumer_id)
            started.set()
            release.wait(30.0)

        instance.serving.decide = slow  # type: ignore[method-assign]
        first = asyncio.create_task(instance.serving.decide_when_free(CONSUMER))
        assert await asyncio.to_thread(started.wait, 30.0)

        await instance.serving.decide_when_free(CONSUMER)
        release.set()
        await first

        assert taken == [CONSUMER]
        assert skipped(instance) == 1.0

    async def test_the_next_interval_decides_again_once_the_last_decision_is_done(
        self,
    ) -> None:
        instance = built(decide_every=1)

        for index in range(3):
            await instance.classify(correlation_id=f"corr-{index}")
            await instance.answered()

        assert len(instance.database.decisions) == 3


class RefusesWindows(MemoryMetricsDatabase):
    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        raise OSError(f"the store cannot answer for {consumer_id}")


class TestADecisionThatFails:
    async def test_the_failure_is_counted_and_the_run_keeps_serving(self) -> None:
        instance = built(database=RefusesWindows())

        instance.serving.decide(CONSUMER)

        body, _ = instance.monitoring.exposition()
        assert series(body, "eemls_decision_failures_total", consumer=CONSUMER) == 1.0
        assert (await instance.classify()).status_code == 202


def skipped(instance: Instance) -> float | None:
    body, _ = instance.monitoring.exposition()
    return series(body, "eemls_decisions_skipped_total", consumer=CONSUMER)


class TestHealthAndMetrics:
    async def test_a_built_instance_is_up_and_ready(self, one: Instance) -> None:
        assert (await one.client.get("/healthz")).status_code == 200
        assert (await one.client.get("/readyz")).status_code == 200

    async def test_the_metrics_path_carries_what_this_instance_served(self, one: Instance) -> None:
        await one.classify()
        await one.answered()

        answer = await one.client.get(CONFIG.services.monitoring.metrics_path)

        assert answer.status_code == 200
        assert (
            series(answer.text, "eemls_queries_total", consumer=CONSUMER, outcome="accepted") == 1.0
        )
