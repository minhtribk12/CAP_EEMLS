from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

import pytest

from eemls.config.reading import Reader
from eemls.config.services import MonitoringConfig
from eemls.domain.cost import DeviceRate
from eemls.domain.types import (
    AggregatedResult,
    Condition,
    MemberPrediction,
    Query,
    RuntimeContext,
)
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.operating_conditions import Conditioning
from eemls.orchestration.aggregation import AggregationError
from eemls.orchestration.ensemble_deployments import EnsembleDeployments, NoSuchConsumerError
from eemls.orchestration.query_forwarding import Contexting, QueryForwarding, member_connections
from eemls.orchestration.testing import PORT, Answers, Members, member
from eemls.testing.records import context, query
from eemls.testing.trees import TREE

RATES: Mapping[str, DeviceRate] = {
    "raspberry_pi5": DeviceRate(
        power_watts=7.0,
        hardware_price=80.0,
        lifetime_years=5,
        energy_price_per_kwh=0.25,
        eur_per_cost_unit=0.00001,
    )
}


class Published:
    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable
        self.sent: list[tuple[Query, AggregatedResult]] = []

    def __call__(self, asked: Query, result: AggregatedResult) -> None:
        if not self.reachable:
            raise OSError("the broker is not there")
        self.sent.append((asked, result))


class Built:
    def __init__(
        self,
        query_forwarding: QueryForwarding,
        database: MemoryMetricsDatabase,
        members: Members,
        broker: Published,
        monitoring: Monitoring,
    ) -> None:
        self.query_forwarding = query_forwarding
        self.database = database
        self.members = members
        self.broker = broker
        self.monitoring = monitoring

    def rows(self, query_id: str = "q-1") -> Sequence[MemberPrediction]:
        return [one for held, one in self.database.predictions.items() if held[0] == query_id]

    def row(self, service_name: str, query_id: str = "q-1") -> MemberPrediction:
        return self.database.predictions[(query_id, service_name, True)]


def conditioned(condition: Condition) -> Conditioning:
    async def of(image_id: str) -> Condition:
        return condition

    return of


class TestTheConditionOfTheImageAQueryIsAbout:
    async def test_the_stored_query_carries_the_condition_of_its_image(self) -> None:
        held = built({"infer-a": Answers(label="car")}, condition="night")

        await held.query_forwarding.answer(
            query(consumer_id="c1_navigation"), arrived_at=time.perf_counter()
        )

        assert held.database.queries["q-1"].context.condition == "night"

    async def test_the_rest_of_the_context_is_the_one_the_control_plane_holds(self) -> None:
        held = built({"infer-a": Answers(label="car")}, condition="night")

        await held.query_forwarding.answer(
            query(consumer_id="c1_navigation"), arrived_at=time.perf_counter()
        )

        stored = held.database.queries["q-1"].context
        assert (stored.period_name, stored.period_index) == (context().period_name, 3)
        assert stored.weather == context().weather

    async def test_the_context_is_the_one_held_for_the_consumer_of_the_query(self) -> None:
        asked: list[str] = []

        def of(consumer_id: str) -> RuntimeContext:
            asked.append(consumer_id)
            return context(period_name=consumer_id)

        held = built({"infer-a": Answers(label="car")}, runtime_context=of)

        await held.query_forwarding.answer(
            query(consumer_id="c1_navigation"), arrived_at=time.perf_counter()
        )

        assert asked == ["c1_navigation"]
        assert held.database.queries["q-1"].context.period_name == "c1_navigation"


def built(
    answering: Mapping[str, Answers],
    *,
    serving: Sequence[str] | None = None,
    reachable: bool = True,
    timeout_ms: float = 1000.0,
    together: int | None = None,
    condition: Condition = "day",
    probe_fraction: float = 0.0,
    runtime_context: Contexting | None = None,
    others: Mapping[str, Sequence[str]] | None = None,
) -> Built:
    monitoring_settings = Reader(TREE).build(MonitoringConfig, "services/monitoring.yaml")
    deployments = EnsembleDeployments(
        available_inference_services=tuple(member(name) for name in answering),
        max_ensemble_size=max(3, len(answering)),
    )
    deployments.register(consumer_id="c1_navigation", service_names=tuple(serving or answering))
    for consumer_id, names in (others or {}).items():
        deployments.register(consumer_id=consumer_id, service_names=tuple(names))
    database = MemoryMetricsDatabase()
    members = Members(answering, together=together)
    broker = Published(reachable=reachable)
    monitoring = Monitoring.from_config(monitoring_settings)
    query_forwarding = QueryForwarding(
        deployments=deployments,
        database=database,
        monitoring=monitoring,
        rates=RATES,
        runtime_context=runtime_context or (lambda _: context()),
        condition_of=conditioned(condition),
        publishing=broker,
        member_port=PORT,
        member_timeout_ms=timeout_ms,
        transport=members,
        probe_fraction=probe_fraction,
        max_connections=100,
    )
    OPENED.append(query_forwarding)
    return Built(query_forwarding, database, members, broker, monitoring)


OPENED: list[QueryForwarding] = []


@pytest.fixture(autouse=True)
async def _let_the_connections_go() -> AsyncIterator[None]:
    yield
    while OPENED:
        await OPENED.pop().close()


@pytest.fixture
def three() -> Iterator[Built]:
    held = built(
        {
            "infer-a": Answers(label="car", confidence=0.9),
            "infer-b": Answers(label="car", confidence=0.7),
            "infer-c": Answers(label="bus", confidence=0.6),
        }
    )
    yield held


class TestAnEnsembleThatAnswers:
    async def test_the_consumer_receives_what_the_members_came_to(self, three: Built) -> None:
        result = await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert (result.predicted_label, result.n_members) == ("car", 3)
        assert result.confidence == pytest.approx(0.8)

    async def test_every_member_is_called_by_its_service_name(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert sorted(three.members.asked) == ["infer-a", "infer-b", "infer-c"]

    async def test_every_member_is_asked_for_the_image_the_query_names(self, three: Built) -> None:
        await three.query_forwarding.answer(
            query(image_id="img-42"), arrived_at=time.perf_counter()
        )

        assert three.members.sent == [{"image_id": "img-42", "query_id": "q-1"}] * 3

    async def test_the_members_are_called_at_once_and_not_one_after_another(self) -> None:
        held = built(
            {
                "infer-a": Answers(label="car", confidence=0.9),
                "infer-b": Answers(label="car", confidence=0.7),
                "infer-c": Answers(label="bus", confidence=0.6),
            },
            together=3,
        )

        result = await asyncio.wait_for(
            held.query_forwarding.answer(query(), arrived_at=time.perf_counter()), timeout=10.0
        )

        assert (result.predicted_label, result.n_members) == ("car", 3)

    async def test_one_query_row_names_the_deployment_that_answered(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        stored = three.database.queries["q-1"]
        assert stored.outcome == "accepted"
        assert stored.deployment_id == "c1_navigation-0001"
        assert (stored.predicted_label, stored.n_members) == ("car", 3)

    async def test_one_row_per_member_carries_the_copy_that_answered(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert len(three.rows()) == 3
        assert three.row("infer-b").instance_id == "infer-b-1"
        assert three.row("infer-b").node == "pi-1"

    async def test_a_member_row_holds_that_member_s_own_timings(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        row = three.row("infer-a")
        assert (row.fetch_ms, row.inference_ms, row.total_ms) == (2.0, 18.0, 21.0)

    async def test_the_result_goes_to_the_broker_after_it_is_ready(self, three: Built) -> None:
        result = await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert [one for _, one in three.broker.sent] == [result]


class TestWhereTheTimeWent:
    async def test_the_response_time_is_measured_from_when_the_query_arrived(
        self, three: Built
    ) -> None:
        arrived = time.perf_counter() - 0.5
        await three.query_forwarding.answer(query(), arrived_at=arrived)

        stored = three.database.queries["q-1"]
        assert stored.response_time_ms is not None
        assert stored.response_time_ms >= 500.0

    async def test_the_response_time_ends_when_the_result_is_ready(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        stored = three.database.queries["q-1"]
        assert stored.t_result_ready is not None and stored.t_publish is not None
        assert stored.t_result_ready > query().t_arrive
        assert stored.t_publish > stored.t_result_ready

    async def test_the_publish_time_is_recorded_under_its_own_name(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        stored = three.database.queries["q-1"]
        assert stored.publish_ms is not None and stored.publish_ms > 0.0

    async def test_the_parts_are_measured_on_the_clock_the_whole_is(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        stored = three.database.queries["q-1"]
        assert stored.wait_ms is not None and stored.wait_ms > 0.0
        assert stored.aggregate_ms is not None and stored.aggregate_ms > 0.0
        assert stored.response_time_ms is not None
        assert stored.wait_ms + stored.aggregate_ms <= stored.response_time_ms

    async def test_it_publishes_what_it_measured(self, three: Built) -> None:
        await three.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        body = three.monitoring.exposition()[0]
        assert 'eemls_queries_total{consumer="c1_navigation",outcome="accepted"} 1.0' in body
        assert (
            'eemls_member_calls_total{device="raspberry_pi5",in_ensemble="true",model="b5",'
            'status="ok"} 3.0' in body
        )


class TestAMemberThatDoesNotAnswer:
    @pytest.mark.parametrize(
        ("answers", "status"),
        [
            (Answers(timeout=True), "timeout"),
            (Answers(unreachable=True), "error"),
            (Answers(status_code=500), "error"),
            (Answers(status_code=503), "refused"),
            (Answers(unreadable=True), "error"),
        ],
    )
    async def test_the_ensemble_answers_without_it(self, answers: Answers, status: str) -> None:
        held = built({"infer-a": Answers(label="car", confidence=0.9), "infer-b": answers})

        result = await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert (result.predicted_label, result.n_answered) == ("car", 1)
        assert (result.n_members, held.row("infer-b").status) == (2, status)

    @pytest.mark.parametrize(
        "answers",
        [
            Answers(timeout=True),
            Answers(unreachable=True),
            Answers(status_code=500),
            Answers(status_code=503),
        ],
    )
    async def test_its_row_names_no_copy_because_none_reported(self, answers: Answers) -> None:
        held = built({"infer-a": Answers(), "infer-b": answers})

        await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        row = held.row("infer-b")
        assert (row.instance_id, row.node, row.replica) == (None, None, None)
        assert (row.provider, row.number_format, row.cache_hit) == (None, None, None)
        assert (row.predicted_label, row.confidence) == (None, None)

    async def test_the_time_query_forwarding_waited_for_it_is_on_its_row(self) -> None:
        held = built({"infer-a": Answers(), "infer-b": Answers(timeout=True)})

        await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        waited = held.row("infer-b").total_ms
        assert 0.0 < waited < held.row("infer-a").total_ms
        assert held.row("infer-b").inference_ms == 0.0

    async def test_an_answer_the_row_cannot_hold_is_a_member_that_did_not_answer(self) -> None:
        held = built({"infer-a": Answers(), "infer-b": Answers(replica=0)})

        result = await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert (result.predicted_label, result.n_answered) == ("car", 1)
        assert held.row("infer-b").status == "error"
        assert held.row("infer-b").instance_id is None

    async def test_it_counts_the_call_that_did_not_come_back(self) -> None:
        held = built({"infer-a": Answers(), "infer-b": Answers(timeout=True)})

        await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        body = held.monitoring.exposition()[0]
        assert (
            'eemls_member_calls_total{device="raspberry_pi5",in_ensemble="true",model="b5",'
            'status="timeout"} 1.0' in body
        )


class TestWhenNoMemberAnswers:
    async def test_it_refuses_with_its_own_error_and_not_a_failure_of_the_service(self) -> None:
        held = built({"infer-a": Answers(timeout=True), "infer-b": Answers(status_code=500)})

        with pytest.raises(AggregationError, match="q-1"):
            await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

    async def test_the_query_is_still_stored_as_one_that_failed(self) -> None:
        held = built({"infer-a": Answers(timeout=True), "infer-b": Answers(status_code=500)})

        with pytest.raises(AggregationError):
            await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        stored = held.database.queries["q-1"]
        assert stored.outcome == "failed"
        assert (stored.predicted_label, stored.deployment_id) == (None, None)
        assert stored.response_time_ms is not None and stored.response_time_ms > 0.0
        assert (stored.cost_per_1k, stored.throughput_rps) == (None, None)

    async def test_every_member_that_was_asked_still_holds_a_row(self) -> None:
        held = built({"infer-a": Answers(timeout=True), "infer-b": Answers(status_code=500)})

        with pytest.raises(AggregationError):
            await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert {one.status for one in held.rows()} == {"timeout", "error"}

    async def test_nothing_is_published_when_there_is_nothing_to_publish(self) -> None:
        held = built({"infer-a": Answers(timeout=True)})

        with pytest.raises(AggregationError):
            await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        assert held.broker.sent == []


class TestWhenTheBrokerIsNotThere:
    async def test_the_query_is_stored_even_though_nobody_received_it(self) -> None:
        held = built({"infer-a": Answers()}, reachable=False)

        result = await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())

        stored = held.database.queries["q-1"]
        assert (result.predicted_label, stored.outcome) == ("car", "accepted")
        assert stored.t_publish is None
        assert stored.t_result_ready is not None


class TestAConsumerItDoesNotServe:
    async def test_it_is_refused_by_name(self) -> None:
        held = built({"infer-a": Answers()})

        with pytest.raises(NoSuchConsumerError, match="c9_nowhere"):
            await held.query_forwarding.answer(
                query(consumer_id="c9_nowhere"), arrived_at=time.perf_counter()
            )


class TestTheConnectionsItCallsMembersOn:
    def test_every_connection_it_opens_is_kept_alive_for_the_next_call(self) -> None:
        monitoring_settings = Reader(TREE).build(MonitoringConfig, "services/monitoring.yaml")
        deployments = EnsembleDeployments(
            available_inference_services=(member("infer-a"),), max_ensemble_size=3
        )
        query_forwarding = QueryForwarding(
            deployments=deployments,
            database=MemoryMetricsDatabase(),
            monitoring=Monitoring.from_config(monitoring_settings),
            rates=RATES,
            runtime_context=lambda _: context(),
            condition_of=conditioned("day"),
            publishing=Published(reachable=True),
            member_port=PORT,
            member_timeout_ms=1000.0,
            probe_fraction=0.0,
            max_connections=12,
        )
        OPENED.append(query_forwarding)

        pool = query_forwarding._client._transport._pool  # type: ignore[attr-defined]

        assert (pool._max_connections, pool._max_keepalive_connections) == (12, 12)

    def test_the_bound_covers_every_member_of_every_query_and_the_probes(self) -> None:
        held = member_connections(
            max_concurrent_queries_per_consumer=64,
            consumers=3,
            max_ensemble_size=3,
            probe_fraction=0.2,
            available=8,
        )

        assert held == 64 * 3 * 3 + math.ceil(64 * 3 * 0.2) * 8

    def test_no_probing_needs_no_headroom(self) -> None:
        held = member_connections(
            max_concurrent_queries_per_consumer=4,
            consumers=2,
            max_ensemble_size=3,
            probe_fraction=0.0,
            available=8,
        )

        assert held == 24


def probing(
    fraction: float = 1.0,
    idle: Mapping[str, Answers] | None = None,
    others: Mapping[str, Sequence[str]] | None = None,
) -> Built:
    return built(
        {
            "infer-a": Answers(label="car", confidence=0.9),
            "infer-b": Answers(label="car", confidence=0.7),
            **(idle or {"infer-idle": Answers(label="bus", confidence=0.99)}),
        },
        serving=("infer-a", "infer-b"),
        probe_fraction=fraction,
        others=others,
    )


class TestTheServicesOutsideTheEnsemble:
    async def test_a_probed_query_is_also_sent_to_every_service_outside_it(self) -> None:
        held = probing()

        await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await held.query_forwarding.drain()

        assert sorted(held.members.asked) == ["infer-a", "infer-b", "infer-idle"]

    async def test_a_service_in_the_ensemble_of_another_consumer_is_never_probed(self) -> None:
        held = probing(
            idle={
                "infer-other": Answers(label="bus"),
                "infer-idle": Answers(label="bus"),
            },
            others={"c2_safety": ("infer-other",)},
        )

        for index in range(4):
            await held.query_forwarding.answer(
                query(query_id=f"q-{index}"), arrived_at=time.perf_counter()
            )
        await held.query_forwarding.drain()

        assert "infer-other" not in held.members.asked
        assert held.members.asked.count("infer-idle") == 4

    async def test_a_service_serving_nobody_is_probed_at_the_fraction(self) -> None:
        held = probing(
            fraction=0.25,
            idle={
                "infer-other": Answers(label="bus"),
                "infer-idle": Answers(label="bus"),
            },
            others={"c2_safety": ("infer-other",)},
        )

        for index in range(8):
            await held.query_forwarding.answer(
                query(query_id=f"q-{index}"), arrived_at=time.perf_counter()
            )
        await held.query_forwarding.drain()

        assert held.members.asked.count("infer-idle") == 2
        assert "infer-other" not in held.members.asked

    async def test_their_answer_is_stored_as_outside_the_ensemble(self) -> None:
        held = probing()

        await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await held.query_forwarding.drain()

        stored = held.database.predictions[("q-1", "infer-idle", False)]
        assert (stored.predicted_label, stored.confidence) == ("bus", 0.99)
        assert ("q-1", "infer-idle", True) not in held.database.predictions

    async def test_their_answer_does_not_change_the_result(self) -> None:
        held = probing()

        result = await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await held.query_forwarding.drain()

        assert (result.predicted_label, result.n_members) == ("car", 2)
        assert held.broker.sent[0][1] == result

    async def test_their_answer_is_not_charged_to_the_consumer(self) -> None:
        alone, probed = probing(fraction=0.0), probing()

        await alone.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await probed.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await probed.query_forwarding.drain()

        assert probed.database.queries["q-1"].cost_per_1k == pytest.approx(
            alone.database.queries["q-1"].cost_per_1k
        )
        assert probed.database.queries["q-1"].n_members == 2

    async def test_the_fraction_decides_how_many_queries_are_probed(self) -> None:
        held = probing(fraction=0.25)

        for index in range(8):
            await held.query_forwarding.answer(
                query(query_id=f"q-{index}"), arrived_at=time.perf_counter()
            )
        await held.query_forwarding.drain()

        assert held.members.asked.count("infer-idle") == 2

    async def test_probe_rows_the_store_refuses_are_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        held = probing()
        stored = held.database.write_member_predictions

        def refused(rows: Sequence[MemberPrediction]) -> None:
            if any(not one.in_ensemble for one in rows):
                raise WriteError("closed")
            stored(rows)

        monkeypatch.setattr(held.database, "write_member_predictions", refused)

        with caplog.at_level(logging.ERROR):
            await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())
            await held.query_forwarding.drain()

        assert "error_type=rows_not_written" in caplog.text
        assert "query_id=q-1" in caplog.text

    async def test_no_query_is_probed_at_a_fraction_of_none(self) -> None:
        held = probing(fraction=0.0)

        for index in range(4):
            await held.query_forwarding.answer(
                query(query_id=f"q-{index}"), arrived_at=time.perf_counter()
            )
        await held.query_forwarding.drain()

        assert "infer-idle" not in held.members.asked

    async def test_a_service_outside_it_that_fails_is_stored_as_failed(self) -> None:
        held = probing(idle={"infer-idle": Answers(timeout=True)})

        result = await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await held.query_forwarding.drain()

        assert result.predicted_label == "car"
        stored = held.database.predictions[("q-1", "infer-idle", False)]
        assert (stored.status, stored.predicted_label) == ("timeout", None)

    async def test_their_calls_are_counted_apart_from_the_calls_to_members(self) -> None:
        held = probing()

        await held.query_forwarding.answer(query(), arrived_at=time.perf_counter())
        await held.query_forwarding.drain()

        body = held.monitoring.exposition()[0]
        member = '{device="raspberry_pi5",in_ensemble="true",model="b5"'
        probe = '{device="raspberry_pi5",in_ensemble="false",model="b5"'
        assert f'eemls_member_calls_total{member},status="ok"}} 2.0' in body
        assert f'eemls_member_calls_total{probe},status="ok"}} 1.0' in body
        assert f"eemls_member_call_seconds_count{member}}} 2.0" in body
        assert f"eemls_member_call_seconds_count{probe}}} 1.0" in body

    async def test_the_result_does_not_wait_for_them(self) -> None:
        held = probing(idle={"infer-idle": Answers(label="bus")})
        held.members.hold("infer-idle")

        result = await asyncio.wait_for(
            held.query_forwarding.answer(query(), arrived_at=time.perf_counter()), timeout=10.0
        )

        assert result.predicted_label == "car"
        assert ("q-1", "infer-idle", False) not in held.database.predictions
        held.members.release("infer-idle")
        await held.query_forwarding.drain()
        assert ("q-1", "infer-idle", False) in held.database.predictions
