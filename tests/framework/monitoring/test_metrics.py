from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from eemls.config.reading import Reader, validate
from eemls.config.services import MonitoringConfig
from eemls.monitoring.declared_metrics import DECLARED, DeclaredMetric
from eemls.monitoring.metrics import MetricError, Monitoring
from eemls.monitoring.metrics_endpoint import metrics_router, service_app
from eemls.testing.metrics import (
    bucket_edges,
    histogram_count,
    histogram_sum,
    published,
    series,
)
from eemls.testing.trees import TREE

MONITORING = "services/monitoring.yaml"


async def get(monitoring: Monitoring, path: str) -> httpx.Response:
    app = FastAPI()
    app.include_router(metrics_router(monitoring))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
        return await client.get(path)


@pytest.fixture
def settings() -> MonitoringConfig:
    return validate(MonitoringConfig, Reader(TREE).read_file(MONITORING), MONITORING)


@pytest.fixture
def monitoring(settings: MonitoringConfig) -> Monitoring:
    return Monitoring.from_config(settings)


def body_of(monitoring: Monitoring) -> str:
    return monitoring.exposition()[0]


class TestTheSurface:
    def test_a_timer_records_one_observation_into_its_histogram(
        self, monitoring: Monitoring
    ) -> None:
        elapsed = iter([10.0, 10.25])
        with monitoring.timer(
            "eemls_response_time_seconds", clock=lambda: next(elapsed), consumer="c1_navigation"
        ):
            pass

        body = body_of(monitoring)

        assert histogram_sum(
            body, "eemls_response_time_seconds", consumer="c1_navigation"
        ) == pytest.approx(0.25)
        assert histogram_count(body, "eemls_response_time_seconds", consumer="c1_navigation") == 1.0

    def test_a_timer_records_even_when_the_block_raises(self, monitoring: Monitoring) -> None:
        elapsed = iter([1.0, 1.5])

        def failing_member() -> None:
            with monitoring.timer(
                "eemls_member_call_seconds",
                clock=lambda: next(elapsed),
                model="b6",
                device="jetson_orin_nano",
                in_ensemble="true",
            ):
                raise ValueError("member failed")

        with pytest.raises(ValueError, match="member failed"):
            failing_member()

        assert (
            histogram_count(
                body_of(monitoring),
                "eemls_member_call_seconds",
                model="b6",
                device="jetson_orin_nano",
                in_ensemble="true",
            )
            == 1.0
        )

    def test_a_counter_adds_up(self, monitoring: Monitoring) -> None:
        for _ in range(3):
            monitoring.count("eemls_queries_total", consumer="c1_navigation", outcome="accepted")
        monitoring.count("eemls_queries_total", consumer="c1_navigation", outcome="refused")

        body = body_of(monitoring)

        assert series(
            body, "eemls_queries_total", consumer="c1_navigation", outcome="accepted"
        ) == pytest.approx(3.0)
        assert series(
            body, "eemls_queries_total", consumer="c1_navigation", outcome="refused"
        ) == pytest.approx(1.0)

    def test_a_gauge_holds_the_last_value(self, monitoring: Monitoring) -> None:
        monitoring.gauge("eemls_queries_in_progress", value=7, consumer="c2_safety")
        monitoring.gauge("eemls_queries_in_progress", value=4, consumer="c2_safety")

        assert series(
            body_of(monitoring), "eemls_queries_in_progress", consumer="c2_safety"
        ) == pytest.approx(4.0)

    def test_two_surfaces_keep_their_own_numbers(self, settings: MonitoringConfig) -> None:
        first = Monitoring.from_config(settings)
        second = Monitoring.from_config(settings)

        first.count("eemls_feedback_total", consumer="c1_navigation")

        assert series(body_of(first), "eemls_feedback_total", consumer="c1_navigation") == 1.0
        assert series(body_of(second), "eemls_feedback_total", consumer="c1_navigation") is None


class TestWhatIsRefused:
    def test_an_unknown_metric_name_is_refused_by_name(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="eemls_made_up_total"):
            monitoring.count("eemls_made_up_total", consumer="c1_navigation")

    def test_a_missing_label_is_refused_and_named(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="outcome"):
            monitoring.count("eemls_queries_total", consumer="c1_navigation")

    def test_a_label_outside_the_declared_set_is_refused_and_named(
        self, monitoring: Monitoring
    ) -> None:
        with pytest.raises(MetricError, match="query_id"):
            monitoring.count(
                "eemls_queries_total",
                consumer="c1_navigation",
                outcome="accepted",
                query_id="q-1",
            )

    def test_a_timer_on_a_counter_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="counter"):
            monitoring.timer("eemls_queries_total", consumer="c1", outcome="accepted").__enter__()

    def test_a_count_on_a_histogram_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="histogram"):
            monitoring.count("eemls_response_time_seconds", consumer="c1_navigation")

    def test_a_gauge_on_a_counter_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="counter"):
            monitoring.gauge("eemls_queries_total", value=1, consumer="c1", outcome="accepted")


class TestTheDeclaration:
    @pytest.mark.parametrize(
        ("name", "kind", "labels"),
        [
            ("eemls_queries_total", "counter", ("consumer", "outcome")),
            ("eemls_response_time_seconds", "histogram", ("consumer",)),
            ("eemls_member_call_seconds", "histogram", ("model", "device", "in_ensemble")),
            (
                "eemls_member_calls_total",
                "counter",
                ("model", "device", "status", "in_ensemble"),
            ),
            ("eemls_image_fetch_seconds", "histogram", ("model", "cache")),
            ("eemls_model_inference_seconds", "histogram", ("model", "device")),
            ("eemls_publish_seconds", "histogram", ("consumer",)),
            ("eemls_queries_in_progress", "gauge", ("consumer",)),
            ("eemls_ensemble_size", "gauge", ("consumer",)),
            ("eemls_decisions_total", "counter", ("consumer", "policy", "action", "accepted")),
            ("eemls_decide_seconds", "histogram", ("consumer", "policy")),
            ("eemls_decision_failures_total", "counter", ("consumer",)),
            ("eemls_decisions_skipped_total", "counter", ("consumer",)),
            ("eemls_cdm_value", "gauge", ("consumer", "metric")),
            ("eemls_feedback_total", "counter", ("consumer",)),
            ("eemls_metrics_database_write_seconds", "histogram", ()),
            ("eemls_metrics_database_buffer_rows", "gauge", ()),
            ("eemls_metrics_database_write_failures_total", "counter", ()),
        ],
    )
    def test_the_table_of_the_plan_is_what_the_module_declares(
        self, name: str, kind: str, labels: tuple[str, ...]
    ) -> None:
        declared = DECLARED[name]

        assert declared.kind == kind
        assert declared.labels == labels

    def test_nothing_else_is_declared(self) -> None:
        assert len(DECLARED) == 26

    def test_no_label_can_carry_an_identifier(self) -> None:
        for declared in DECLARED.values():
            for label in declared.labels:
                assert not label.endswith("_id"), label
                assert label not in {"image", "query"}, label


class TestTheEndpoint:
    def test_the_exposition_names_every_declared_metric(self, monitoring: Monitoring) -> None:
        body, content_type = monitoring.exposition()

        assert "text/plain" in content_type
        for name in DECLARED:
            assert published(body, name), name

    def test_the_exposition_carries_a_recorded_value(self, monitoring: Monitoring) -> None:
        monitoring.count("eemls_feedback_total", consumer="c3_surveillance")

        body, _ = monitoring.exposition()

        assert series(body, "eemls_feedback_total", consumer="c3_surveillance") == 1.0

    def test_a_service_mounting_the_router_answers_on_that_path(
        self, monitoring: Monitoring
    ) -> None:
        monitoring.gauge("eemls_ensemble_size", value=3, consumer="c1_navigation")

        response = asyncio.run(get(monitoring, monitoring.path))

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert 'eemls_ensemble_size{consumer="c1_navigation"} 3.0' in response.text

    def test_a_path_the_configuration_does_not_name_is_not_served(
        self, monitoring: Monitoring
    ) -> None:
        response = asyncio.run(get(monitoring, "/metrics-other"))

        assert response.status_code == 404

    def test_the_path_comes_from_the_configuration(
        self, monitoring: Monitoring, settings: MonitoringConfig
    ) -> None:
        assert monitoring.path == settings.metrics_path

    def test_a_request_level_histogram_uses_the_request_level_buckets(
        self, monitoring: Monitoring, settings: MonitoringConfig
    ) -> None:
        with monitoring.timer("eemls_response_time_seconds", consumer="c1_navigation"):
            pass

        edges = bucket_edges(
            body_of(monitoring), "eemls_response_time_seconds", consumer="c1_navigation"
        )

        assert edges[:-1] == [edge / 1000.0 for edge in settings.histogram_buckets_ms]
        assert edges[-1] == float("inf")

    def test_a_fast_histogram_uses_the_fine_buckets(
        self, monitoring: Monitoring, settings: MonitoringConfig
    ) -> None:
        with monitoring.timer("eemls_image_fetch_seconds", model="b6", cache="hit"):
            pass

        edges = bucket_edges(
            body_of(monitoring), "eemls_image_fetch_seconds", model="b6", cache="hit"
        )

        assert edges[:-1] == [edge / 1000.0 for edge in settings.fast_histogram_buckets_ms]
        assert edges[0] < 0.001


class TestTheOffSwitch:
    @pytest.fixture
    def off(self, settings: MonitoringConfig) -> Monitoring:
        return Monitoring.from_config(settings.model_copy(update={"enabled": False}))

    def test_it_is_on_in_the_committed_tree(self, settings: MonitoringConfig) -> None:
        assert settings.enabled is True

    def test_nothing_is_recorded_when_it_is_off(self, off: Monitoring) -> None:
        off.count("eemls_queries_total", consumer="c1_navigation", outcome="accepted")
        off.gauge("eemls_queries_in_progress", value=7, consumer="c1_navigation")
        with off.timer("eemls_response_time_seconds", consumer="c1_navigation"):
            pass

        assert body_of(off) == ""

    def test_the_endpoint_still_answers_when_it_is_off(self, off: Monitoring) -> None:
        assert off.enabled is False
        assert off.path


class TestMemberCalls:
    @pytest.mark.parametrize("status", ["ok", "timeout", "error", "refused"])
    def test_a_member_call_is_counted_by_its_status(
        self, monitoring: Monitoring, status: str
    ) -> None:
        monitoring.count(
            "eemls_member_calls_total",
            model="b6",
            device="jetson_orin_nano",
            status=status,
            in_ensemble="true",
        )

        assert series(
            body_of(monitoring),
            "eemls_member_calls_total",
            device="jetson_orin_nano",
            model="b6",
            status=status,
            in_ensemble="true",
        ) == pytest.approx(1.0)

    def test_a_status_outside_the_four_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="maybe"):
            monitoring.count(
                "eemls_member_calls_total",
                model="b6",
                device="jetson_orin_nano",
                status="maybe",
                in_ensemble="true",
            )


class TestNumbersThatAreNotMeasurements:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_count_is_refused(self, monitoring: Monitoring, value: float) -> None:
        with pytest.raises(MetricError, match="finite"):
            monitoring.count(
                "eemls_queries_total", amount=value, consumer="c1_navigation", outcome="accepted"
            )

        assert (
            series(
                body_of(monitoring),
                "eemls_queries_total",
                consumer="c1_navigation",
                outcome="accepted",
            )
            is None
        )

    def test_a_non_finite_gauge_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="finite"):
            monitoring.gauge(
                "eemls_queries_in_progress", value=float("nan"), consumer="c1_navigation"
            )

    def test_a_counter_cannot_be_asked_to_go_down(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="counts upwards"):
            monitoring.count(
                "eemls_queries_total", amount=-1.0, consumer="c1_navigation", outcome="accepted"
            )

    def test_a_gauge_may_be_negative(self, monitoring: Monitoring) -> None:
        monitoring.gauge("eemls_metrics_database_buffer_rows", value=-1.0)

        assert series(body_of(monitoring), "eemls_metrics_database_buffer_rows") == -1.0


class TestLabelValues:
    @pytest.mark.parametrize(
        ("action", "accepted", "refusal"),
        [
            ("grow", "true", "action"),
            ("add", "True", "accepted"),
        ],
    )
    def test_a_value_outside_the_set_is_refused(
        self, monitoring: Monitoring, action: str, accepted: str, refusal: str
    ) -> None:
        with pytest.raises(MetricError, match=refusal):
            monitoring.count(
                "eemls_decisions_total",
                consumer="c1_navigation",
                policy="cap",
                action=action,
                accepted=accepted,
            )

    def test_the_label_set_is_checked_before_the_values(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="unknown"):
            monitoring.count("eemls_queries_total", consumer="c1", outcome="accepted", cache="nope")


class TestTheMountPath:
    @pytest.mark.parametrize("path", ["metrics", "", "metrics/eemls"])
    def test_a_path_without_a_leading_slash_is_refused(
        self, settings: MonitoringConfig, path: str
    ) -> None:
        with pytest.raises(ValidationError, match="metrics_path"):
            MonitoringConfig.model_validate(settings.model_dump() | {"metrics_path": path})

    def test_buckets_out_of_order_are_refused(self, settings: MonitoringConfig) -> None:
        with pytest.raises(ValidationError, match="rising order"):
            MonitoringConfig.model_validate(
                settings.model_dump() | {"histogram_buckets_ms": [10, 5]}
            )


class TestADeclarationThatCannotBeWrittenDown:
    @pytest.mark.parametrize(
        "description",
        ["Queries | accepted or refused", "First line\nsecond line", "Carriage\rreturn", ""],
    )
    def test_a_description_that_breaks_a_table_cell_is_refused(self, description: str) -> None:
        with pytest.raises(ValueError, match="pipe or a line break|needs a description"):
            DeclaredMetric("counter", ("consumer",), description)

    def test_only_a_histogram_may_be_fast(self) -> None:
        with pytest.raises(ValueError, match="no buckets"):
            DeclaredMetric("counter", ("consumer",), "Something counted", fast=True)


class TestRecordingADurationAlreadyMeasured:
    def test_an_observed_duration_reaches_the_endpoint(self, monitoring: Monitoring) -> None:
        monitoring.observe("eemls_image_fetch_seconds", seconds=0.004, model="b5", cache="miss")

        body = monitoring.exposition()[0]

        assert series(body, "eemls_image_fetch_seconds_count", model="b5", cache="miss") == 1.0
        assert series(
            body, "eemls_image_fetch_seconds_sum", model="b5", cache="miss"
        ) == pytest.approx(0.004)

    def test_a_negative_duration_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="eemls_image_fetch_seconds"):
            monitoring.observe("eemls_image_fetch_seconds", seconds=-1.0, model="b5", cache="miss")

    def test_observing_a_counter_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(MetricError, match="histogram"):
            monitoring.observe("eemls_feedback_total", seconds=0.1, consumer="c1_navigation")


async def asked(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
        return await client.get(path)


class TestAServiceApp:
    def test_a_live_service_answers_its_health_check(self, monitoring: Monitoring) -> None:
        app = service_app("eemls test", monitoring, ready=lambda: False)

        assert asyncio.run(asked(app, "/healthz")).status_code == 200

    def test_a_service_that_is_not_ready_says_so(self, monitoring: Monitoring) -> None:
        app = service_app("eemls test", monitoring, ready=lambda: False)

        assert asyncio.run(asked(app, "/readyz")).status_code == 503

    def test_a_ready_service_says_so(self, monitoring: Monitoring) -> None:
        app = service_app("eemls test", monitoring, ready=lambda: True)

        assert asyncio.run(asked(app, "/readyz")).status_code == 200

    def test_it_serves_the_metrics_on_the_configured_path(self, monitoring: Monitoring) -> None:
        app = service_app("eemls test", monitoring, ready=lambda: True)

        assert asyncio.run(asked(app, monitoring.path)).status_code == 200
