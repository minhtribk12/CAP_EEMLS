from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta

import httpx
import pytest

from eemls.domain.cost import DeviceRate, cost_per_1k, energy_wh_per_1k
from eemls.domain.measuring import NoMeasuring
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Row, Timestamp
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.observation_service.metrics_reporting import MetricsReporter, ReportingDatabase
from eemls.observation_service.profile_updates import (
    ProfileError,
    ProfileSettings,
    ProfileUpdatesClient,
    update_profiles,
    updated_profiles,
)
from eemls.observation_service.testing import (
    InProcessObservationService,
    in_process_profile_updates,
)
from eemls.testing.records import AT, feedback, prediction, profile, served

RATES = {
    "jetson_orin_nano": DeviceRate(
        power_watts=20.0,
        hardware_price=699.0,
        lifetime_years=3,
        energy_price_per_kwh=0.20,
        eur_per_cost_unit=0.00001,
    )
}
SERVICE = "infer-b6-jetson_orin_nano"
URL = "http://observation-service:8083"


def stored(*, wrong: int = 4, in_ensemble: bool = False) -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase()
    database.write_profile(
        profile(
            service_name=SERVICE,
            general_accuracy=0.81,
            window_start=AT - timedelta(minutes=10),
            window_end=AT - timedelta(minutes=5),
        )
    )
    for index in range(wrong):
        query_id = f"q-{index}"
        arrived = AT + timedelta(seconds=index)
        database.write_query(
            served(
                query_id=query_id,
                t_arrive=arrived,
                t_result_ready=arrived + timedelta(milliseconds=174),
            )
        )
        database.write_member_predictions(
            (
                prediction(
                    query_id=query_id,
                    in_ensemble=in_ensemble,
                    predicted_label="bus",
                    total_ms=30.0,
                ),
            )
        )
        database.write_feedback(feedback(query_id=query_id, true_label="car"))
    return database


class Unreadable(MemoryMetricsDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self.reads += 1
        raise OSError("the database is gone")


def unreadable() -> Unreadable:
    database = Unreadable()
    for one in stored().profiles.values():
        database.write_profile(one)
    return database


def updates(database: MemoryMetricsDatabase, **changes: float) -> ProfileUpdatesClient:
    settings: dict[str, float] = {"window_s": 60.0, "interval_s": 30.0} | changes
    return in_process_profile_updates(
        database,
        rates=RATES,
        window_s=settings["window_s"],
        interval_s=settings["interval_s"],
        min_labeled_per_class=1,
        min_labeled_overall=int(settings.get("min_labeled_overall", 1)),
    )


class TestTheProfilesItHands:
    def test_before_the_first_interval_the_stored_profiles_are_handed(self) -> None:
        held = updates(stored())

        shown = held.latest(AT + timedelta(seconds=10))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.81)

    def test_after_an_interval_the_profile_is_updated_from_the_runtime_answers(self) -> None:
        held = updates(stored())
        held.latest(AT + timedelta(seconds=10))

        shown = held.latest(AT + timedelta(seconds=40))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.0)
        assert shown[SERVICE].mean_ms == pytest.approx(30.0)

    def test_the_answers_of_members_count_as_well(self) -> None:
        held = updates(stored(in_ensemble=True))
        held.latest(AT + timedelta(seconds=10))

        shown = held.latest(AT + timedelta(seconds=40))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.0)

    def test_the_updated_profile_is_stored(self) -> None:
        database = stored()
        held = updates(database)
        held.latest(AT + timedelta(seconds=10))

        held.latest(AT + timedelta(seconds=40))

        latest = {one.service_name: one for one in database.read_profiles()}
        assert latest[SERVICE].general_accuracy == pytest.approx(0.0)
        assert latest[SERVICE].window_end == AT + timedelta(seconds=40)

    def test_within_an_interval_nothing_is_updated(self) -> None:
        database = stored()
        held = updates(database)
        held.latest(AT + timedelta(seconds=10))
        held.latest(AT + timedelta(seconds=40))

        held.latest(AT + timedelta(seconds=50))

        assert len(database.profiles) == 2

    def test_only_the_answers_inside_the_window_are_read(self) -> None:
        held = updates(stored(), window_s=1.0)
        held.latest(AT)

        shown = held.latest(AT + timedelta(seconds=40))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.81)

    def test_fewer_labels_than_the_overall_minimum_keep_the_general_accuracy(self) -> None:
        held = updates(stored(), min_labeled_overall=5)
        held.latest(AT + timedelta(seconds=10))

        shown = held.latest(AT + timedelta(seconds=40))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.81)

    def test_a_service_with_no_answer_keeps_its_profile(self) -> None:
        held = updates(stored(wrong=0))
        held.latest(AT)

        shown = held.latest(AT + timedelta(seconds=40))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.81)


class TestAnUpdateThatFails:
    def test_the_held_profiles_are_handed(self) -> None:
        held = updates(unreadable())
        held.latest(AT + timedelta(seconds=10))

        shown = held.latest(AT + timedelta(seconds=40))

        assert shown[SERVICE].general_accuracy == pytest.approx(0.81)

    def test_it_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        held = updates(unreadable())
        held.latest(AT + timedelta(seconds=10))

        with caplog.at_level(logging.ERROR):
            held.latest(AT + timedelta(seconds=40))

        assert "error_type=profile_rebuild_failed" in caplog.text
        assert "recovery_action=the_held_profiles_are_kept" in caplog.text

    def test_it_is_tried_again_only_after_an_interval(self) -> None:
        database = unreadable()
        held = updates(database)
        held.latest(AT + timedelta(seconds=10))
        held.latest(AT + timedelta(seconds=40))

        held.latest(AT + timedelta(seconds=50))
        assert database.reads == 1
        held.latest(AT + timedelta(seconds=70))
        assert database.reads == 2


START = AT + timedelta(minutes=10)
END = AT + timedelta(minutes=11)
RATE = RATES["jetson_orin_nano"]
CURRENT = {"infer-b6-orin": profile()}


def answered(
    total_ms: float,
    *,
    predicted: str | None = "car",
    true_label: str | None = "car",
    service_name: str = "infer-b6-orin",
    instance_id: str = "i-9",
    in_ensemble: bool = True,
    status: str = "ok",
    confidence: float | None = 0.9,
) -> Row:
    return {
        "instance_id": instance_id,
        "service_name": service_name,
        "model": "b6",
        "device": "jetson_orin_nano",
        "node": "orin-2",
        "in_ensemble": in_ensemble,
        "total_ms": total_ms,
        "status": status,
        "predicted_label": predicted,
        "true_label": true_label,
        "confidence": confidence,
    }


def profiles_updated(
    rows: tuple[Row, ...],
    *,
    current: dict[str, InferenceServiceProfile] | None = None,
    min_labeled_per_class: int = 1,
    min_labeled_overall: int = 1,
) -> tuple[InferenceServiceProfile, ...]:
    return updated_profiles(
        rows,
        current=CURRENT if current is None else current,
        settings=ProfileSettings(
            window_start=START,
            window_end=END,
            rates=RATES,
            min_labeled_per_class=min_labeled_per_class,
            min_labeled_overall=min_labeled_overall,
        ),
    )


class TestOneProfileFromKnownRows:
    def rows(self) -> tuple[Row, ...]:
        return (
            answered(10.0),
            answered(20.0),
            answered(30.0),
            answered(40.0, predicted="truck"),
        )

    def profile(self) -> InferenceServiceProfile:
        held = profiles_updated(self.rows())
        assert len(held) == 1
        return held[0]

    def test_the_response_times_match_hand_arithmetic(self) -> None:
        held = self.profile()

        assert held.mean_ms == pytest.approx(25.0)
        assert held.p50_ms == pytest.approx(25.0)
        assert held.max_ms == pytest.approx(40.0)
        assert held.p95_ms == pytest.approx(38.5)

    def test_the_counts_match_hand_arithmetic(self) -> None:
        assert self.profile().n_queries == 4

    def test_the_accuracy_matches_hand_arithmetic(self) -> None:
        held = self.profile()

        assert held.general_accuracy == pytest.approx(0.75)
        assert held.accuracy_by_class["car"] == pytest.approx(0.75)

    def test_the_throughput_is_what_one_copy_answers_a_second(self) -> None:
        assert self.profile().throughput_rps == pytest.approx(1000.0 / 25.0)

    def test_the_energy_and_the_cost_come_from_the_rate_and_the_mean(self) -> None:
        held = self.profile()

        assert held.energy_wh_per_1k == pytest.approx(
            energy_wh_per_1k(RATE, mean_response_time_ms=25.0)
        )
        assert held.cost_per_1k == pytest.approx(cost_per_1k(RATE, mean_response_time_ms=25.0))

    def test_it_says_it_was_measured(self) -> None:
        assert self.profile().source == "measured"

    def test_it_carries_the_window_it_was_built_over(self) -> None:
        held = self.profile()

        assert (held.window_start, held.window_end) == (START, END)

    def test_it_keeps_the_identity_of_the_profile_it_updates(self) -> None:
        held = self.profile()

        assert (held.model, held.device, held.node, held.instance_id, held.service_name) == (
            "b6",
            "jetson_orin_nano",
            "orin-1",
            "i-1",
            "infer-b6-orin",
        )


class TestTheRowsItReads:
    def test_answers_inside_and_outside_the_ensemble_both_count(self) -> None:
        rows = (answered(10.0), answered(30.0, predicted="bus", in_ensemble=False))

        held = profiles_updated(rows)[0]

        assert held.n_queries == 2
        assert held.general_accuracy == pytest.approx(0.5)

    def test_every_copy_of_one_service_makes_one_profile(self) -> None:
        rows = (answered(10.0, instance_id="i-1"), answered(30.0, instance_id="i-2"))

        held = profiles_updated(rows)

        assert len(held) == 1
        assert held[0].mean_ms == pytest.approx(20.0)

    def test_a_service_with_no_profile_to_update_gets_none(self) -> None:
        rows = (answered(10.0, service_name="infer-nobody"),)

        assert profiles_updated(rows) == ()

    def test_the_profiles_come_back_in_a_stable_order(self) -> None:
        current = {
            "infer-b6-orin": profile(),
            "infer-b5-orin": profile(service_name="infer-b5-orin", instance_id="i-5"),
        }
        rows = (answered(10.0), answered(10.0, service_name="infer-b5-orin"))

        held = profiles_updated(rows, current=current)

        assert [one.service_name for one in held] == ["infer-b5-orin", "infer-b6-orin"]


class TestWhatItKeepsFromTheProfileItUpdates:
    def test_a_class_with_too_few_labels_keeps_its_profiled_values(self) -> None:
        rows = (answered(10.0, predicted="bus", true_label="truck", confidence=0.3),)

        held = profiles_updated(rows, min_labeled_per_class=2)[0]

        assert held.accuracy_by_class["truck"] == pytest.approx(0.72)
        assert held.confidence_by_class["truck"] == pytest.approx(0.64)

    def test_a_class_with_enough_labels_takes_its_measured_values(self) -> None:
        rows = (
            answered(10.0, predicted="bus", true_label="truck", confidence=0.3),
            answered(10.0, predicted="truck", true_label="truck", confidence=0.5),
        )

        held = profiles_updated(rows, min_labeled_per_class=2)[0]

        assert held.accuracy_by_class["truck"] == pytest.approx(0.5)
        assert held.confidence_by_class["truck"] == pytest.approx(0.4)

    def test_a_class_never_seen_keeps_its_profiled_values(self) -> None:
        held = profiles_updated((answered(10.0),))[0]

        assert held.accuracy_by_class["truck"] == pytest.approx(0.72)

    def test_too_few_labels_keep_the_profiled_general_accuracy(self) -> None:
        held = profiles_updated((answered(10.0, true_label=None),), min_labeled_per_class=1)[0]

        assert held.general_accuracy == pytest.approx(0.81)
        assert held.mean_ms == pytest.approx(10.0)

    def test_fewer_labels_than_the_overall_minimum_keep_the_profiled_general_accuracy(
        self,
    ) -> None:
        rows = (answered(10.0), answered(10.0, predicted="bus"))

        held = profiles_updated(rows, min_labeled_per_class=1, min_labeled_overall=3)[0]

        assert held.general_accuracy == pytest.approx(0.81)

    def test_the_overall_minimum_alone_gates_the_general_accuracy(self) -> None:
        rows = (
            answered(10.0),
            answered(10.0, predicted="bus", true_label="truck"),
            answered(10.0, predicted="bus", true_label="bus"),
        )

        held = profiles_updated(rows, min_labeled_per_class=5, min_labeled_overall=3)[0]

        assert held.general_accuracy == pytest.approx(2 / 3)
        assert held.accuracy_by_class["truck"] == pytest.approx(0.72)

    def test_no_labels_and_no_overall_minimum_give_a_general_accuracy_of_zero(self) -> None:
        held = profiles_updated((answered(10.0, true_label=None),), min_labeled_overall=0)[0]

        assert held.general_accuracy == 0.0


class TestWhatIsNotCounted:
    def test_a_call_that_was_not_answered_is_not_counted_and_lends_no_timing(self) -> None:
        rows = (
            answered(10.0),
            answered(900.0, predicted=None, status="timeout", confidence=None),
        )

        held = profiles_updated(rows)[0]

        assert held.n_queries == 1
        assert held.mean_ms == pytest.approx(10.0)
        assert held.general_accuracy == pytest.approx(1.0)

    def test_a_service_that_answered_nothing_gets_no_new_profile(self) -> None:
        rows = (answered(900.0, predicted=None, status="error", confidence=None),)

        assert profiles_updated(rows) == ()


class TestWhatItRefuses:
    def test_a_device_with_no_rate_is_refused_by_name(self) -> None:
        current = {"infer-b6-orin": profile(device="a_device_nobody_priced")}

        with pytest.raises(ProfileError, match="a_device_nobody_priced"):
            profiles_updated((answered(10.0),), current=current)

    def test_a_window_with_no_length_is_refused(self) -> None:
        with pytest.raises(ProfileError, match="no length"):
            updated_profiles(
                (),
                current=CURRENT,
                settings=ProfileSettings(
                    window_start=END,
                    window_end=END,
                    rates=RATES,
                    min_labeled_per_class=1,
                    min_labeled_overall=1,
                ),
            )


UPDATE_END = AT + timedelta(seconds=40)


def measured_database() -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase()
    database.write_profile(
        profile(
            service_name=SERVICE,
            window_start=AT - timedelta(minutes=10),
            window_end=AT - timedelta(minutes=5),
        )
    )
    for index in range(4):
        query_id = f"q-{index}"
        arrived = AT + timedelta(seconds=index)
        database.write_query(
            served(
                query_id=query_id,
                t_arrive=arrived,
                t_result_ready=arrived + timedelta(milliseconds=174),
            )
        )
        database.write_member_predictions(
            (prediction(query_id=query_id, predicted_label="bus", total_ms=30.0),)
        )
        database.write_feedback(feedback(query_id=query_id, true_label="car"))
    return database


def update_settings() -> ProfileSettings:
    return ProfileSettings.ending_at(
        UPDATE_END, window_s=60.0, rates=RATES, min_labeled_per_class=1, min_labeled_overall=1
    )


class TestUpdatingTheStoredProfiles:
    def test_the_settings_cover_the_window_that_ends_at_the_update(self) -> None:
        held = update_settings()

        assert (held.window_start, held.window_end) == (
            UPDATE_END - timedelta(seconds=60),
            UPDATE_END,
        )

    def test_the_updated_profile_is_measured_over_the_window(self) -> None:
        (updated,) = update_profiles(measured_database(), update_settings())

        assert updated.general_accuracy == pytest.approx(0.0)
        assert updated.mean_ms == pytest.approx(30.0)
        assert updated.window_end == UPDATE_END

    def test_the_updated_profiles_are_stored(self) -> None:
        database = measured_database()

        updated = update_profiles(database, update_settings())

        assert updated[0] in database.profiles.values()


def forty_answered() -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase()
    database.write_profile(
        profile(
            service_name=SERVICE,
            general_accuracy=0.81,
            window_start=AT - timedelta(minutes=10),
            window_end=AT - timedelta(minutes=5),
        )
    )
    for index in range(40):
        query_id = f"q-{index}"
        arrived = AT + timedelta(seconds=5 * index)
        database.write_query(
            served(
                query_id=query_id,
                t_arrive=arrived,
                t_result_ready=arrived + timedelta(milliseconds=174),
            )
        )
        database.write_member_predictions(
            (prediction(query_id=query_id, predicted_label="bus", total_ms=30.0 + index),)
        )
        database.write_feedback(feedback(query_id=query_id, true_label="car"))
    return database


class Updates(MemoryMetricsDatabase):
    def __init__(self) -> None:
        super().__init__()
        source = forty_answered()
        self.queries = source.queries
        self.predictions = source.predictions
        self.feedback = source.feedback
        self.profiles = source.profiles
        self.at: list[datetime] = []

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self.at.append(until)
        return super().read_member_answers(since=since, until=until)


MOMENTS = [AT + timedelta(seconds=s) for s in (0, 5, 29, 30, 31, 59, 61, 62, 95, 200, 201)]


def client_on(database: MetricsDatabase, service: httpx.MockTransport) -> ProfileUpdatesClient:
    return ProfileUpdatesClient(
        database=database,
        base_url=URL,
        interval_s=30.0,
        request_timeout_ms=500.0,
        transport=service,
    )


def in_process(database: MemoryMetricsDatabase) -> InProcessObservationService:
    return InProcessObservationService(
        database,
        run_id="run-1",
        rates=RATES,
        window_s=60.0,
        min_labeled_per_class=1,
        min_labeled_overall=1,
    )


class TestTheProfileUpdatesClient:
    def test_it_updates_once_an_interval_has_passed_since_the_last_update(self) -> None:
        remote = Updates()
        client = client_on(remote, in_process(remote))

        for at in MOMENTS:
            client.latest(at)

        assert remote.at == [AT + timedelta(seconds=s) for s in (30, 61, 95, 200)]

    def test_the_metrics_reporter_is_flushed_before_the_update(self) -> None:
        log: list[str] = []
        stored = forty_answered()
        service = in_process(stored)

        def recorded(request: httpx.Request) -> httpx.Response:
            log.append(request.url.path)
            return service.handle_request(request)

        reporter = MetricsReporter(
            base_url=URL,
            run_id="run-1",
            sender="control-plane-1",
            monitoring=NoMeasuring(),
            buffer_rows=100,
            flush_interval_ms=60_000.0,
            request_timeout_ms=500.0,
            transport=httpx.MockTransport(recorded),
        )
        database = ReportingDatabase(stored, reporter)
        client = client_on(database, httpx.MockTransport(recorded))
        client.latest(AT)
        database.write_feedback(feedback(query_id="q-90"))

        client.latest(AT + timedelta(seconds=30))

        assert log == ["/rows", "/profiles/update"]

    def test_a_failed_update_keeps_the_held_profiles(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stored = forty_answered()
        client = client_on(stored, httpx.MockTransport(lambda request: httpx.Response(503)))
        before = client.latest(AT)

        with caplog.at_level(logging.ERROR):
            after = client.latest(AT + timedelta(seconds=30))

        assert after == before
        assert "profile_rebuild_failed" in caplog.text
