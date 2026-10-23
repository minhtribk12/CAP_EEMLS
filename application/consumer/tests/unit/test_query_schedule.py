from __future__ import annotations

import pytest
from pydantic import ValidationError

from eemls_consumer.query_schedule import (
    ImageQualityEvent,
    Period,
    QuerySurge,
    Schedule,
    sampled_label,
)

DAY = Period(
    name="morning",
    starts_at_s=0.0,
    class_mix={"pedestrian": 0.5, "car": 0.5},
    condition="day",
    weather="clear",
)
NIGHT = Period(
    name="night",
    starts_at_s=60.0,
    class_mix={"pedestrian": 0.2, "car": 0.8},
    condition="night",
    weather="rain",
)


def schedule(**over: object) -> Schedule:
    held: dict[str, object] = {
        "request_rate_per_s": 20.0,
        "periods": (DAY, NIGHT),
        "day_length_s": 120.0,
        "events": (ImageQualityEvent(starts_at_s=30.0, ends_at_s=40.0, image_quality="blurred"),),
    }
    return Schedule.model_validate(held | over)


class TestThePlaceInTheSimulatedDay:
    @pytest.mark.parametrize(
        ("elapsed_s", "place"),
        [(0.0, 0.0), (30.0, 0.25), (119.9, 119.9 / 120.0), (120.0, 0.0), (390.0, 0.25)],
    )
    def test_it_is_the_time_since_the_day_started_over_the_day_length(
        self, elapsed_s: float, place: float
    ) -> None:
        assert schedule().time_of_day_at(elapsed_s) == pytest.approx(place)

    def test_it_is_always_below_one(self) -> None:
        assert 0.0 <= schedule().time_of_day_at(239.999999) < 1.0

    def test_a_day_with_no_length_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            schedule(day_length_s=0.0)

    def test_a_schedule_must_state_its_day_length(self) -> None:
        with pytest.raises(ValidationError, match="day_length_s"):
            Schedule.model_validate({"request_rate_per_s": 20.0, "periods": (DAY,)})


class TestThePeriod:
    @pytest.mark.parametrize(
        ("elapsed_s", "name"),
        [(0.0, "morning"), (59.9, "morning"), (60.0, "night"), (600.0, "night")],
    )
    def test_the_period_a_moment_falls_in(self, elapsed_s: float, name: str) -> None:
        assert schedule().period_at(elapsed_s).name == name

    def test_the_first_period_starts_the_run(self) -> None:
        with pytest.raises(ValidationError, match="starts at"):
            schedule(periods=(NIGHT,))

    def test_the_periods_follow_one_another(self) -> None:
        with pytest.raises(ValidationError, match="order"):
            schedule(periods=(DAY, NIGHT, NIGHT))

    def test_a_schedule_with_no_period_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            schedule(periods=())


class TestTheImageQuality:
    @pytest.mark.parametrize(
        ("elapsed_s", "quality"),
        [(0.0, "clear"), (29.9, "clear"), (30.0, "blurred"), (39.9, "blurred"), (40.0, "clear")],
    )
    def test_an_event_changes_which_copies_are_asked_for(
        self, elapsed_s: float, quality: str
    ) -> None:
        assert schedule().image_quality_at(elapsed_s) == quality

    def test_an_event_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no length"):
            schedule(events=(ImageQualityEvent(starts_at_s=40.0, ends_at_s=40.0),))

    def test_two_events_over_one_moment_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            schedule(
                events=(
                    ImageQualityEvent(starts_at_s=30.0, ends_at_s=40.0),
                    ImageQualityEvent(starts_at_s=35.0, ends_at_s=50.0),
                )
            )


class TestTheRate:
    @pytest.mark.parametrize(("elapsed_s", "due"), [(0.0, 0), (0.05, 1), (1.0, 20), (10.0, 200)])
    def test_how_many_queries_are_due_by_a_moment(self, elapsed_s: float, due: int) -> None:
        assert schedule().due_by(elapsed_s) == due

    def test_the_rate_is_held_over_a_long_run(self) -> None:
        assert schedule().due_by(300.0) == 6000


class TestTheClassMix:
    @pytest.mark.parametrize(
        ("draw", "label"), [(0.0, "car"), (0.49, "car"), (0.5, "pedestrian"), (0.99, "pedestrian")]
    )
    def test_the_mix_decides_which_class_is_asked_for(self, draw: float, label: str) -> None:
        assert sampled_label(DAY.class_mix, draw) == label

    def test_a_mix_that_does_not_add_up_to_one_is_still_shared_out(self) -> None:
        mix = {"car": 3.0, "bus": 1.0}

        assert [sampled_label(mix, draw) for draw in (0.0, 0.26, 0.9)] == ["bus", "car", "car"]

    def test_a_period_with_no_class_mix_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Period(name="empty", starts_at_s=0.0, class_mix={}, condition="day", weather="clear")


class TestAQuerySurge:
    @pytest.mark.parametrize(
        ("elapsed_s", "due"), [(10.0, 200), (15.0, 500), (20.0, 800), (30.0, 1000)]
    )
    def test_the_rate_is_multiplied_while_the_surge_runs(self, elapsed_s: float, due: int) -> None:
        surged = schedule(
            events=(), surges=(QuerySurge(starts_at_s=10.0, ends_at_s=20.0, multiplier=3.0),)
        )

        assert surged.due_by(elapsed_s) == due

    def test_a_surge_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no length"):
            QuerySurge(starts_at_s=20.0, ends_at_s=20.0, multiplier=3.0)

    @pytest.mark.parametrize("multiplier", [1.0, 0.5])
    def test_a_surge_that_does_not_raise_the_rate_is_refused(self, multiplier: float) -> None:
        with pytest.raises(ValidationError, match="raise"):
            QuerySurge(starts_at_s=10.0, ends_at_s=20.0, multiplier=multiplier)

    def test_two_surges_over_one_moment_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="overlaps"):
            schedule(
                surges=(
                    QuerySurge(starts_at_s=10.0, ends_at_s=20.0, multiplier=2.0),
                    QuerySurge(starts_at_s=15.0, ends_at_s=25.0, multiplier=2.0),
                )
            )

    def test_a_surge_may_run_during_a_quality_drop(self) -> None:
        surged = schedule(surges=(QuerySurge(starts_at_s=30.0, ends_at_s=40.0, multiplier=2.0),))

        assert surged.image_quality_at(35.0) == "blurred"


class TestAPeriodAnEventProduced:
    @pytest.mark.parametrize("event", ["period_change", "class_burst"])
    def test_a_period_names_the_event_that_produced_it(self, event: str) -> None:
        assert DAY.model_copy(update={"event": event}).event == event
        assert Period.model_validate(DAY.model_dump() | {"event": event}).event == event

    def test_an_ordinary_period_names_no_event(self) -> None:
        assert DAY.event is None

    def test_an_event_name_the_paper_does_not_use_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="event"):
            Period.model_validate(DAY.model_dump() | {"event": "earthquake"})
