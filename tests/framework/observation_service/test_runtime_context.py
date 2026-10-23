from __future__ import annotations

from eemls.domain.types import Row
from eemls.evaluation_engine.window import Window
from eemls.observation_service.runtime_context import UNNAMED, RuntimeContextReader

DOMINANT = ("pedestrian", "cyclist", "motorcycle", "car", "truck", "bus")

ONE = "c1_navigation"

OTHER = "c2_safety"


def row(label: str | None, *, condition: str = "day") -> Row:
    return {"predicted_label": label, "condition": condition}


def window(*rows: Row) -> Window:
    return Window(rows)


def built() -> RuntimeContextReader:
    return RuntimeContextReader(dominant_classes=DOMINANT)


class TestBeforeAnythingHasBeenServed:
    def test_no_period_is_named(self) -> None:
        held = built()(ONE)

        assert held.period_name == UNNAMED
        assert held.period_index == 0

    def test_the_condition_is_the_day(self) -> None:
        assert built()(ONE).condition == "day"

    def test_an_empty_window_names_nothing(self) -> None:
        held = built()

        held.read_from(ONE, window())

        assert held(ONE).period_name == UNNAMED


class TestNamingThePeriod:
    def test_the_period_is_the_class_most_of_the_answers_carry(self) -> None:
        held = built()

        held.read_from(ONE, window(row("pedestrian"), row("pedestrian"), row("car")))

        assert held(ONE).period_name == "pedestrian"

    def test_the_index_is_the_place_of_that_class_among_the_dominant_ones(self) -> None:
        held = built()

        held.read_from(ONE, window(row("motorcycle")))

        assert held(ONE).period_index == 3

    def test_every_dominant_class_can_name_a_period(self) -> None:
        held = built()

        for index, name in enumerate(DOMINANT, start=1):
            held.read_from(ONE, window(row(name)))

            assert (held(ONE).period_name, held(ONE).period_index) == (name, index)

    def test_the_period_changes_when_another_class_comes_to_dominate(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car")))
        held.read_from(ONE, window(row("bus"), row("bus"), row("car")))

        assert held(ONE).period_name == "bus"

    def test_a_class_that_is_not_a_dominant_one_names_no_period(self) -> None:
        held = built()

        held.read_from(ONE, window(row("bird"), row("bird"), row("car")))

        assert held(ONE).period_name == "car"

    def test_a_window_of_tail_classes_alone_keeps_the_period_it_had(self) -> None:
        held = built()

        held.read_from(ONE, window(row("truck")))
        held.read_from(ONE, window(row("bird"), row("boat")))

        assert held(ONE).period_name == "truck"

    def test_a_query_nobody_answered_does_not_vote(self) -> None:
        held = built()

        held.read_from(ONE, window(row(None), row(None), row("cyclist")))

        assert held(ONE).period_name == "cyclist"


class TestWhenNoSingleClassLeads:
    def test_a_tie_keeps_the_period_it_had(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car")))
        held.read_from(ONE, window(row("bus"), row("truck")))

        assert held(ONE).period_name == "car"

    def test_a_tie_before_any_period_names_nothing(self) -> None:
        held = built()

        held.read_from(ONE, window(row("bus"), row("truck")))

        assert held(ONE).period_name == UNNAMED
        assert held(ONE).period_index == 0


class TestTheCondition:
    def test_it_is_the_condition_of_the_most_recent_query(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car", condition="day"), row("car", condition="night")))

        assert held(ONE).condition == "night"

    def test_it_holds_until_a_query_says_otherwise(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car", condition="night")))
        held.read_from(ONE, window())

        assert held(ONE).condition == "night"

    def test_a_condition_the_runtime_context_does_not_know_is_ignored(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car", condition="night")))
        held.read_from(ONE, window(row("car", condition="dusk")))

        assert held(ONE).condition == "night"


class TestEachConsumerHasItsOwnContext:
    def test_a_consumer_does_not_take_the_period_another_consumer_named(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car")))
        held.read_from(OTHER, window(row("bus"), row("truck")))

        assert held(OTHER).period_name == UNNAMED
        assert held(ONE).period_name == "car"

    def test_two_consumers_keep_their_own_periods(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car")))
        held.read_from(OTHER, window(row("pedestrian")))

        assert (held(ONE).period_name, held(OTHER).period_name) == ("car", "pedestrian")

    def test_a_consumer_does_not_take_the_condition_another_consumer_read(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car", condition="night")))
        held.read_from(OTHER, window())

        assert held(OTHER).condition == "day"
        assert held(ONE).condition == "night"

    def test_a_consumer_nothing_was_read_for_has_no_period_yet(self) -> None:
        held = built()

        held.read_from(ONE, window(row("car")))

        assert (held(OTHER).period_name, held(OTHER).period_index) == (UNNAMED, 0)
