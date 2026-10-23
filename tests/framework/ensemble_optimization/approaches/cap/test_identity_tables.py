from __future__ import annotations

import pytest

from eemls.domain.types import Condition, RuntimeContext
from eemls.ensemble_optimization.approaches.cap.identity_tables import (
    IdentityTable,
    IdentityTables,
    TableError,
)

KEYS = ("period_name", "condition")


def context(
    *, period: str = "pedestrian", index: int = 0, condition: Condition = "day"
) -> RuntimeContext:
    return RuntimeContext(
        period_name=period, period_index=index, condition=condition, weather="clear"
    )


class TestATableIsKeyedOnOneAttributeOfTheContext:
    def test_the_period_table_reads_the_period_of_the_context(self) -> None:
        held = IdentityTable("period_name")

        assert held.key_of(context(period="cyclist")) == "cyclist"

    def test_the_condition_table_reads_the_condition_of_the_context(self) -> None:
        held = IdentityTable("condition")

        assert held.key_of(context(condition="night")) == "night"

    def test_a_key_the_runtime_context_does_not_carry_is_refused(self) -> None:
        with pytest.raises(TableError, match="season"):
            IdentityTable("season")

    def test_the_table_says_which_key_it_is_keyed_on(self) -> None:
        assert IdentityTable("condition").context_key == "condition"

    def test_what_the_key_ignores_does_not_split_an_entry(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(condition="day"), "service_a", 0.2)
        held.learn(context(condition="night"), "service_a", 0.4)

        assert held.contribution(context(), "service_a") == pytest.approx(0.3)


class TestItStoresTheAverageContribution:
    def test_one_contribution_is_its_own_average(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(), "service_a", 0.25)

        assert held.contribution(context(), "service_a") == pytest.approx(0.25)

    def test_two_contributions_average(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(), "service_a", 0.2)
        held.learn(context(), "service_a", 0.4)

        assert held.contribution(context(), "service_a") == pytest.approx(0.3)

    def test_it_is_the_average_and_not_the_last_reading(self) -> None:
        held = IdentityTable("period_name")

        for one in (1.0, 0.0, 0.0, 0.0):
            held.learn(context(), "service_a", one)

        assert held.contribution(context(), "service_a") == pytest.approx(0.25)

    def test_a_negative_contribution_lowers_the_average(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(), "service_a", 0.4)
        held.learn(context(), "service_a", -0.4)

        assert held.contribution(context(), "service_a") == pytest.approx(0.0)

    def test_every_inference_service_has_its_own_average(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(), "service_a", 0.5)
        held.learn(context(), "service_b", -0.5)

        assert held.contribution(context(), "service_a") == pytest.approx(0.5)
        assert held.contribution(context(), "service_b") == pytest.approx(-0.5)

    def test_every_value_of_the_key_has_its_own_average(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(period="pedestrian"), "service_a", 0.5)
        held.learn(context(period="cyclist"), "service_a", -0.5)

        assert held.contribution(context(period="pedestrian"), "service_a") == pytest.approx(0.5)
        assert held.contribution(context(period="cyclist"), "service_a") == pytest.approx(-0.5)

    def test_it_averages_without_keeping_the_readings(self) -> None:
        held = IdentityTable("period_name")
        readings = [0.4, -0.2, 0.9, 0.1, -0.7, 0.33, 0.05]

        for one in readings:
            held.learn(context(), "service_a", one)

        expected = sum(readings) / len(readings)
        assert held.contribution(context(), "service_a") == pytest.approx(expected)

    def test_a_long_run_stays_accurate(self) -> None:
        held = IdentityTable("period_name")
        readings = [0.5 + index * 1e-9 for index in range(5000)]

        for one in readings:
            held.learn(context(), "service_a", one)

        expected = sum(readings) / len(readings)
        assert held.contribution(context(), "service_a") == pytest.approx(expected, abs=1e-12)


class TestAContextItHasNothingAbout:
    def test_a_value_of_the_key_it_has_never_measured_contributes_nothing(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(period="pedestrian"), "service_a", 0.5)

        assert held.contribution(context(period="motorcycle"), "service_a") == 0.0

    def test_a_service_it_has_never_measured_contributes_nothing(self) -> None:
        held = IdentityTable("period_name")

        held.learn(context(), "service_a", 0.5)

        assert held.contribution(context(), "service_b") == 0.0

    def test_an_empty_table_contributes_nothing(self) -> None:
        assert IdentityTable("condition").contribution(context(), "service_a") == 0.0


class TestBothTablesOfOneConsumer:
    def test_they_are_the_period_and_the_condition(self) -> None:
        held = IdentityTables(KEYS)

        assert held.by_key["period_name"].context_key == "period_name"
        assert held.by_key["condition"].context_key == "condition"

    def test_one_contribution_reaches_both_of_them(self) -> None:
        held = IdentityTables(KEYS)

        held.learn(context(period="cyclist", condition="night"), "service_a", 0.3)

        by_period = held.by_key["period_name"].contribution(context(period="cyclist"), "service_a")
        by_condition = held.by_key["condition"].contribution(
            context(condition="night"), "service_a"
        )

        assert by_period == pytest.approx(0.3)
        assert by_condition == pytest.approx(0.3)

    def test_the_lookup_gives_the_period_value_and_then_the_condition_value(self) -> None:
        held = IdentityTables(KEYS)

        held.learn(context(period="cyclist", condition="day"), "service_a", 0.2)
        held.learn(context(period="pedestrian", condition="day"), "service_a", 0.6)

        period, condition = held.looked_up(context(period="cyclist", condition="day"), "service_a")

        assert period == pytest.approx(0.2)
        assert condition == pytest.approx(0.4)

    def test_the_condition_separates_two_services_with_the_same_period(self) -> None:
        held = IdentityTables(KEYS)

        measured: tuple[tuple[str, Condition, float], ...] = (
            ("service_a", "day", 0.1),
            ("service_a", "night", 0.5),
            ("service_b", "day", 0.1),
            ("service_b", "night", -0.3),
        )
        for service_name, condition, mine in measured:
            held.learn(context(condition=condition), service_name, mine)

        at_night = context(condition="night")

        assert held.looked_up(at_night, "service_a")[1] > held.looked_up(at_night, "service_b")[1]
        by_day = context(condition="day")
        assert held.looked_up(by_day, "service_a")[1] == pytest.approx(
            held.looked_up(by_day, "service_b")[1]
        )

    def test_a_consumer_that_has_learned_nothing_looks_up_nothing(self) -> None:
        assert IdentityTables(KEYS).looked_up(context(), "service_a") == (0.0, 0.0)

    def test_two_consumers_keep_their_own_tables(self) -> None:
        one, other = IdentityTables(KEYS), IdentityTables(KEYS)

        one.learn(context(), "service_a", 0.5)

        assert other.looked_up(context(), "service_a") == (0.0, 0.0)
