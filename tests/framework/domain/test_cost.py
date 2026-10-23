from __future__ import annotations

import pytest

from eemls.domain.cost import (
    CostError,
    DeviceRate,
    charge_of,
    cost_per_1k,
    energy_wh_per_1k,
    occupied_hours,
    throughput_per_s,
)

# 20 W, 699 EUR, three years (26,280 h), 0.20 EUR per kWh, one cost unit 0.00001 EUR.
UNIT = 0.00001
RATE = DeviceRate(
    power_watts=20.0,
    hardware_price=699.0,
    lifetime_years=3,
    energy_price_per_kwh=0.20,
    eur_per_cost_unit=UNIT,
)


class TestHowLongAThousandQueriesHoldTheDevice:
    def test_a_thousand_queries_of_one_hundred_milliseconds_hold_it_for_one_hundred_seconds(
        self,
    ) -> None:
        assert occupied_hours(mean_response_time_ms=100.0) == pytest.approx(100.0 / 3600.0)

    def test_a_query_that_takes_no_time_holds_it_for_no_time(self) -> None:
        assert occupied_hours(mean_response_time_ms=0.0) == 0.0


class TestTheEnergyItSpends:
    def test_the_energy_is_the_power_for_as_long_as_it_is_held(self) -> None:
        held = energy_wh_per_1k(RATE, mean_response_time_ms=100.0)

        assert held == pytest.approx(20.0 * 100.0 / 3600.0)
        assert held == pytest.approx(0.5555555, abs=1e-6)

    def test_twice_the_response_time_is_twice_the_energy(self) -> None:
        once = energy_wh_per_1k(RATE, mean_response_time_ms=100.0)
        twice = energy_wh_per_1k(RATE, mean_response_time_ms=200.0)

        assert twice == pytest.approx(2.0 * once)


class TestWhatItCharges:
    def test_the_charge_is_the_energy_it_spends_and_the_hardware_it_occupies(self) -> None:
        energy = 20.0 * 100.0 / 3600.0 / 1000.0 * 0.20
        hardware = 699.0 * (100.0 / 3600.0) / (3 * 365 * 24)

        held = cost_per_1k(RATE, mean_response_time_ms=100.0)

        assert held == pytest.approx((energy + hardware) / UNIT)

    def test_the_charge_is_in_cost_units(self) -> None:
        twice = RATE.model_copy(update={"eur_per_cost_unit": 2 * UNIT})

        assert cost_per_1k(twice, mean_response_time_ms=100.0) == pytest.approx(
            cost_per_1k(RATE, mean_response_time_ms=100.0) / 2
        )

    def test_the_hardware_is_the_larger_part_for_a_device_of_this_price(self) -> None:
        energy = 20.0 * 100.0 / 3600.0 / 1000.0 * 0.20 / UNIT
        held = cost_per_1k(RATE, mean_response_time_ms=100.0)

        assert held - energy > energy

    def test_a_device_that_costs_nothing_charges_only_for_its_energy(self) -> None:
        free = RATE.model_copy(update={"hardware_price": 0.0})

        assert cost_per_1k(free, mean_response_time_ms=100.0) == pytest.approx(
            20.0 * 100.0 / 3600.0 / 1000.0 * 0.20 / UNIT
        )


class TestWhatOneEnsembleCostsOneQuery:
    def test_the_members_that_ran_are_charged_together(self) -> None:
        charge = charge_of([("orin", 20.0), ("pi", 50.0)], {"orin": RATE, "pi": RATE})

        assert charge is not None
        assert charge.cost_per_1k == pytest.approx(
            cost_per_1k(RATE, mean_response_time_ms=20.0)
            + cost_per_1k(RATE, mean_response_time_ms=50.0)
        )

    def test_the_slowest_member_sets_the_rate_the_ensemble_answered_at(self) -> None:
        charge = charge_of([("orin", 20.0), ("pi", 50.0)], {"orin": RATE, "pi": RATE})

        assert charge is not None
        assert charge.throughput_rps == pytest.approx(1000.0 / 50.0)

    def test_a_query_no_member_answered_carries_no_cost(self) -> None:
        assert charge_of([], {"orin": RATE}) is None

    def test_a_device_with_no_price_is_refused_by_name(self) -> None:
        with pytest.raises(CostError, match="orin"):
            charge_of([("orin", 20.0)], {"pi": RATE})

    def test_a_member_that_took_no_time_sets_no_rate(self) -> None:
        with pytest.raises(CostError, match="no time at all"):
            charge_of([("orin", 0.0)], {"orin": RATE})


class TestWhatItRefuses:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("power_watts", 0.0),
            ("power_watts", -1.0),
            ("lifetime_years", 0),
            ("energy_price_per_kwh", -0.1),
            ("hardware_price", -1.0),
        ],
    )
    def test_a_rate_that_cannot_be_charged_is_refused(self, field: str, value: float) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RATE.model_copy(update={field: value})

    def test_a_negative_response_time_is_not_a_response_time(self) -> None:
        from eemls.domain.cost import CostError

        with pytest.raises(CostError, match="response time"):
            cost_per_1k(RATE, mean_response_time_ms=-1.0)


class TestThroughput:
    def test_a_service_answering_in_fifty_milliseconds_serves_twenty_queries_a_second(
        self,
    ) -> None:
        assert throughput_per_s(50.0) == pytest.approx(20.0)

    def test_a_service_that_answered_nothing_serves_no_query_a_second(self) -> None:
        assert throughput_per_s(0.0) == 0.0
