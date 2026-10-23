from __future__ import annotations

from collections.abc import Mapping, Sequence

from eemls.domain.errors import EemlsError
from eemls.domain.types import (
    FrozenModel,
    NonNegativeNumber,
    PositiveCount,
    PositiveNumber,
)

HOURS_IN_A_YEAR = 365 * 24
SECONDS_IN_AN_HOUR = 3600.0
WATT_HOURS_IN_A_KILOWATT_HOUR = 1000.0
QUERIES = 1000
MILLISECONDS_IN_A_SECOND = 1000.0


class CostError(EemlsError): ...


class DeviceRate(FrozenModel):
    power_watts: PositiveNumber
    hardware_price: NonNegativeNumber
    lifetime_years: PositiveCount
    energy_price_per_kwh: NonNegativeNumber
    eur_per_cost_unit: PositiveNumber


class Charge(FrozenModel):
    cost_per_1k: NonNegativeNumber
    throughput_rps: PositiveNumber


def charge_of(ran: Sequence[tuple[str, float]], rates: Mapping[str, DeviceRate]) -> Charge | None:
    if not ran:
        return None
    charged = 0.0
    slowest = 0.0
    for device, total_ms in ran:
        rate = rates.get(device)
        if rate is None:
            raise CostError(f"{device} has no rate, so nothing can be charged for it")
        charged += cost_per_1k(rate, mean_response_time_ms=total_ms)
        slowest = max(slowest, total_ms)
    if slowest <= 0.0:
        raise CostError("a member answered in no time at all, so no rate follows from it")
    return Charge(cost_per_1k=charged, throughput_rps=throughput_per_s(slowest))


def throughput_per_s(response_time_ms: float) -> float:
    return MILLISECONDS_IN_A_SECOND / response_time_ms if response_time_ms > 0.0 else 0.0


def occupied_hours(*, mean_response_time_ms: float) -> float:
    if mean_response_time_ms < 0:
        raise CostError(f"{mean_response_time_ms} is not a response time")
    return QUERIES * mean_response_time_ms / MILLISECONDS_IN_A_SECOND / SECONDS_IN_AN_HOUR


def energy_wh_per_1k(rate: DeviceRate, *, mean_response_time_ms: float) -> float:
    return rate.power_watts * occupied_hours(mean_response_time_ms=mean_response_time_ms)


def cost_per_1k(rate: DeviceRate, *, mean_response_time_ms: float) -> float:
    held = occupied_hours(mean_response_time_ms=mean_response_time_ms)
    energy_wh = energy_wh_per_1k(rate, mean_response_time_ms=mean_response_time_ms)
    energy = energy_wh / WATT_HOURS_IN_A_KILOWATT_HOUR * rate.energy_price_per_kwh
    hardware = rate.hardware_price * held / (rate.lifetime_years * HOURS_IN_A_YEAR)
    return (energy + hardware) / rate.eur_per_cost_unit
