from __future__ import annotations

from collections.abc import Mapping

from pydantic import model_validator

from eemls.config.schema import FrozenMap, FrozenModel, PositiveCount, PositiveNumber, Source
from eemls.domain.cost import DeviceRate


class DeviceCost(FrozenModel):
    hardware_price_eur: PositiveNumber
    idle_power_watts: PositiveNumber
    peak_power_watts: PositiveNumber
    count: PositiveCount
    memory_gb: PositiveCount
    accelerator: str | None
    execution_provider: str
    source: Source

    @model_validator(mode="after")
    def _power_is_ordered(self) -> DeviceCost:
        if self.idle_power_watts >= self.peak_power_watts:
            raise ValueError(
                f"idle power {self.idle_power_watts} W is not below peak power "
                f"{self.peak_power_watts} W"
            )
        return self


class CostConfig(FrozenModel):
    charge_unit: str
    energy_price_per_kwh: PositiveNumber
    device_lifetime_years: PositiveCount
    eur_per_cost_unit: PositiveNumber
    devices: FrozenMap[DeviceCost]

    def rates(self) -> Mapping[str, DeviceRate]:
        return {
            name: DeviceRate(
                power_watts=device.peak_power_watts,
                hardware_price=device.hardware_price_eur,
                lifetime_years=self.device_lifetime_years,
                energy_price_per_kwh=self.energy_price_per_kwh,
                eur_per_cost_unit=self.eur_per_cost_unit,
            )
            for name, device in self.devices.items()
        }

    @model_validator(mode="after")
    def _the_charged_unit_is_the_one_the_paper_reports(self) -> CostConfig:
        if self.charge_unit != "per_1k_queries":
            raise ValueError(f"charge_unit {self.charge_unit} is not per_1k_queries")
        if not self.devices:
            raise ValueError("devices is empty")
        return self
