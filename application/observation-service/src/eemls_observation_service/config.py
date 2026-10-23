from __future__ import annotations

from pathlib import Path

from eemls.config.cost import CostConfig
from eemls.config.reading import Reader, read_services, root_of
from eemls.config.schema import FrozenModel
from eemls.config.services import (
    BrokerConfig,
    MetricsDatabaseConfig,
    MonitoringConfig,
    ObservationServiceConfig,
)

FILES_READ: frozenset[str] = frozenset(
    {
        "services/observation_service.yaml",
        "services/metrics_database.yaml",
        "services/monitoring.yaml",
        "services/broker.yaml",
        "cost/prices.yaml",
    }
)


class ObservationServiceViewServices(FrozenModel):
    observation_service: ObservationServiceConfig
    metrics_database: MetricsDatabaseConfig
    monitoring: MonitoringConfig
    broker: BrokerConfig


class ObservationServiceView(FrozenModel):
    services: ObservationServiceViewServices
    cost: CostConfig


def load_observation_service_config(directory: Path | str | None = None) -> ObservationServiceView:
    reader = Reader(root_of(directory))
    return ObservationServiceView(
        services=read_services(reader, ObservationServiceViewServices),
        cost=reader.build(CostConfig, "cost/prices.yaml"),
    )
