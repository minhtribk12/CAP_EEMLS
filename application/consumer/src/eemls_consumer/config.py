from __future__ import annotations

from pathlib import Path

from eemls.config.consumers import ConsumerConfig, load_consumers
from eemls.config.datasets import DatasetClasses
from eemls.config.reading import Reader, read_services, root_of
from eemls.config.schema import FrozenMap, FrozenModel
from eemls.config.services import (
    BrokerConfig,
    ConsumerServiceConfig,
    ControlPlaneConfig,
    MetricsDatabaseConfig,
    MonitoringConfig,
)
from eemls.domain.contract import ConsumerContract

FILES_READ: frozenset[str] = frozenset(
    {
        "services/consumer.yaml",
        "services/monitoring.yaml",
        "services/broker.yaml",
        "consumers",
    }
)


class ConsumerViewServices(FrozenModel):
    consumer: ConsumerServiceConfig
    monitoring: MonitoringConfig
    broker: BrokerConfig


class ConsumerView(FrozenModel):
    services: ConsumerViewServices
    contracts: FrozenMap[ConsumerContract]


class LoadTestViewServices(FrozenModel):
    control_plane: ControlPlaneConfig
    consumer: ConsumerServiceConfig
    metrics_database: MetricsDatabaseConfig


class LoadTestView(FrozenModel):
    services: LoadTestViewServices
    consumers: FrozenMap[ConsumerConfig]
    datasets: DatasetClasses


def load_consumer_config(directory: Path | str | None = None) -> ConsumerView:
    reader = Reader(root_of(directory))
    return ConsumerView(
        services=read_services(reader, ConsumerViewServices),
        contracts={name: held.contract for name, held in load_consumers(reader).items()},
    )


def load_load_test_config(directory: Path | str | None = None) -> LoadTestView:
    reader = Reader(root_of(directory))
    return LoadTestView(
        services=read_services(reader, LoadTestViewServices),
        consumers=load_consumers(reader),
        datasets=reader.build(DatasetClasses, "datasets/sources.yaml"),
    )
