from __future__ import annotations

from pathlib import Path

from eemls.config.reading import Reader, read_services, root_of
from eemls.config.schema import FrozenModel
from eemls.config.services import DataHubConfig, MonitoringConfig

FILES_READ: frozenset[str] = frozenset(
    {
        "services/data_hub.yaml",
        "services/monitoring.yaml",
    }
)


class DataHubViewServices(FrozenModel):
    data_hub: DataHubConfig
    monitoring: MonitoringConfig


class DataHubView(FrozenModel):
    services: DataHubViewServices


def load_data_hub_config(directory: Path | str | None = None) -> DataHubView:
    reader = Reader(root_of(directory))
    return DataHubView(services=read_services(reader, DataHubViewServices))
