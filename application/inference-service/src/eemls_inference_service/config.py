from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from eemls.config.models import InferenceServicesConfig, ModelRepositoryConfig
from eemls.config.reading import (
    ConfigError,
    Reader,
    check_the_device_types_are_priced,
    read_services,
    root_of,
    validate,
)
from eemls.config.schema import (
    ExtraFieldsIgnored,
    FrozenMap,
    FrozenModel,
)
from eemls.config.services import BrokerConfig, InferenceConfig, MonitoringConfig

FILES_READ: frozenset[str] = frozenset(
    {
        "services/inference_service.yaml",
        "services/monitoring.yaml",
        "services/broker.yaml",
        "models/model_repository.yaml",
        "models/inference_services.yaml",
        "cost/prices.yaml",
    }
)


class DeviceProvider(ExtraFieldsIgnored):
    execution_provider: str


class DeviceProviders(ExtraFieldsIgnored):
    devices: FrozenMap[DeviceProvider]


class InferenceViewServices(FrozenModel):
    inference_service: InferenceConfig
    monitoring: MonitoringConfig
    broker: BrokerConfig


class InferenceViewModels(FrozenModel):
    model_repository: ModelRepositoryConfig
    inference_services: InferenceServicesConfig


class InferenceView(FrozenModel):
    services: InferenceViewServices
    models: InferenceViewModels
    cost: DeviceProviders


def check_the_inference_files(
    model_repository: ModelRepositoryConfig,
    inference_services: InferenceServicesConfig,
    execution_providers: Mapping[str, str],
    providers: Mapping[str, object],
) -> None:
    for name, service in inference_services.services.items():
        if service.model not in model_repository.models:
            raise ConfigError(
                f"models/inference_services.yaml: {name} names model {service.model}, "
                "which the model_repository does not hold"
            )
    check_the_device_types_are_priced(inference_services, execution_providers, "cost/prices.yaml")
    for name, execution_provider in execution_providers.items():
        if execution_provider not in providers:
            raise ConfigError(
                f"cost/prices.yaml: {name} names execution provider {execution_provider}, "
                "which services/inference_service.yaml does not hold"
            )


def load_inference_config(directory: Path | str | None = None) -> InferenceView:
    reader = Reader(root_of(directory))
    repository = "models/model_repository.yaml"
    bodies = reader.read_with_model_sources({repository: "models"})
    config = InferenceView(
        services=read_services(reader, InferenceViewServices),
        models=InferenceViewModels(
            model_repository=validate(ModelRepositoryConfig, bodies[repository], repository),
            inference_services=reader.build(
                InferenceServicesConfig, "models/inference_services.yaml"
            ),
        ),
        cost=reader.build(DeviceProviders, "cost/prices.yaml"),
    )
    check_the_inference_files(
        config.models.model_repository,
        config.models.inference_services,
        {name: device.execution_provider for name, device in config.cost.devices.items()},
        config.services.inference_service.providers,
    )
    return config
