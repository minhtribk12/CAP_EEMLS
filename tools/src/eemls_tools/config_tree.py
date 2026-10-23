from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from eemls_consumer import config as consumer_config
from eemls_control_plane import config as control_plane_config
from eemls_control_plane.config import check_the_control_plane_files
from eemls_data_hub import config as data_hub_config
from eemls_inference_service import config as inference_config
from eemls_inference_service.config import check_the_inference_files
from eemls_observation_service import config as observation_service_config

from eemls.config.consumers import ConsumerConfig, load_consumers
from eemls.config.cost import CostConfig
from eemls.config.models import InferenceServicesConfig, ModelRepositoryConfig
from eemls.config.reading import ConfigError as ConfigError
from eemls.config.reading import Reader, root_of, validate
from eemls.config.schema import FrozenMap, FrozenModel
from eemls.ensemble_optimization.approach import ApproachSettings
from eemls.ensemble_optimization.approach_loading import load_approaches
from eemls.metrics_database.schema import check_every_scored_cdm_can_be_computed
from eemls_tools.datasets import DatasetsConfig
from eemls_tools.model_config import ModelsConfig, VariationProfilesConfig, VariationRecipeConfig
from eemls_tools.services import IMAGES, SERVICE_FILES, DeployServicesConfig, ImagesConfig

FILES_READ_BY: Mapping[str, frozenset[str]] = {
    "inference": inference_config.FILES_READ,
    "data_hub": data_hub_config.FILES_READ,
    "consumer": consumer_config.FILES_READ,
    "control_plane": control_plane_config.FILES_READ,
    "observation_service": observation_service_config.FILES_READ,
}

SINGLE_FILES: dict[str, type[FrozenModel]] = {
    "models/model_repository.yaml": ModelRepositoryConfig,
    "models/inference_services.yaml": InferenceServicesConfig,
    "models/variations.yaml": VariationRecipeConfig,
    "models/variation_profiles.yaml": VariationProfilesConfig,
    "datasets/sources.yaml": DatasetsConfig,
    "cost/prices.yaml": CostConfig,
}
MODEL_SOURCE_SECTIONS = {
    "models/model_repository.yaml": "models",
    "models/variations.yaml": "base_models",
}


class ConfigTree(FrozenModel):
    services: DeployServicesConfig
    consumers: FrozenMap[ConsumerConfig]
    approaches: FrozenMap[ApproachSettings]
    models: ModelsConfig
    datasets: DatasetsConfig
    cost: CostConfig
    images: ImagesConfig | None = None


def _check_across_files(config: ConfigTree) -> None:
    check_the_control_plane_files(
        config.services.control_plane,
        config.approaches,
        config.consumers,
        config.models.inference_services,
        config.datasets.dominant_classes,
        config.datasets.classes,
        config.cost.devices,
    )
    check_every_scored_cdm_can_be_computed(config.consumers)
    check_the_inference_files(
        config.models.model_repository,
        config.models.inference_services,
        {name: device.execution_provider for name, device in config.cost.devices.items()},
        config.services.inference_service.providers,
    )
    images = set(config.services.deployment.image_tags)
    for name, device in config.cost.devices.items():
        if device.execution_provider not in images:
            raise ConfigError(
                f"cost/prices.yaml: {name} names execution provider {device.execution_provider}, "
                "which services/deployment.yaml gives no image"
            )


def load_config(directory: Path | str | None = None) -> ConfigTree:
    root = root_of(directory)
    reader = Reader(root)
    services = {name: reader.read_file(f"services/{name}.yaml") for name in SERVICE_FILES}
    bodies = reader.read_with_model_sources(MODEL_SOURCE_SECTIONS)
    single = {
        name: validate(model, bodies[name], name) if name in bodies else reader.build(model, name)
        for name, model in SINGLE_FILES.items()
    }
    config = ConfigTree(
        services=validate(DeployServicesConfig, services, "services/"),
        consumers=load_consumers(reader),
        approaches=load_approaches(reader),
        models=ModelsConfig(
            model_repository=single["models/model_repository.yaml"],
            inference_services=single["models/inference_services.yaml"],
            variations=single["models/variations.yaml"],
            variation_profiles=single["models/variation_profiles.yaml"],
        ),
        datasets=single["datasets/sources.yaml"],
        cost=single["cost/prices.yaml"],
        images=reader.build(ImagesConfig, IMAGES) if (root / IMAGES).is_file() else None,
    )
    reader.refuse_unread_files()
    _check_across_files(config)
    return config
