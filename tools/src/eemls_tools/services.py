from __future__ import annotations

from pathlib import Path

from pydantic import model_validator

from eemls.config.reading import Reader, root_of
from eemls.config.schema import FrozenMap, FrozenModel, PositiveNumber, Source
from eemls.config.services import (
    BrokerConfig,
    ConsumerServiceConfig,
    ControlPlaneConfig,
    DataHubConfig,
    InferenceConfig,
    MetricsDatabaseConfig,
    MonitoringConfig,
    ObservationServiceConfig,
)

IMAGES = "local/images.yaml"
DATABASE_IMAGE = "postgres:16.4"
BROKER_IMAGE = "eclipse-mosquitto:2.0.20"


class ImagesConfig(FrozenModel):
    registry: Source
    version: Source


PLACED_SERVICES = (
    "control_plane",
    "observation_service",
    "data_hub",
    "consumer",
    "broker",
    "metrics_database",
)


class DeploymentConfig(FrozenModel):
    image_tags: FrozenMap[str]
    jetpack_release: str
    runtime_memory_mb: FrozenMap[PositiveNumber]
    model_memory_factor: PositiveNumber
    service_memory_mb: FrozenMap[PositiveNumber]
    inference_cpu_cores: PositiveNumber
    runtime_classes: FrozenMap[str]

    @model_validator(mode="after")
    def _every_service_has_its_memory(self) -> DeploymentConfig:
        missing = sorted(set(PLACED_SERVICES) - set(self.service_memory_mb))
        if missing:
            raise ValueError(f"service_memory_mb names no memory for {missing}")
        missing = sorted(set(self.image_tags) - set(self.runtime_memory_mb))
        if missing:
            raise ValueError(f"runtime_memory_mb names no memory for {missing}")
        return self

    def copy_memory_mb(self, *, model_memory_mb: float, execution_provider: str) -> float:
        return (
            self.runtime_memory_mb[execution_provider] + self.model_memory_factor * model_memory_mb
        )


class DeployServicesConfig(FrozenModel):
    control_plane: ControlPlaneConfig
    observation_service: ObservationServiceConfig
    inference_service: InferenceConfig
    data_hub: DataHubConfig
    consumer: ConsumerServiceConfig
    broker: BrokerConfig
    metrics_database: MetricsDatabaseConfig
    monitoring: MonitoringConfig
    deployment: DeploymentConfig

    @model_validator(mode="after")
    def _both_sides_name_one_result_topic(self) -> DeployServicesConfig:
        if self.consumer.reply_topic_prefix != self.control_plane.result_topic_prefix:
            raise ValueError(
                f"the consumer listens under {self.consumer.reply_topic_prefix} and the "
                f"control plane publishes under {self.control_plane.result_topic_prefix}"
            )
        return self

    @model_validator(mode="after")
    def _the_observation_service_is_reached_on_its_port(self) -> DeployServicesConfig:
        held = self.observation_service
        if held.url.port != held.port:
            raise ValueError(f"observation_service url {held.url} is not on port {held.port}")
        return self

    @model_validator(mode="after")
    def _two_services_state_one_value(self) -> DeployServicesConfig:
        for own, other, key in (
            ("consumer", "control_plane", "data_hub_url"),
            ("consumer", "control_plane", "decision_interval_queries"),
            ("inference_service", "data_hub", "max_image_bytes"),
        ):
            stated = getattr(getattr(self, own), key)
            expected = getattr(getattr(self, other), key)
            if stated != expected:
                raise ValueError(f"{own} {key} {stated} is not the {other} {key} {expected}")
        return self


SERVICE_FILES: tuple[str, ...] = tuple(DeployServicesConfig.model_fields)


def load_images_config(directory: Path | str | None = None) -> ImagesConfig:
    return Reader(root_of(directory)).build(ImagesConfig, IMAGES)
