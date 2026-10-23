from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from eemls.domain.errors import EemlsError
from eemls_tools.config_tree import ConfigTree

DEVICE_LABEL = "eemls.device"
CORE_LABEL = "eemls.role.core"
CONFIG_PATH = "/opt/eemls/config"
SCENARIO_PATH = "/opt/eemls/scenarios"
IMAGES_PATH = "/var/lib/eemls/images"
MODELS_PATH = "/opt/eemls/models"
MODEL_SOURCES = "models.yaml"
LATEST_VERSION = "latest"
OBSERVATION_SERVICE_INSTANCE = "observation-service-1"
MB_PER_GB = 1024
BROKER_COMMAND = ("mosquitto", "-c", "/mosquitto-no-auth.conf")
VOLUME_NAME = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?")


class DeployError(EemlsError): ...


@dataclass(frozen=True)
class Images:
    registry: str
    version: str

    def of(self, name: str, tag: str | None = None) -> str:
        version = f"{tag}-{self.version}" if tag else self.version
        return f"{self.registry}/eemls-{name}:{version}"


@dataclass(frozen=True)
class Mounts:
    images: str
    config: str
    models: str
    scenarios: str | None = None

    @property
    def model_sources(self) -> str:
        return f"{self.models}/{MODEL_SOURCES}"


class Consumers(NamedTuple):
    scenario: str
    duration_s: float
    seed: int


@dataclass(frozen=True)
class RunSettings:
    run_id: str
    scenario: str | None = None
    duration_s: float | None = None
    seed: int = field(kw_only=True)

    def __post_init__(self) -> None:
        if not VOLUME_NAME.fullmatch(self.run_id):
            raise ValueError(
                f"run id {self.run_id!r} must be lowercase letters, digits and hyphens, "
                "since it names the run's database volume"
            )
        if (self.scenario is None) != (self.duration_s is None):
            raise ValueError("consumers need both a scenario and a length to run for")

    @property
    def volume(self) -> str:
        return f"metrics-database-{self.run_id}"

    def consumers(self) -> Consumers | None:
        if self.scenario is None or self.duration_s is None:
            return None
        return Consumers(scenario=self.scenario, duration_s=self.duration_s, seed=self.seed)


@dataclass(frozen=True)
class ServiceReplicas:
    service_name: str
    device_type: str
    image_tag: str
    replicas: int
    memory_mb: int


@dataclass(frozen=True)
class Deployment:
    config: ConfigTree
    pool: tuple[ServiceReplicas, ...]


def plan_deployment(config: ConfigTree, throughput_rps: Mapping[str, float]) -> Deployment:
    demand = sum(consumer.contract.request_rate_per_s for consumer in config.consumers.values())
    pool = tuple(
        _copies(config, name, throughput_rps, demand)
        for name in sorted(config.models.inference_services.services)
    )
    _devices_hold(config, pool)
    return Deployment(config=config, pool=pool)


def _copies(
    config: ConfigTree, name: str, throughput_rps: Mapping[str, float], demand: float
) -> ServiceReplicas:
    service = config.models.inference_services.services[name]
    device = config.cost.devices[service.device_type]
    if name not in throughput_rps:
        raise DeployError(f"{name} has no measured throughput, so its copies cannot be sized")
    carried = service.replicas * throughput_rps[name]
    if carried < demand:
        raise DeployError(
            f"{name} carries {service.replicas} x {throughput_rps[name]:.1f} = {carried:.1f} "
            f"requests/s, below the {demand:.1f} requests/s its consumers ask for together"
        )
    model = config.models.model_repository.models[service.model]
    needed = round(
        config.services.deployment.copy_memory_mb(
            model_memory_mb=model.memory_mb, execution_provider=device.execution_provider
        )
    )
    if needed > device.memory_gb * MB_PER_GB:
        raise DeployError(
            f"{name} needs {needed} MB a copy, and a {service.device_type} holds "
            f"{device.memory_gb * MB_PER_GB} MB"
        )
    return ServiceReplicas(
        service_name=name,
        device_type=service.device_type,
        image_tag=config.services.deployment.image_tags[device.execution_provider],
        replicas=service.replicas,
        memory_mb=needed,
    )


def _devices_hold(config: ConfigTree, pool: tuple[ServiceReplicas, ...]) -> None:
    for device_type, device in config.cost.devices.items():
        placed = sorted(
            (
                copy_mb
                for one in pool
                if one.device_type == device_type
                for copy_mb in [one.memory_mb] * one.replicas
            ),
            reverse=True,
        )
        free = [device.memory_gb * MB_PER_GB] * device.count
        for copy_mb in placed:
            roomiest = max(range(len(free)), key=lambda index: free[index])
            if free[roomiest] < copy_mb:
                raise DeployError(
                    f"the {len(placed)} copies on {device_type} do not fit, each whole on one "
                    f"of its {device.count} device(s) of {device.memory_gb * MB_PER_GB} MB"
                )
            free[roomiest] -= copy_mb


def models_on(config: ConfigTree, device_type: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                service.model
                for service in config.models.inference_services.services.values()
                if service.device_type == device_type
            }
        )
    )


def _host(url: object) -> str:
    return str(urlparse(str(url)).hostname)


def service_names(config: ConfigTree) -> dict[str, str]:
    services = config.services
    return {
        "metrics_database": services.metrics_database.host,
        "broker": services.broker.host,
        "data_hub": _host(services.inference_service.data_hub_url),
        "control_plane": _host(services.consumer.control_plane_url),
        "observation_service": _host(services.observation_service.url),
    }


def node_labels(config: ConfigTree, device_types: tuple[str, ...], *, core: bool) -> list[str]:
    unknown = [name for name in device_types if name not in config.cost.devices]
    if unknown:
        raise DeployError(f"{', '.join(unknown)} is not a device type of config/cost/prices.yaml")
    labels = [f"{DEVICE_LABEL}.{name}" for name in device_types]
    return [*labels, CORE_LABEL] if core else labels


def consumer_name(consumer: str) -> str:
    return f"consumer-{consumer.replace('_', '-')}"


def consumer_arguments(consumer: str, consumers: Consumers) -> list[str]:
    return [
        "--consumer",
        consumer,
        "--scenario",
        f"{SCENARIO_PATH}/{consumers.scenario}.yaml",
        "--duration-s",
        f"{consumers.duration_s:g}",
        "--client-id",
        consumer_name(consumer),
        "--seed",
        str(consumers.seed),
    ]


def scenario_path(mounts: Mounts, consumers: Consumers) -> Path:
    if mounts.scenarios is None:
        raise DeployError("consumers need the scenarios folder their scenario is in")
    return Path(mounts.scenarios) / f"{consumers.scenario}.yaml"


def check_the_scenario(mounts: Mounts, consumers: Consumers) -> None:
    if not scenario_path(mounts, consumers).is_file():
        raise DeployError(f"no scenario {consumers.scenario} in {mounts.scenarios}")
