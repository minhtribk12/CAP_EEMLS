from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eemls_tools.config_tree import ConfigTree
from eemls_tools.deployment_plan import (
    BROKER_COMMAND,
    CONFIG_PATH,
    CORE_LABEL,
    DEVICE_LABEL,
    IMAGES_PATH,
    MODEL_SOURCES,
    MODELS_PATH,
    OBSERVATION_SERVICE_INSTANCE,
    SCENARIO_PATH,
    Consumers,
    Deployment,
    Images,
    Mounts,
    RunSettings,
    ServiceReplicas,
    check_the_scenario,
    consumer_arguments,
    consumer_name,
    service_names,
)
from eemls_tools.services import BROKER_IMAGE, DATABASE_IMAGE

PASSWORD = "${EEMLS_DATABASE_PASSWORD}"  # noqa: S105


def _ready(port: int, path: str) -> list[str]:
    probe = (
        "import sys, urllib.request; "
        f"sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:{port}{path}', timeout=2)"
        ".status == 200 else 1)"
    )
    return ["CMD", "python", "-c", probe]


def _swarm_deploy(constraint: str, memory_mb: float, replicas: int = 1) -> dict[str, Any]:
    return {
        "replicas": replicas,
        "placement": {"constraints": [f"node.labels.{constraint} == true"]},
        "resources": {
            "limits": {"memory": f"{round(memory_mb)}M"},
            "reservations": {"memory": f"{round(memory_mb)}M"},
        },
        "restart_policy": {"condition": "on-failure"},
    }


def _swarm_mounts(mounts: Mounts) -> list[str]:
    return [f"{mounts.config}:{CONFIG_PATH}:ro"]


def swarm_stack(
    deployment: Deployment, *, images: Images, mounts: Mounts, run: RunSettings
) -> dict[str, Any]:
    config = deployment.config
    names = service_names(config)
    memory = config.services.deployment.service_memory_mb
    services: dict[str, Any] = {
        names["metrics_database"]: _swarm_database(config, memory["metrics_database"], run),
        names["broker"]: {
            "image": BROKER_IMAGE,
            "command": list(BROKER_COMMAND),
            "networks": ["eemls"],
            "healthcheck": {
                "test": ["CMD", "mosquitto_sub", "-t", "$$SYS/broker/uptime", "-C", "1", "-W", "3"]
            },
            "deploy": _swarm_deploy(CORE_LABEL, memory["broker"]),
        },
        names["data_hub"]: {
            "image": images.of("data-hub"),
            "volumes": [f"{mounts.images}:{IMAGES_PATH}:ro", *_swarm_mounts(mounts)],
            "networks": ["eemls"],
            "healthcheck": {"test": _ready(config.services.data_hub.port, "/readyz")},
            "deploy": _swarm_deploy(CORE_LABEL, memory["data_hub"]),
        },
        names["control_plane"]: _swarm_control_plane(config, images, mounts, run),
        names["observation_service"]: _swarm_observation_service(config, images, mounts, run),
    }
    for copies in deployment.pool:
        services[copies.service_name] = _swarm_inference(config, copies, images, mounts, run)
    consumers = run.consumers()
    if consumers is not None:
        check_the_scenario(mounts, consumers)
        for consumer in sorted(config.consumers):
            services[consumer_name(consumer)] = _swarm_consumer(
                consumer, images, mounts, consumers, memory
            )
    return {
        "version": "3.8",
        "networks": {"eemls": {"driver": "overlay", "attachable": True}},
        "volumes": {run.volume: {}},
        "services": services,
    }


def _swarm_database(config: ConfigTree, memory_mb: float, run: RunSettings) -> dict[str, Any]:
    database = config.services.metrics_database
    return {
        "image": DATABASE_IMAGE,
        "environment": {
            "POSTGRES_USER": database.user,
            "POSTGRES_DB": database.database,
            "POSTGRES_PASSWORD": PASSWORD,
        },
        "volumes": [f"{run.volume}:/var/lib/postgresql/data"],
        "networks": ["eemls"],
        "healthcheck": {"test": ["CMD", "pg_isready", "-U", database.user]},
        "deploy": _swarm_deploy(CORE_LABEL, memory_mb),
    }


def _swarm_control_plane(
    config: ConfigTree, images: Images, mounts: Mounts, run: RunSettings
) -> dict[str, Any]:
    return {
        "image": images.of("control-plane"),
        "environment": {
            "EEMLS_DATABASE_PASSWORD": PASSWORD,
            "EEMLS_RUN_ID": run.run_id,
            "EEMLS_INSTANCE_ID": "control-plane-{{.Task.Slot}}",
            "EEMLS_SEED": str(run.seed),
        },
        "volumes": _swarm_mounts(mounts),
        "networks": ["eemls"],
        "healthcheck": {"test": _ready(config.services.control_plane.port, "/readyz")},
        "deploy": _swarm_deploy(
            CORE_LABEL, config.services.deployment.service_memory_mb["control_plane"]
        ),
    }


def _swarm_observation_service(
    config: ConfigTree, images: Images, mounts: Mounts, run: RunSettings
) -> dict[str, Any]:
    return {
        "image": images.of("observation-service"),
        "environment": {
            "EEMLS_DATABASE_PASSWORD": PASSWORD,
            "EEMLS_RUN_ID": run.run_id,
            "EEMLS_INSTANCE_ID": OBSERVATION_SERVICE_INSTANCE,
        },
        "volumes": _swarm_mounts(mounts),
        "networks": ["eemls"],
        "healthcheck": {"test": _ready(config.services.observation_service.port, "/readyz")},
        "deploy": _swarm_deploy(
            CORE_LABEL, config.services.deployment.service_memory_mb["observation_service"]
        ),
    }


def _swarm_inference(
    config: ConfigTree, copies: ServiceReplicas, images: Images, mounts: Mounts, run: RunSettings
) -> dict[str, Any]:
    image = images.of("inference-service", copies.image_tag)
    return {
        "image": image,
        "environment": {
            "EEMLS_SERVICE_NAME": copies.service_name,
            "EEMLS_RUN_ID": run.run_id,
            "EEMLS_IMAGE": image,
            "EEMLS_INSTANCE_ID": "{{.Task.Name}}",
            "EEMLS_NODE": "{{.Node.Hostname}}",
            "EEMLS_REPLICA": "{{.Task.Slot}}",
        },
        "volumes": [
            *_swarm_mounts(mounts),
            f"{mounts.models}:{MODELS_PATH}:ro",
            f"{mounts.model_sources}:{CONFIG_PATH}/local/{MODEL_SOURCES}:ro",
        ],
        "networks": ["eemls"],
        "healthcheck": {"test": _ready(config.services.inference_service.port, "/readyz")},
        "deploy": _swarm_deploy(
            f"{DEVICE_LABEL}.{copies.device_type}", copies.memory_mb, copies.replicas
        ),
    }


def _swarm_consumer(
    consumer: str,
    images: Images,
    mounts: Mounts,
    consumers: Consumers,
    memory: Mapping[str, float],
) -> dict[str, Any]:
    deploy = _swarm_deploy(CORE_LABEL, memory["consumer"])
    deploy["restart_policy"] = {"condition": "none"}
    return {
        "image": images.of("consumer"),
        "command": consumer_arguments(consumer, consumers),
        "volumes": [*_swarm_mounts(mounts), f"{mounts.scenarios}:{SCENARIO_PATH}:ro"],
        "networks": ["eemls"],
        "deploy": deploy,
    }
