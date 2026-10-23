from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eemls.config.models import Autoscale
from eemls.config.reading import SOURCES_VAR
from eemls_tools.config_tree import FILES_READ_BY, ConfigTree
from eemls_tools.deployment_plan import (
    BROKER_COMMAND,
    CONFIG_PATH,
    CORE_LABEL,
    DEVICE_LABEL,
    IMAGES_PATH,
    LATEST_VERSION,
    MODEL_SOURCES,
    MODELS_PATH,
    OBSERVATION_SERVICE_INSTANCE,
    SCENARIO_PATH,
    Consumers,
    DeployError,
    Deployment,
    Images,
    Mounts,
    RunSettings,
    ServiceReplicas,
    check_the_scenario,
    consumer_arguments,
    consumer_name,
    scenario_path,
    service_names,
)
from eemls_tools.services import BROKER_IMAGE, DATABASE_IMAGE

PASSWORD_HOLDER = "eemls-database"  # noqa: S105
MILLICORES = 1000
NAMESPACE = "eemls"
KUSTOMIZATION = "kustomization.yaml"
CONFIG_FILE = "config.yaml"
CONFIG_MAP = "eemls-config"
SCENARIO_MAP = "eemls-scenario"
SCENARIO_FILE = "scenario.yaml"
CONFIG_MAP_BYTES = 1024 * 1024
RUN_FILE = "run.yaml"
CONTROL_PLANE_INSTANCE = "control-plane-1"

# A pod volume and its container mount.
Volume = tuple[dict[str, Any], dict[str, Any]]
Manifest = dict[str, Any]
KubeFiles = dict[str, list[Manifest]]


@dataclass(frozen=True)
class PodSettings:
    replicas: int | None = 1
    runtime: str | None = None
    spread: bool = False
    volumes: tuple[Volume, ...] = ()
    single: bool = False


def kube_files(
    deployment: Deployment, *, images: Images, mounts: Mounts, run: RunSettings
) -> KubeFiles:
    config = deployment.config
    tree = {service: _kube_tree(mounts.config, service) for service in FILES_READ_BY}
    names = service_names(config)
    memory = config.services.deployment.service_memory_mb
    services = config.services
    core = {CORE_LABEL: "true"}
    files: KubeFiles = {
        RUN_FILE: [
            {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NAMESPACE}},
            _kube_volume_claim(run),
        ],
        f"core/{names['metrics_database']}.yaml": _kube_served(
            names["metrics_database"],
            _kube_database(config, memory["metrics_database"]),
            services.metrics_database.port,
            core,
            PodSettings(
                volumes=(
                    (
                        {"name": "data", "persistentVolumeClaim": {"claimName": run.volume}},
                        {"name": "data", "mountPath": "/var/lib/postgresql/data"},
                    ),
                ),
                single=True,
            ),
        ),
        f"core/{names['broker']}.yaml": _kube_served(
            names["broker"],
            _container("broker", BROKER_IMAGE, memory["broker"], command=BROKER_COMMAND)
            | {"readinessProbe": {"tcpSocket": {"port": services.broker.port}}},
            services.broker.port,
            core,
            PodSettings(),
        ),
        f"core/{names['data_hub']}.yaml": _kube_served(
            names["data_hub"],
            _probed(
                _container("data-hub", images.of("data-hub"), memory["data_hub"]),
                services.data_hub.port,
            ),
            services.data_hub.port,
            core,
            PodSettings(
                volumes=(
                    (
                        {
                            "name": "images",
                            "hostPath": {"path": mounts.images, "type": "Directory"},
                        },
                        {"name": "images", "mountPath": IMAGES_PATH, "readOnly": True},
                    ),
                    *tree["data_hub"],
                )
            ),
        ),
        f"core/{names['control_plane']}.yaml": _kube_served(
            names["control_plane"],
            _kube_control_plane(config, images, run),
            services.control_plane.port,
            core,
            PodSettings(volumes=tree["control_plane"], single=True),
        ),
        f"core/{names['observation_service']}.yaml": _kube_served(
            names["observation_service"],
            _kube_observation_service(config, images, run),
            services.observation_service.port,
            core,
            PodSettings(volumes=tree["observation_service"], single=True),
        ),
    }
    files[CONFIG_FILE] = [_kube_config_map(Path(mounts.config))]
    for copies in deployment.pool:
        files[f"inference/{copies.service_name}.yaml"] = _kube_inference(
            config, copies, images, mounts, tree["inference"], run
        )
    consumers = run.consumers()
    if consumers is not None:
        check_the_scenario(mounts, consumers)
        scenario = scenario_path(mounts, consumers)
        files[SCENARIO_FILE] = [_kube_scenario_map(scenario)]
        held = (*tree["consumer"], _kube_scenario_volume(scenario))
        for one in sorted(config.consumers):
            files[f"consumers/{consumer_name(one)}.yaml"] = [
                _kube_consumer(one, images, consumers, memory, held)
            ]
    if images.version == LATEST_VERSION:
        _always_pulled(files, images)
    files[KUSTOMIZATION] = [
        {
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "namespace": NAMESPACE,
            "resources": sorted(set(files) - {RUN_FILE}),
        }
    ]
    return files


def one_off_pod(config_folder: str, *, name: str, image: str, command: Sequence[str]) -> Manifest:
    items = [{"key": _config_key(path), "path": path} for path in _tree_files(Path(config_folder))]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _named(name),
        "spec": {
            "nodeSelector": {CORE_LABEL: "true"},
            "restartPolicy": "Never",
            "volumes": [{"name": "config", "configMap": {"name": CONFIG_MAP, "items": items}}],
            "containers": [
                {
                    "name": name,
                    "image": image,
                    "imagePullPolicy": "Always",
                    "command": list(command),
                    "env": [{"name": "EEMLS_DATABASE_PASSWORD", **_password()}],
                    "volumeMounts": [
                        {"name": "config", "mountPath": CONFIG_PATH, "readOnly": True}
                    ],
                }
            ],
        },
    }


def _always_pulled(files: Mapping[str, list[Manifest]], images: Images) -> None:
    for held in files.values():
        for one in held:
            if one["kind"] not in ("Deployment", "Job"):
                continue
            for container in one["spec"]["template"]["spec"]["containers"]:
                if container["image"].startswith(f"{images.registry}/eemls-"):
                    container["imagePullPolicy"] = "Always"


def _kube_tree(folder: str, service: str) -> tuple[Volume, ...]:
    read = FILES_READ_BY[service]
    items = [
        {"key": _config_key(path), "path": path}
        for path in _tree_files(Path(folder))
        if any(path == name or path.startswith(f"{name}/") for name in read)
    ]
    return (
        (
            {"name": "config", "configMap": {"name": CONFIG_MAP, "items": items}},
            {"name": "config", "mountPath": CONFIG_PATH, "readOnly": True},
        ),
    )


def _kube_config_map(folder: Path) -> Manifest:
    data = {
        _config_key(path): (folder / path).read_text(encoding="utf-8")
        for path in _tree_files(folder)
    }
    size = sum(len(text.encode("utf-8")) for text in data.values())
    if size > CONFIG_MAP_BYTES:
        raise DeployError(
            f"the tree at {folder} holds {size} bytes, and a ConfigMap holds at most 1 MiB"
        )
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": _named(CONFIG_MAP), "data": data}


def _kube_scenario_map(path: Path) -> Manifest:
    text = path.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > CONFIG_MAP_BYTES:
        raise DeployError(f"{path} is larger than the 1 MiB a ConfigMap holds")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": _named(SCENARIO_MAP),
        "data": {path.name: text},
    }


def _kube_scenario_volume(path: Path) -> Volume:
    return (
        {
            "name": "scenario",
            "configMap": {"name": SCENARIO_MAP, "items": [{"key": path.name, "path": path.name}]},
        },
        {"name": "scenario", "mountPath": SCENARIO_PATH, "readOnly": True},
    )


def _tree_files(folder: Path) -> list[str]:
    return sorted(
        str(path.relative_to(folder))
        for path in folder.rglob("*.yaml")
        if path.relative_to(folder).parts[0] != "local"
    )


def _config_key(path: str) -> str:
    return path.replace("/", "__")


def _container(
    name: str, image: str, memory_mb: float, *, command: Sequence[str] = ()
) -> dict[str, Any]:
    held: dict[str, Any] = {
        "name": name,
        "image": image,
        "resources": {"limits": {"memory": f"{round(memory_mb)}Mi"}},
    }
    if command:
        held["command"] = list(command)
    return held


def _probed(container: dict[str, Any], port: int) -> dict[str, Any]:
    return container | {
        "ports": [{"containerPort": port}],
        "readinessProbe": {"httpGet": {"path": "/readyz", "port": port}},
        "livenessProbe": {"httpGet": {"path": "/healthz", "port": port}},
    }


def _named(name: str, **labels: str) -> dict[str, Any]:
    return {"name": name, "namespace": NAMESPACE} | ({"labels": labels} if labels else {})


def _kube_volume_claim(run: RunSettings) -> Manifest:
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": _named(run.volume),
        "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "10Gi"}}},
    }


def _password() -> dict[str, Any]:
    return {"valueFrom": {"secretKeyRef": {"name": PASSWORD_HOLDER, "key": "password"}}}


def _kube_database(config: ConfigTree, memory_mb: float) -> dict[str, Any]:
    database = config.services.metrics_database
    container = _container("postgres", DATABASE_IMAGE, memory_mb)
    container["env"] = [
        {"name": "POSTGRES_USER", "value": database.user},
        {"name": "POSTGRES_DB", "value": database.database},
        {"name": "POSTGRES_PASSWORD", **_password()},
        {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"},
    ]
    container["readinessProbe"] = {"exec": {"command": ["pg_isready", "-U", database.user]}}
    return container


def _kube_control_plane(config: ConfigTree, images: Images, run: RunSettings) -> dict[str, Any]:
    port = config.services.control_plane.port
    memory = config.services.deployment.service_memory_mb["control_plane"]
    container = _probed(_container("control-plane", images.of("control-plane"), memory), port)
    container["env"] = [
        {"name": "EEMLS_DATABASE_PASSWORD", **_password()},
        {"name": "EEMLS_RUN_ID", "value": run.run_id},
        {"name": "EEMLS_INSTANCE_ID", "value": CONTROL_PLANE_INSTANCE},
        {"name": "EEMLS_SEED", "value": str(run.seed)},
    ]
    return container


def _kube_observation_service(
    config: ConfigTree, images: Images, run: RunSettings
) -> dict[str, Any]:
    port = config.services.observation_service.port
    memory = config.services.deployment.service_memory_mb["observation_service"]
    container = _probed(
        _container("observation-service", images.of("observation-service"), memory), port
    )
    container["env"] = [
        {"name": "EEMLS_DATABASE_PASSWORD", **_password()},
        {"name": "EEMLS_RUN_ID", "value": run.run_id},
        {"name": "EEMLS_INSTANCE_ID", "value": OBSERVATION_SERVICE_INSTANCE},
    ]
    return container


def _kube_inference(
    config: ConfigTree,
    copies: ServiceReplicas,
    images: Images,
    mounts: Mounts,
    tree: tuple[Volume, ...],
    run: RunSettings,
) -> list[Manifest]:
    port = config.services.inference_service.port
    container = _probed(
        _container("inference", images.of("inference-service", copies.image_tag), copies.memory_mb),
        port,
    )
    cores = config.services.deployment.inference_cpu_cores
    container["resources"]["requests"] = {"cpu": f"{round(cores * MILLICORES)}m"}
    container["env"] = [
        {"name": "EEMLS_SERVICE_NAME", "value": copies.service_name},
        {"name": "EEMLS_RUN_ID", "value": run.run_id},
        {"name": "EEMLS_IMAGE", "value": container["image"]},
        {"name": SOURCES_VAR, "value": f"{MODELS_PATH}/{MODEL_SOURCES}"},
        {"name": "EEMLS_INSTANCE_ID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        {"name": "EEMLS_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
    ]
    autoscale = config.models.inference_services.autoscale
    provider = config.cost.devices[copies.device_type].execution_provider
    served = _kube_served(
        copies.service_name,
        container,
        port,
        {f"{DEVICE_LABEL}.{copies.device_type}": "true"},
        PodSettings(
            replicas=None if autoscale.enabled else copies.replicas,
            runtime=config.services.deployment.runtime_classes.get(provider),
            spread=True,
            volumes=(
                *tree,
                (
                    {"name": "models", "hostPath": {"path": mounts.models, "type": "Directory"}},
                    {"name": "models", "mountPath": MODELS_PATH, "readOnly": True},
                ),
            ),
        ),
    )
    if autoscale.enabled:
        served.append(_kube_autoscaler(copies, autoscale))
    return served


def _kube_autoscaler(copies: ServiceReplicas, autoscale: Autoscale) -> Manifest:
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": _named(copies.service_name),
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": copies.service_name,
            },
            "minReplicas": copies.replicas,
            "maxReplicas": autoscale.max_replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": autoscale.cpu_percent,
                        },
                    },
                }
            ],
            "behavior": {"scaleDown": {"stabilizationWindowSeconds": autoscale.cooldown_s}},
        },
    }


def _kube_served(
    name: str,
    container: dict[str, Any],
    port: int,
    selector: Mapping[str, str],
    options: PodSettings,
) -> list[Manifest]:
    labels = {"app": name}
    pod: dict[str, Any] = {"nodeSelector": dict(selector), "containers": [container]}
    if options.runtime is not None:
        pod["runtimeClassName"] = options.runtime
    if options.spread:
        pod["topologySpreadConstraints"] = [
            {
                "maxSkew": 1,
                "topologyKey": "kubernetes.io/hostname",
                "whenUnsatisfiable": "ScheduleAnyway",
                "labelSelector": {"matchLabels": labels},
            }
        ]
    if options.volumes:
        pod["volumes"] = [volume for volume, _ in options.volumes]
        container["volumeMounts"] = [mount for _, mount in options.volumes]
    return [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": _named(name, **labels),
            "spec": ({"replicas": options.replicas} if options.replicas is not None else {})
            | {
                "selector": {"matchLabels": labels},
                "template": {"metadata": {"labels": labels}, "spec": pod},
            }
            | ({"strategy": {"type": "Recreate"}} if options.single else {}),
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": _named(name),
            "spec": {"selector": labels, "ports": [{"port": port, "targetPort": port}]},
        },
    ]


def _kube_consumer(
    consumer: str,
    images: Images,
    consumers: Consumers,
    memory: Mapping[str, float],
    tree: tuple[Volume, ...],
) -> Manifest:
    container = _container("consumer", images.of("consumer"), memory["consumer"])
    container["args"] = consumer_arguments(consumer, consumers)
    pod: dict[str, Any] = {
        "restartPolicy": "Never",
        "nodeSelector": {CORE_LABEL: "true"},
        "containers": [container],
    }
    if tree:
        pod["volumes"] = [volume for volume, _ in tree]
        container["volumeMounts"] = [mount for _, mount in tree]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": _named(consumer_name(consumer)),
        "spec": {
            "backoffLimit": 0,
            "template": {"spec": pod},
        },
    }
