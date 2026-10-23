from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import FILES_READ_BY, ConfigTree, load_config
from eemls_tools.deployment_plan import (
    DeployError,
    Images,
    Mounts,
    RunSettings,
    models_on,
    plan_deployment,
)
from eemls_tools.kube_files import kube_files
from eemls_tools.services import BROKER_IMAGE, DATABASE_IMAGE
from eemls_tools.swarm_stack import swarm_stack
from support.deployment import of_kind
from support.repository import ROOT

TREE = ROOT / "config"
SCENARIOS = ROOT / "experiments" / "scenarios"
IMAGES = Images(registry="registry.local:5000", version="1.0")
MOUNTS = Mounts(
    images="/srv/eemls/images",
    config=str(TREE),
    models="/srv/eemls/models",
    scenarios=str(SCENARIOS),
)
FAST = 100.0
KUSTOMIZATION = "kustomization.yaml"


def config() -> ConfigTree:
    held = load_config(TREE)
    pool = held.models.inference_services
    fixed = pool.model_copy(
        update={"autoscale": pool.autoscale.model_copy(update={"enabled": False})}
    )
    return held.model_copy(
        update={"models": held.models.model_copy(update={"inference_services": fixed})}
    )


def throughput(rps: float = FAST, **changes: float) -> dict[str, float]:
    pool = config().models.inference_services.services
    return dict.fromkeys(pool, rps) | changes


def stack(**changes: Any) -> dict[str, Any]:
    run: RunSettings = changes.pop("run", RunSettings(run_id="unnamed", seed=0))
    deployment = plan_deployment(config(), throughput(**changes))
    return swarm_stack(deployment, images=IMAGES, mounts=MOUNTS, run=run)


def files(**changes: Any) -> dict[str, list[dict[str, Any]]]:
    run: RunSettings = changes.pop("run", RunSettings(run_id="unnamed", seed=0))
    deployment = plan_deployment(config(), throughput(**changes))
    return kube_files(deployment, images=IMAGES, mounts=MOUNTS, run=run)


def manifests(**changes: Any) -> list[dict[str, Any]]:
    return [one for path, held in files(**changes).items() if path != KUSTOMIZATION for one in held]


def device_of(service: str) -> str:
    return config().models.inference_services.services[service].device_type


def tag_of(service: str) -> str:
    settings = config()
    provider = settings.cost.devices[device_of(service)].execution_provider
    return settings.services.deployment.image_tags[provider]


POOL = sorted(load_config(TREE).models.inference_services.services)


class TestEveryServiceOfThePool:
    @pytest.mark.parametrize("service", POOL)
    def test_it_is_a_swarm_service_with_the_copies_the_map_asks_for(self, service: str) -> None:
        held = stack()["services"][service]
        replicas = config().models.inference_services.services[service].replicas

        assert held["deploy"]["replicas"] == replicas
        assert held["environment"]["EEMLS_SERVICE_NAME"] == service

    @pytest.mark.parametrize("service", POOL)
    def test_it_is_a_kube_deployment_and_a_service_the_control_plane_can_call(
        self, service: str
    ) -> None:
        documents = manifests()
        port = config().services.inference_service.port

        replicas = config().models.inference_services.services[service].replicas

        assert of_kind(documents, "Deployment")[service]["spec"]["replicas"] == replicas
        assert of_kind(documents, "Service")[service]["spec"]["ports"][0]["port"] == port

    @pytest.mark.parametrize("service", POOL)
    def test_it_runs_the_image_of_its_device_class(self, service: str) -> None:
        image = f"{IMAGES.registry}/eemls-inference-service:{tag_of(service)}-{IMAGES.version}"
        container = of_kind(manifests(), "Deployment")[service]["spec"]["template"]["spec"][
            "containers"
        ][0]

        assert stack()["services"][service]["image"] == image
        assert container["image"] == image

    @pytest.mark.parametrize("service", POOL)
    def test_it_is_placed_only_on_a_node_of_its_device_type(self, service: str) -> None:
        label = f"eemls.device.{device_of(service)}"
        pod = of_kind(manifests(), "Deployment")[service]["spec"]["template"]["spec"]

        assert stack()["services"][service]["deploy"]["placement"]["constraints"] == [
            f"node.labels.{label} == true"
        ]
        assert pod["nodeSelector"] == {label: "true"}

    @pytest.mark.parametrize("service", POOL)
    def test_it_is_ready_only_once_it_answers_readyz(self, service: str) -> None:
        container = of_kind(manifests(), "Deployment")[service]["spec"]["template"]["spec"][
            "containers"
        ][0]

        assert "/readyz" in " ".join(stack()["services"][service]["healthcheck"]["test"])
        assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"

    @pytest.mark.parametrize("service", POOL)
    def test_its_memory_limit_is_what_one_copy_needs(self, service: str) -> None:
        settings = config()
        placed = settings.models.inference_services.services[service]
        model = settings.models.model_repository.models[placed.model]
        needed = settings.services.deployment.copy_memory_mb(
            model_memory_mb=model.memory_mb,
            execution_provider=settings.cost.devices[placed.device_type].execution_provider,
        )

        held = stack()["services"][service]["deploy"]["resources"]["limits"]["memory"]

        assert held == f"{round(needed)}M"


class TestTheServicesAroundThePool:
    def test_each_is_named_as_the_configuration_calls_it(self) -> None:
        held = set(stack()["services"])

        assert {"postgres", "broker", "data-hub", "control-plane", "observation-service"} <= held

    def test_they_run_on_the_core_nodes(self) -> None:
        for name in ("postgres", "broker", "data-hub", "control-plane", "observation-service"):
            assert stack()["services"][name]["deploy"]["placement"]["constraints"] == [
                "node.labels.eemls.role.core == true"
            ]

    def test_the_data_hub_reads_the_prepared_images_read_only(self) -> None:
        assert (
            f"{MOUNTS.images}:/var/lib/eemls/images:ro"
            in stack()["services"]["data-hub"]["volumes"]
        )

    def test_the_password_comes_from_the_environment_and_is_never_written(self) -> None:
        written = yaml.safe_dump(stack()) + yaml.safe_dump_all(manifests())

        assert "${EEMLS_DATABASE_PASSWORD}" in written
        assert "secretKeyRef" in written

    def test_the_control_plane_stores_under_the_run_id_with_or_without_consumers(self) -> None:
        run = RunSettings(run_id="one-host", seed=0)

        held = stack(run=run)["services"]["control-plane"]["environment"]["EEMLS_RUN_ID"]

        assert held == "one-host"
        assert not [name for name in stack(run=run)["services"] if name.startswith("consumer-")]

    def test_each_run_keeps_its_rows_in_a_volume_of_its_own(self) -> None:
        one = stack(run=RunSettings(run_id="first", seed=0))
        two = stack(run=RunSettings(run_id="second", seed=0))

        assert set(one["volumes"]) == {"metrics-database-first"}
        assert set(two["volumes"]) == {"metrics-database-second"}
        assert one["services"]["postgres"]["volumes"] == [
            "metrics-database-first:/var/lib/postgresql/data"
        ]

    def test_a_run_id_that_cannot_name_a_volume_is_refused(self) -> None:
        with pytest.raises(ValueError, match="run id"):
            RunSettings(run_id="one host/2", seed=0)

    def test_a_run_without_a_seed_is_refused(self) -> None:
        with pytest.raises(TypeError, match="seed"):
            RunSettings(run_id="r1")  # type: ignore[call-arg]

    def test_a_scenario_without_a_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="scenario and a length"):
            RunSettings(run_id="r1", scenario="smoke", seed=0)

    def test_no_consumer_runs_unless_a_scenario_is_given(self) -> None:
        assert not [name for name in stack()["services"] if name.startswith("consumer-")]
        assert of_kind(manifests(), "Job") == {}

    def test_a_scenario_runs_one_service_per_consumer(self) -> None:
        run = RunSettings(run_id="r1", scenario="smoke", duration_s=60.0, seed=0)

        consumers = sorted(
            name for name in stack(run=run)["services"] if name.startswith("consumer-")
        )

        assert consumers == [
            f"consumer-{one.replace('_', '-')}" for one in sorted(config().consumers)
        ]
        assert sorted(of_kind(manifests(run=run), "Job")) == consumers

    def test_every_consumer_of_a_run_draws_from_the_run_s_seed(self) -> None:
        run = RunSettings(run_id="r1", scenario="smoke", duration_s=60.0, seed=3)

        swarm = stack(run=run)["services"]["consumer-c1-navigation"]["command"]
        kube = of_kind(manifests(run=run), "Job")["consumer-c1-navigation"]["spec"]["template"][
            "spec"
        ]["containers"][0]["args"]

        for arguments in (swarm, kube):
            assert arguments[arguments.index("--seed") + 1] == "3"

    def test_the_control_plane_draws_from_the_run_s_seed(self) -> None:
        run = RunSettings(run_id="r1", seed=3)

        swarm = stack(run=run)["services"]["control-plane"]["environment"]
        pod = of_kind(manifests(run=run), "Deployment")["control-plane"]["spec"]["template"]
        kube = {one["name"]: one for one in pod["spec"]["containers"][0]["env"]}

        assert swarm["EEMLS_SEED"] == "3"
        assert kube["EEMLS_SEED"] == {"name": "EEMLS_SEED", "value": "3"}

    def test_the_same_map_writes_the_same_files(self) -> None:
        assert yaml.safe_dump(stack()) == yaml.safe_dump(stack())
        assert yaml.safe_dump_all(manifests()) == yaml.safe_dump_all(manifests())


class TestTheObservationService:
    def test_swarm_runs_one_copy_on_a_core_node_that_stores_under_the_run_id(self) -> None:
        held = stack(run=RunSettings(run_id="r1", seed=0))["services"]["observation-service"]
        memory = config().services.deployment.service_memory_mb["observation_service"]

        assert held["image"] == f"{IMAGES.registry}/eemls-observation-service:{IMAGES.version}"
        assert held["environment"] == {
            "EEMLS_DATABASE_PASSWORD": "${EEMLS_DATABASE_PASSWORD}",
            "EEMLS_RUN_ID": "r1",
            "EEMLS_INSTANCE_ID": "observation-service-1",
        }
        assert held["volumes"] == [f"{TREE}:/opt/eemls/config:ro"]
        assert held["deploy"]["replicas"] == 1
        assert held["deploy"]["resources"]["limits"]["memory"] == f"{round(memory)}M"
        assert "http://127.0.0.1:8083/readyz" in " ".join(held["healthcheck"]["test"])

    def test_kube_runs_one_copy_probed_on_its_port(self) -> None:
        documents = manifests(run=RunSettings(run_id="r1", seed=0))
        deployment = of_kind(documents, "Deployment")["observation-service"]
        pod = deployment["spec"]["template"]["spec"]
        [container] = pod["containers"]
        env = {one["name"]: one for one in container["env"]}

        assert deployment["spec"]["replicas"] == 1
        assert pod["nodeSelector"] == {"eemls.role.core": "true"}
        assert container["ports"] == [{"containerPort": 8083}]
        assert container["readinessProbe"]["httpGet"] == {"path": "/readyz", "port": 8083}
        assert container["livenessProbe"]["httpGet"] == {"path": "/healthz", "port": 8083}
        assert env["EEMLS_RUN_ID"] == {"name": "EEMLS_RUN_ID", "value": "r1"}
        assert env["EEMLS_INSTANCE_ID"] == {
            "name": "EEMLS_INSTANCE_ID",
            "value": "observation-service-1",
        }
        assert "secretKeyRef" in env["EEMLS_DATABASE_PASSWORD"]["valueFrom"]
        assert of_kind(documents, "Service")["observation-service"]["spec"]["ports"] == [
            {"port": 8083, "targetPort": 8083}
        ]


class TestWhatAPlatformIsToldToReserve:
    @pytest.mark.parametrize("service", POOL)
    def test_swarm_reserves_the_memory_a_copy_needs_so_it_places_it_where_it_fits(
        self, service: str
    ) -> None:
        resources = stack()["services"][service]["deploy"]["resources"]

        assert resources["reservations"]["memory"] == resources["limits"]["memory"]

    @pytest.mark.parametrize("name", ["control-plane", "observation-service", "postgres"])
    def test_kube_replaces_a_single_instance_rather_than_running_two(self, name: str) -> None:
        assert of_kind(manifests(), "Deployment")[name]["spec"]["strategy"] == {"type": "Recreate"}

    def test_the_broker_has_a_health_check_on_both_targets(self) -> None:
        port = config().services.broker.port
        broker = of_kind(manifests(), "Deployment")["broker"]["spec"]["template"]["spec"]

        assert "healthcheck" in stack()["services"]["broker"]
        assert broker["containers"][0]["readinessProbe"] == {"tcpSocket": {"port": port}}


def read_by(service: str) -> set[str]:
    found: set[str] = set()
    for name in FILES_READ_BY[service]:
        held = TREE / name
        paths = sorted(held.rglob("*.yaml")) if held.is_dir() else [held]
        found |= {str(path.relative_to(TREE)) for path in paths}
    return found


class TestTheMountedConfigTree:
    def mounted(self) -> dict[str, list[dict[str, Any]]]:
        deployment = plan_deployment(config(), throughput())
        return kube_files(
            deployment, images=IMAGES, mounts=MOUNTS, run=RunSettings("r1", "smoke", 60.0, seed=0)
        )

    def test_the_config_map_holds_every_file_of_the_tree_but_the_local_ones(self) -> None:
        [held] = self.mounted()["config.yaml"]
        committed = {
            str(path.relative_to(TREE)): path.read_text(encoding="utf-8")
            for path in TREE.rglob("*.yaml")
            if path.relative_to(TREE).parts[0] != "local"
        }

        assert held["kind"] == "ConfigMap"
        assert held["metadata"] == {"name": "eemls-config", "namespace": "eemls"}
        assert {key.replace("__", "/"): text for key, text in held["data"].items()} == committed

    def test_every_service_of_ours_reads_only_its_own_files_at_the_tree_s_path(self) -> None:
        written = self.mounted()
        ours = {
            "control-plane": "control_plane",
            "observation-service": "observation_service",
            "data-hub": "data_hub",
        }
        ours |= dict.fromkeys(POOL, "inference")
        ours |= {f"consumer-{one.replace('_', '-')}": "consumer" for one in config().consumers}
        pods = {
            one["metadata"]["name"]: one["spec"]["template"]["spec"]
            for path, held in written.items()
            if path != KUSTOMIZATION
            for one in held
            if one["kind"] in ("Deployment", "Job")
        }

        for name, pod in pods.items():
            volumes = {one["name"]: one for one in pod.get("volumes", [])}
            mounts = {one["name"]: one for one in pod["containers"][0].get("volumeMounts", [])}
            if name in ours:
                items = volumes["config"]["configMap"]["items"]
                assert {one["path"] for one in items} == read_by(ours[name]), name
                assert all(one["key"] == one["path"].replace("/", "__") for one in items), name
                assert mounts["config"] == {
                    "name": "config",
                    "mountPath": "/opt/eemls/config",
                    "readOnly": True,
                }, name
            else:
                assert "config" not in volumes, name

    def test_a_deployment_names_the_config_folder_it_mounts(self) -> None:
        with pytest.raises(TypeError, match="config"):
            Mounts(images="/srv/eemls/images", models="/srv/eemls/models")  # type: ignore[call-arg]

    def test_a_tree_kubernetes_cannot_hold_is_refused(self, tmp_path: Path) -> None:
        tree = committed_tree(tmp_path / "config")
        (tree / "big.yaml").write_text("x" * (1024 * 1024), encoding="utf-8")
        deployment = plan_deployment(config(), throughput())
        mounts = Mounts(images="/srv/eemls/images", config=str(tree), models="/srv/eemls/models")

        with pytest.raises(DeployError, match="ConfigMap holds at most 1 MiB"):
            kube_files(
                deployment, images=IMAGES, mounts=mounts, run=RunSettings(run_id="r1", seed=0)
            )


CONSUMERS_RUN = RunSettings(run_id="r1", scenario="smoke", duration_s=60.0, seed=0)


def ours_in_the_stack() -> dict[str, dict[str, Any]]:
    services = stack(run=CONSUMERS_RUN)["services"]
    return {name: body for name, body in services.items() if name not in ("postgres", "broker")}


class TestEveryStartGetsItsConfig:
    def test_every_swarm_service_of_ours_mounts_the_config_read_only(self) -> None:
        held = ours_in_the_stack()

        assert {"data-hub", "control-plane", "observation-service", *POOL} <= set(held)
        for name, body in held.items():
            assert f"{TREE}:/opt/eemls/config:ro" in body["volumes"], name

    def test_every_swarm_consumer_mounts_the_scenarios_read_only(self) -> None:
        consumers = {
            name: body for name, body in ours_in_the_stack().items() if name.startswith("consumer-")
        }

        assert consumers
        for name, body in consumers.items():
            assert f"{SCENARIOS}:/opt/eemls/scenarios:ro" in body["volumes"], name

    def test_kube_holds_the_run_s_scenario_in_a_config_map_of_its_own(self) -> None:
        [held] = files(run=CONSUMERS_RUN)["scenario.yaml"]

        assert held["kind"] == "ConfigMap"
        assert held["metadata"] == {"name": "eemls-scenario", "namespace": "eemls"}
        assert held["data"] == {"smoke.yaml": (SCENARIOS / "smoke.yaml").read_text("utf-8")}
        assert "scenario.yaml" in files(run=CONSUMERS_RUN)[KUSTOMIZATION][0]["resources"]

    def test_every_kube_consumer_mounts_the_run_s_scenario_read_only(self) -> None:
        jobs = of_kind(manifests(run=CONSUMERS_RUN), "Job")

        assert jobs
        for name, job in jobs.items():
            pod = job["spec"]["template"]["spec"]
            volumes = {one["name"]: one for one in pod["volumes"]}
            mounts = {one["name"]: one for one in pod["containers"][0]["volumeMounts"]}
            assert volumes["scenario"]["configMap"] == {
                "name": "eemls-scenario",
                "items": [{"key": "smoke.yaml", "path": "smoke.yaml"}],
            }, name
            assert mounts["scenario"] == {
                "name": "scenario",
                "mountPath": "/opt/eemls/scenarios",
                "readOnly": True,
            }, name

    def test_without_consumers_there_is_no_scenario(self) -> None:
        assert "scenario.yaml" not in files()

    @pytest.mark.parametrize("target", ["swarm", "kube"])
    def test_consumers_without_a_scenarios_folder_are_refused(self, target: str) -> None:
        deployment = plan_deployment(config(), throughput())
        mounts = Mounts(images="/srv/eemls/images", config=str(TREE), models="/srv/eemls/models")
        write = swarm_stack if target == "swarm" else kube_files

        with pytest.raises(DeployError, match="scenarios folder"):
            write(deployment, images=IMAGES, mounts=mounts, run=CONSUMERS_RUN)

    @pytest.mark.parametrize("target", ["swarm", "kube"])
    def test_a_scenario_the_folder_does_not_hold_is_refused(self, target: str) -> None:
        deployment = plan_deployment(config(), throughput())
        run = RunSettings(run_id="r1", scenario="no_such_scenario", duration_s=60.0, seed=0)
        write = swarm_stack if target == "swarm" else kube_files

        with pytest.raises(DeployError, match="no_such_scenario"):
            write(deployment, images=IMAGES, mounts=MOUNTS, run=run)


class TestWhatItRefusesToPlace:
    def test_a_service_with_no_measured_throughput_is_refused_by_name(self) -> None:
        measured = throughput()
        del measured["infer-b7-general-p5820"]

        with pytest.raises(DeployError, match="infer-b7-general-p5820 .* no measured"):
            plan_deployment(config(), measured)

    def test_copies_that_cannot_carry_every_consumer_are_refused(self) -> None:
        with pytest.raises(DeployError, match=r"infer-b7-general-p5820 .* 1\.0 .* 60\.0"):
            plan_deployment(config(), throughput(**{"infer-b7-general-p5820": 1.0}))

    def test_a_copy_larger_than_its_device_is_refused(self) -> None:
        settings = config()
        devices = {
            name: device.model_copy(update={"memory_gb": 1})
            for name, device in settings.cost.devices.items()
        }
        small = settings.model_copy(
            update={"cost": settings.cost.model_copy(update={"devices": devices})}
        )

        with pytest.raises(DeployError, match="needs .* MB a copy, and a .* holds 1024 MB"):
            plan_deployment(small, throughput())

    def test_copies_are_placed_whole_on_one_device_each(self) -> None:
        settings = config()
        bt3 = {
            name: service.model_copy(update={"replicas": 3})
            if service.device_type == "beelink_bt3"
            else service
            for name, service in settings.models.inference_services.services.items()
        }
        deployment = settings.services.deployment.model_copy(
            update={
                "runtime_memory_mb": dict.fromkeys(
                    ("x86_cpu", "x86_gpu", "arm_cpu", "jetson"), 2560.0
                )
            }
        )
        crowded = settings.model_copy(
            update={
                "models": settings.models.model_copy(
                    update={
                        "inference_services": settings.models.inference_services.model_copy(
                            update={"services": bt3}
                        )
                    }
                ),
                "services": settings.services.model_copy(update={"deployment": deployment}),
            }
        )

        with pytest.raises(DeployError, match="copies on beelink_bt3 do not fit, .* its 2 device"):
            plan_deployment(crowded, throughput())

    def test_more_copies_than_the_devices_of_a_type_hold_are_refused(self) -> None:
        settings = config()
        services = {
            name: service.model_copy(update={"replicas": 40})
            for name, service in settings.models.inference_services.services.items()
        }
        autoscale = settings.models.inference_services.autoscale
        crowded = settings.model_copy(
            update={
                "models": settings.models.model_copy(
                    update={
                        "inference_services": settings.models.inference_services.model_copy(
                            update={
                                "services": services,
                                "autoscale": autoscale.model_copy(update={"max_replicas": 40}),
                            }
                        )
                    }
                )
            }
        )

        with pytest.raises(DeployError, match="copies on .* do not fit, each whole on one of its"):
            plan_deployment(crowded, throughput())


def test_an_image_is_named_eemls_and_its_service_under_the_registry() -> None:
    images = Images(registry="docker.io/example", version="1.0")

    assert images.of("control-plane") == "docker.io/example/eemls-control-plane:1.0"
    assert (
        images.of("inference-service", "jetson")
        == "docker.io/example/eemls-inference-service:jetson-1.0"
    )


class TestTheModelsOfEachDevice:
    @pytest.mark.parametrize("service", POOL)
    def test_swarm_mounts_the_device_models_and_their_sources_read_only(self, service: str) -> None:
        volumes = stack()["services"][service]["volumes"]

        assert "/srv/eemls/models:/opt/eemls/models:ro" in volumes
        assert "/srv/eemls/models/models.yaml:/opt/eemls/config/local/models.yaml:ro" in volumes

    @pytest.mark.parametrize("service", POOL)
    def test_kube_mounts_the_device_models_and_reads_their_sources_there(
        self, service: str
    ) -> None:
        pod = of_kind(manifests(), "Deployment")[service]["spec"]["template"]["spec"]
        container = pod["containers"][0]
        mounts = {one["name"]: one for one in container["volumeMounts"]}
        volumes = {one["name"]: one for one in pod["volumes"]}
        env = {one["name"]: one.get("value") for one in container["env"]}

        assert mounts["models"] == {
            "name": "models",
            "mountPath": "/opt/eemls/models",
            "readOnly": True,
        }
        assert volumes["models"]["hostPath"] == {"path": "/srv/eemls/models", "type": "Directory"}
        assert env["EEMLS_MODEL_SOURCES"] == "/opt/eemls/models/models.yaml"
        assert set(mounts) == {"config", "models"}

    def test_no_other_service_mounts_the_models(self) -> None:
        others = {name: body for name, body in stack()["services"].items() if name not in set(POOL)}

        assert others
        for name, body in others.items():
            assert not any("models" in one for one in body.get("volumes", [])), name


class TestTheModelsADeviceHolds:
    def test_a_device_type_holds_the_model_of_every_service_placed_on_it(self) -> None:
        settings = config()
        services = settings.models.inference_services.services
        device = next(iter(services.values())).device_type

        held = models_on(settings, device)

        assert held == tuple(
            sorted({one.model for one in services.values() if one.device_type == device})
        )

    def test_a_device_type_no_service_is_placed_on_holds_none(self) -> None:
        assert models_on(config(), "nowhere") == ()


class TestTheFolderItWrites:
    def test_every_object_is_in_the_eemls_namespace(self) -> None:
        held = manifests()
        named = [one for one in held if one["kind"] != "Namespace"]

        assert of_kind(held, "Namespace") == {
            "eemls": {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "eemls"}}
        }
        assert {one["metadata"]["namespace"] for one in named} == {"eemls"}

    def test_the_kustomization_applies_every_file_but_what_the_run_keeps(self) -> None:
        written = files(run=RunSettings("r1", "smoke", 60.0, seed=0))
        [kustomization] = written[KUSTOMIZATION]

        assert kustomization["apiVersion"] == "kustomize.config.k8s.io/v1beta1"
        assert kustomization["kind"] == "Kustomization"
        assert kustomization["namespace"] == "eemls"
        assert kustomization["resources"] == sorted(set(written) - {KUSTOMIZATION, "run.yaml"})

    def test_the_run_keeps_its_namespace_and_the_volume_of_its_rows(self) -> None:
        held = files(run=RunSettings("r1", "smoke", 60.0, seed=0))["run.yaml"]

        assert [(one["kind"], one["metadata"]["name"]) for one in held] == [
            ("Namespace", "eemls"),
            ("PersistentVolumeClaim", "metrics-database-r1"),
        ]

    def test_the_control_plane_keeps_its_name_when_its_pod_is_replaced(self) -> None:
        pod = of_kind(manifests(), "Deployment")["control-plane"]["spec"]["template"]["spec"]
        env = {one["name"]: one for one in pod["containers"][0]["env"]}

        assert env["EEMLS_INSTANCE_ID"] == {"name": "EEMLS_INSTANCE_ID", "value": "control-plane-1"}

    @pytest.mark.parametrize("service", POOL)
    def test_a_service_on_a_gpu_runs_with_the_nvidia_runtime(self, service: str) -> None:
        settings = config()
        provider = settings.cost.devices[device_of(service)].execution_provider
        pod = of_kind(manifests(), "Deployment")[service]["spec"]["template"]["spec"]

        assert pod.get("runtimeClassName") == settings.services.deployment.runtime_classes.get(
            provider
        )

    @pytest.mark.parametrize("service", POOL)
    def test_an_inference_service_has_its_own_file(self, service: str) -> None:
        held = files()[f"inference/{service}.yaml"]

        assert {(one["kind"], one["metadata"]["name"]) for one in held} == {
            ("Deployment", service),
            ("Service", service),
        }

    def test_the_core_services_and_the_consumers_are_in_their_own_folders(self) -> None:
        written = files(run=RunSettings("r1", "smoke", 60.0, seed=0))

        assert sorted(path for path in written if path.startswith("core/")) == [
            "core/broker.yaml",
            "core/control-plane.yaml",
            "core/data-hub.yaml",
            "core/observation-service.yaml",
            "core/postgres.yaml",
        ]
        assert sorted(path for path in written if path.startswith("consumers/")) == [
            f"consumers/consumer-{consumer.replace('_', '-')}.yaml"
            for consumer in sorted(config().consumers)
        ]
        assert [one["kind"] for one in written["core/postgres.yaml"]] == ["Deployment", "Service"]

    def test_a_folder_without_a_run_has_no_consumers(self) -> None:
        assert not any(path.startswith("consumers/") for path in files())


@pytest.mark.parametrize("service", POOL)
def test_the_copies_of_a_service_spread_over_the_nodes_of_its_device_type(service: str) -> None:
    pod = of_kind(manifests(), "Deployment")[service]["spec"]["template"]["spec"]

    assert pod["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "ScheduleAnyway",
            "labelSelector": {"matchLabels": {"app": service}},
        }
    ]


class TestWhenTheImagesArePulled:
    def containers(self, images: Images) -> list[dict[str, Any]]:
        deployment = plan_deployment(config(), throughput())
        written = kube_files(
            deployment, images=images, mounts=MOUNTS, run=RunSettings("r1", "smoke", 60.0, seed=0)
        )
        return [
            container
            for path, held in written.items()
            if path != KUSTOMIZATION
            for one in held
            if one["kind"] in ("Deployment", "Job")
            for container in one["spec"]["template"]["spec"]["containers"]
            if "/eemls-" in container["image"]
        ]

    def test_the_latest_images_are_always_pulled(self) -> None:
        held = self.containers(Images(registry="docker.io/someone", version="latest"))

        assert held
        assert {one["imagePullPolicy"] for one in held} == {"Always"}

    def test_a_fixed_version_is_pulled_once(self) -> None:
        held = self.containers(Images(registry="docker.io/someone", version="1.0"))

        assert all("imagePullPolicy" not in one for one in held)


@pytest.mark.parametrize("service", POOL)
def test_a_copy_knows_its_run_and_its_image_on_both_targets(service: str) -> None:
    run = RunSettings("r1", seed=0)
    image = f"{IMAGES.registry}/eemls-inference-service:{tag_of(service)}-{IMAGES.version}"
    swarm = stack(run=run)["services"][service]["environment"]
    container = of_kind(manifests(run=run), "Deployment")[service]["spec"]["template"]["spec"][
        "containers"
    ][0]
    kube = {one["name"]: one.get("value") for one in container["env"]}

    assert (swarm["EEMLS_RUN_ID"], swarm["EEMLS_IMAGE"]) == ("r1", image)
    assert (kube["EEMLS_RUN_ID"], kube["EEMLS_IMAGE"]) == ("r1", image)


@pytest.mark.parametrize("image", [DATABASE_IMAGE, BROKER_IMAGE])
def test_an_outside_image_names_an_exact_version(image: str) -> None:
    _, _, tag = image.rpartition(":")

    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", tag), image
