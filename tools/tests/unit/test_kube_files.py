from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import ConfigError, ConfigTree, load_config
from eemls_tools.deployment_plan import Images, Mounts, RunSettings, plan_deployment
from eemls_tools.kube_files import kube_files
from support.deployment import of_kind
from support.repository import ROOT

IMAGES = Images(registry="registry.local:5000", version="1.0")
MOUNTS = Mounts(
    images="/srv/eemls/images",
    config=str(ROOT / "config"),
    models="/srv/eemls/models",
)
SERVICES = "models/inference_services.yaml"


STARTS_WITH_TWO = "infer-b1-general-pi4"


def tree_with(root: Path, **autoscale: Any) -> ConfigTree:
    tree = committed_tree(root / "config")
    path = tree / SERVICES
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["autoscale"] |= autoscale
    body["services"][STARTS_WITH_TWO]["replicas"] = 2
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_config(tree)


def documents(config: ConfigTree) -> list[dict[str, Any]]:
    deployment = plan_deployment(
        config, dict.fromkeys(config.models.inference_services.services, 100.0)
    )
    files = kube_files(
        deployment, images=IMAGES, mounts=MOUNTS, run=RunSettings(run_id="r1", seed=0)
    )
    return [one for path, held in files.items() if path.startswith("inference/") for one in held]


class TestTheAutoscaleBlock:
    def test_the_committed_tree_autoscales(self) -> None:
        tree = ROOT / "config"

        assert load_config(tree).models.inference_services.autoscale.enabled is True

    def test_a_ceiling_below_the_copies_a_service_starts_with_is_refused(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigError, match="max_replicas 0"):
            tree_with(tmp_path, max_replicas=0)

    def test_a_ceiling_below_a_service_s_copies_is_refused_by_name(self, tmp_path: Path) -> None:
        tree = committed_tree(tmp_path / "config")
        path = tree / SERVICES
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["services"]["infer-b1-general-pi4"]["replicas"] = 3
        body["autoscale"]["max_replicas"] = 2
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="infer-b1-general-pi4 starts with 3"):
            load_config(tree)


def test_a_cooldown_over_the_hour_kubernetes_allows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cooldown_s"):
        tree_with(tmp_path, cooldown_s=3601)


class TestWithAutoscalingOff:
    def test_no_autoscaler_is_written_and_each_deployment_holds_its_copies(
        self, tmp_path: Path
    ) -> None:
        config = tree_with(tmp_path, enabled=False)
        held = documents(config)
        services = config.models.inference_services.services

        assert of_kind(held, "HorizontalPodAutoscaler") == {}
        for name, deployment in of_kind(held, "Deployment").items():
            assert deployment["spec"]["replicas"] == services[name].replicas
        assert of_kind(held, "Deployment")[STARTS_WITH_TWO]["spec"]["replicas"] == 2


class TestWithAutoscalingOn:
    def test_every_inference_service_has_one_autoscaler_of_its_deployment(
        self, tmp_path: Path
    ) -> None:
        held = documents(tree_with(tmp_path, enabled=True))
        scalers = of_kind(held, "HorizontalPodAutoscaler")

        assert set(scalers) == set(of_kind(held, "Deployment"))
        for name, scaler in scalers.items():
            assert scaler["apiVersion"] == "autoscaling/v2"
            assert scaler["spec"]["scaleTargetRef"] == {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": name,
            }

    def test_the_bounds_are_the_copies_of_the_map_and_the_ceiling_of_the_block(
        self, tmp_path: Path
    ) -> None:
        config = tree_with(tmp_path, enabled=True, max_replicas=5)
        services = config.models.inference_services.services

        for name, scaler in of_kind(documents(config), "HorizontalPodAutoscaler").items():
            assert scaler["spec"]["minReplicas"] == services[name].replicas
            assert scaler["spec"]["maxReplicas"] == 5
        assert (
            of_kind(documents(config), "HorizontalPodAutoscaler")[STARTS_WITH_TWO]["spec"][
                "minReplicas"
            ]
            == 2
        )

    def test_the_target_is_the_cpu_share_of_the_block(self, tmp_path: Path) -> None:
        config = tree_with(tmp_path, enabled=True, cpu_percent=60)

        for scaler in of_kind(documents(config), "HorizontalPodAutoscaler").values():
            assert scaler["spec"]["metrics"] == [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": 60},
                    },
                }
            ]

    def test_a_copy_is_removed_only_after_the_cooldown(self, tmp_path: Path) -> None:
        config = tree_with(tmp_path, enabled=True, cooldown_s=90)

        for scaler in of_kind(documents(config), "HorizontalPodAutoscaler").values():
            assert scaler["spec"]["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == 90

    def test_a_deployment_leaves_its_copies_to_the_autoscaler(self, tmp_path: Path) -> None:
        held = documents(tree_with(tmp_path, enabled=True))

        for deployment in of_kind(held, "Deployment").values():
            assert "replicas" not in deployment["spec"]


class TestTheCpuACopyAsksFor:
    def test_every_inference_copy_requests_the_cpu_the_autoscaler_measures_against(
        self,
    ) -> None:
        config = load_config(ROOT / "config")
        cores = config.services.deployment.inference_cpu_cores

        for deployment in of_kind(documents(config), "Deployment").values():
            container = deployment["spec"]["template"]["spec"]["containers"][0]
            assert container["resources"]["requests"] == {"cpu": f"{round(cores * 1000)}m"}
