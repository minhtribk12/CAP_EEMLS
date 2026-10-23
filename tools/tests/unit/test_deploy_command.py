from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from eemls.testing.records import profile
from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import load_config
from eemls_tools.deploy_command import main, measured_throughput
from support.repository import ROOT

TREE = ROOT / "config"


def profiles_file(root: Path, rps: float = 100.0, **later: float) -> Path:
    rows = [
        profile(service_name=name, throughput_rps=rps)
        for name in load_config(TREE).models.inference_services.services
    ]
    for name, value in later.items():
        first = profile(service_name=name)
        rows.append(
            profile(
                service_name=name,
                throughput_rps=value,
                window_start=first.window_start + timedelta(hours=1),
                window_end=first.window_end + timedelta(hours=1),
            )
        )
    path = root / "profiles.jsonl"
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    return path


def deployed(root: Path, target: str, *extra: str) -> int:
    return main(
        [
            target,
            "--config",
            str(TREE),
            "--profiles",
            str(profiles_file(root)),
            "--images",
            "/srv/eemls/images",
            "--out",
            str(root / "out.yaml"),
            "--run-id",
            "r1",
            "--seed",
            "0",
            *extra,
        ]
    )


class TestTheThroughputItSizesBy:
    def test_the_newest_profile_of_a_service_is_the_one_read(self, tmp_path: Path) -> None:
        held = measured_throughput(profiles_file(tmp_path, **{"infer-b7-general-p5820": 7.0}))

        assert held["infer-b7-general-p5820"] == pytest.approx(7.0)
        assert held["infer-b1-general-pi4"] == pytest.approx(100.0)


class TestWhatItWrites:
    def test_a_swarm_stack_holding_every_service_of_the_pool(self, tmp_path: Path) -> None:
        assert deployed(tmp_path, "swarm") == 0

        stack = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))

        assert set(load_config(TREE).models.inference_services.services) <= set(stack["services"])

    def test_a_kube_folder_holding_every_service_of_the_pool(self, tmp_path: Path) -> None:
        assert deployed(tmp_path, "kube") == 0

        folder = tmp_path / "out.yaml"
        listed = yaml.safe_load((folder / "kustomization.yaml").read_text(encoding="utf-8"))
        documents = [
            one
            for path in listed["resources"]
            for one in yaml.safe_load_all((folder / path).read_text(encoding="utf-8"))
        ]
        deployments = {one["metadata"]["name"] for one in documents if one["kind"] == "Deployment"}

        assert set(load_config(TREE).models.inference_services.services) <= deployments

    def test_a_kube_folder_written_again_holds_no_file_of_the_last(self, tmp_path: Path) -> None:
        stale = tmp_path / "out.yaml" / "inference" / "infer-gone.yaml"
        stale.parent.mkdir(parents=True)
        stale.write_text("kind: Deployment\n", encoding="utf-8")

        assert deployed(tmp_path, "kube") == 0
        assert not stale.exists()

    def test_a_run_adds_one_service_per_consumer(self, tmp_path: Path) -> None:
        code = deployed(
            tmp_path, "swarm", "--scenario", "smoke", "--duration-s", "60", "--seed", "2"
        )
        stack = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))

        assert code == 0
        assert len([name for name in stack["services"] if name.startswith("consumer-")]) == 3
        command = stack["services"]["consumer-c2-safety"]["command"]
        assert command[command.index("--seed") + 1] == "2"


class TestWhatItRefuses:
    def test_a_pool_its_copies_cannot_carry_is_refused_by_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "swarm",
                "--config",
                str(TREE),
                "--profiles",
                str(profiles_file(tmp_path, **{"infer-b7-general-p5820": 1.0})),
                "--images",
                "/srv/eemls/images",
                "--out",
                str(tmp_path / "out.yaml"),
                "--run-id",
                "r1",
                "--seed",
                "0",
            ]
        )

        assert code == 1
        assert "refused: infer-b7-general-p5820" in capsys.readouterr().err
        assert not (tmp_path / "out.yaml").exists()

    def test_a_scenario_without_a_length_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            deployed(tmp_path, "swarm", "--scenario", "smoke")

    def test_a_seed_is_required(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "swarm",
                    "--config",
                    str(TREE),
                    "--profiles",
                    str(profiles_file(tmp_path)),
                    "--images",
                    "/srv/eemls/images",
                    "--out",
                    str(tmp_path / "out.yaml"),
                    "--run-id",
                    "r1",
                ]
            )

    def test_a_run_id_is_required(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "swarm",
                    "--config",
                    str(TREE),
                    "--profiles",
                    str(profiles_file(tmp_path)),
                    "--images",
                    "/srv/eemls/images",
                    "--out",
                    str(tmp_path / "out.yaml"),
                ]
            )


class TestTheRegistryItPullsFrom:
    def tree_naming(self, root: Path, registry: str) -> Path:
        tree = committed_tree(root / "config")
        (tree / "local").mkdir()
        (tree / "local" / "images.yaml").write_text(
            yaml.safe_dump({"registry": registry, "version": "2.0"}), encoding="utf-8"
        )
        return tree

    def stack_of(self, root: Path, tree: Path, *extra: str) -> Any:
        code = main(
            [
                "swarm",
                "--config",
                str(tree),
                "--profiles",
                str(profiles_file(root)),
                "--images",
                "/srv/eemls/images",
                "--out",
                str(root / "out.yaml"),
                "--run-id",
                "r1",
                "--seed",
                "0",
                *extra,
            ]
        )
        assert code == 0
        return yaml.safe_load((root / "out.yaml").read_text(encoding="utf-8"))

    def test_with_no_registry_given_the_local_file_names_it(self, tmp_path: Path) -> None:
        stack = self.stack_of(tmp_path, self.tree_naming(tmp_path, "docker.io/someone"))

        assert stack["services"]["control-plane"]["image"] == (
            "docker.io/someone/eemls-control-plane:2.0"
        )

    def test_a_registry_given_is_the_one_used(self, tmp_path: Path) -> None:
        tree = self.tree_naming(tmp_path, "docker.io/someone")

        stack = self.stack_of(tmp_path, tree, "--registry", "eemls.local", "--version", "x")

        assert stack["services"]["control-plane"]["image"] == ("eemls.local/eemls-control-plane:x")


def test_a_kube_folder_carries_a_mounted_tree_as_a_config_map(tmp_path: Path) -> None:
    code = deployed(tmp_path, "kube", "--mount-config", str(TREE))
    folder = tmp_path / "out.yaml"
    listed = yaml.safe_load((folder / "kustomization.yaml").read_text(encoding="utf-8"))

    assert code == 0
    assert "config.yaml" in listed["resources"]


def test_without_a_mounted_tree_the_services_mount_the_tree_they_are_planned_from(
    tmp_path: Path,
) -> None:
    code = deployed(tmp_path, "swarm")
    stack = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))

    assert code == 0
    assert f"{TREE.resolve()}:/opt/eemls/config:ro" in stack["services"]["data-hub"]["volumes"]


def test_the_consumers_mount_the_scenario_folder_given(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "mine.yaml").write_text(
        (TREE.parent / "experiments" / "scenarios" / "smoke.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    code = deployed(
        tmp_path,
        "swarm",
        "--scenario",
        "mine",
        "--duration-s",
        "60",
        "--scenarios",
        str(scenarios),
    )
    stack = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))

    assert code == 0
    volumes = stack["services"]["consumer-c2-safety"]["volumes"]
    assert f"{scenarios}:/opt/eemls/scenarios:ro" in volumes


def test_a_scenario_the_folder_does_not_hold_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = deployed(
        tmp_path, "swarm", "--scenario", "mine", "--duration-s", "60", "--scenarios", str(tmp_path)
    )

    assert code == 1
    assert "refused: no scenario mine" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("service", "host"),
    [
        ("metrics_database", "postgres"),
        ("broker", "broker"),
        ("data_hub", "data-hub"),
        ("control_plane", "control-plane"),
        ("observation_service", "observation-service"),
    ],
)
def test_the_name_command_prints_the_host_of_each_service(
    service: str, host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["name", "--config", str(TREE), service]) == 0
    assert capsys.readouterr().out == f"{host}\n"


def test_the_name_command_refuses_a_service_it_has_no_host_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["name", "--config", str(TREE), "nowhere"]) == 1
    assert "nowhere" in capsys.readouterr().err
