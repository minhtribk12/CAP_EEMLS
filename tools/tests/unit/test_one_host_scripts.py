from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from support.deployment import copied_scripts, deploy_options
from support.repository import ROOT

# Each tool appends its arguments, one per line, to <tool>.args; docker answers what up.sh asks.
FAKE = """#!/bin/sh
printf '%s\\n' "$@" >> "$(dirname "$0")/$(basename "$0").args"
case "$1 $2" in
  "info --format") echo active ;;
  "ps --filter") echo running ;;
  "volume inspect") exit 1 ;;
esac
"""


def host(root: Path) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "poetry"):
        (bin_dir / tool).write_text(FAKE, encoding="utf-8")
        (bin_dir / tool).chmod(0o755)
    copied_scripts(root, "one_host")
    (root / "config" / "local").mkdir(parents=True)
    (root / "config" / "local" / "models.yaml").write_text("models: {}\n", encoding="utf-8")
    (root / "models").mkdir()
    return bin_dir


def ran(root: Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", f"deploy/one_host/{script}"],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
            "EEMLS_DATABASE_PASSWORD": "test",
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def given(bin_dir: Path, tool: str) -> list[str]:
    return (bin_dir / f"{tool}.args").read_text(encoding="utf-8").splitlines()


def volume_calls(bin_dir: Path) -> list[list[str]]:
    args = given(bin_dir, "docker")
    return [args[i : i + 3] for i, arg in enumerate(args) if arg == "volume"]


class TestStartingTheSystem:
    def test_it_renders_the_pool_with_the_overrides_it_is_given(self, tmp_path: Path) -> None:
        bin_dir = host(tmp_path)

        done = ran(tmp_path, "up.sh", OVERRIDES="my_overrides.yaml")

        assert done.returncode == 0, done.stdout + done.stderr
        args = given(bin_dir, "poetry")
        assert args[args.index("--overrides") + 1] == "my_overrides.yaml"

    def test_it_loads_and_keeps_the_profiles_it_is_given(self, tmp_path: Path) -> None:
        bin_dir = host(tmp_path)

        ran(tmp_path, "up.sh", SERVICE_PROFILES="models/profiles_cluster.jsonl")

        mount = f"{tmp_path}/models/profiles_cluster.jsonl:/opt/profiles.jsonl:ro"
        assert mount in given(bin_dir, "docker")
        held = (tmp_path / "runs" / "one_host" / "profiles").read_text(encoding="utf-8")
        assert held == "models/profiles_cluster.jsonl\n"

    def test_it_loads_profiles_given_by_a_full_path(self, tmp_path: Path) -> None:
        bin_dir = host(tmp_path)

        ran(tmp_path, "up.sh", SERVICE_PROFILES="/data/profiles.jsonl")

        assert "/data/profiles.jsonl:/opt/profiles.jsonl:ro" in given(bin_dir, "docker")

    def test_it_warms_the_inference_services(self, tmp_path: Path) -> None:
        bin_dir = host(tmp_path)

        ran(tmp_path, "up.sh", SEED="3")

        args = given(bin_dir, "docker")
        assert "eemls-warm" in args
        assert args[args.index("--seed") + 1] == "3"


class TestARun:
    def test_it_sizes_the_copies_from_the_profiles_the_system_started_from(
        self, tmp_path: Path
    ) -> None:
        bin_dir = host(tmp_path)
        out = tmp_path / "runs" / "one_host"
        out.mkdir(parents=True)
        (out / "run_id").write_text("r1\n", encoding="utf-8")
        (out / "profiles").write_text("models/profiles_cluster.jsonl\n", encoding="utf-8")
        (out / "seed").write_text("0\n", encoding="utf-8")

        ran(tmp_path, "run.sh", DURATION_S="1")

        args = given(bin_dir, "poetry")
        assert args[args.index("--profiles") + 1] == "models/profiles_cluster.jsonl"


class TestStoppingTheSystem:
    def test_no_run_is_up_once_it_stopped(self, tmp_path: Path) -> None:
        host(tmp_path)
        out = tmp_path / "runs" / "one_host"
        out.mkdir(parents=True)
        (out / "run_id").write_text("r1\n", encoding="utf-8")

        ran(tmp_path, "down.sh")

        assert not (out / "run_id").exists()


def test_the_local_images_carry_the_name_the_local_system_runs() -> None:
    planned = subprocess.run(
        ["make", "-n", "images-local"], cwd=ROOT, capture_output=True, text=True, check=True
    )

    baked = "REGISTRY=eemls.local poetry run python tools/pin.py --bake --load local"

    assert baked in planned.stdout


def test_one_host_plans_an_archived_profile_against_the_archived_scenarios(
    tmp_path: Path,
) -> None:
    bin_dir = host(tmp_path)

    ran(
        tmp_path,
        "experiment.sh",
        PLAN="experiments/archived/runs/shuffled_days_seed_3.yaml",
        OUT="out",
    )

    args = given(bin_dir, "poetry")
    assert args[args.index("--scenarios") + 1] == "experiments/archived/scenarios"


def test_the_system_runs_the_local_images_whatever_version_the_local_file_names(
    tmp_path: Path,
) -> None:
    bin_dir = host(tmp_path)
    out = tmp_path / "runs" / "one_host"

    ran(tmp_path, "up.sh")
    (out / "run_id").write_text("r1\n", encoding="utf-8")
    ran(tmp_path, "run.sh", DURATION_S="1")

    args = given(bin_dir, "poetry")
    versions = [args[at + 1] for at, one in enumerate(args) if one == "--version"]
    assert versions == ["latest", "latest"]


def test_up_deploys_the_control_plane_with_the_seed_it_is_given(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)

    ran(tmp_path, "up.sh", SEED="3")

    (up,) = deploy_options(given(bin_dir, "poetry"))
    assert up["--seed"] == "3"


def test_up_and_run_deploy_the_control_plane_alike_for_one_seed(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)

    ran(tmp_path, "up.sh", SEED="3")
    ran(tmp_path, "run.sh", DURATION_S="1")

    up, run = deploy_options(given(bin_dir, "poetry"))
    consumers = {"--scenario", "--scenarios", "--duration-s"}
    assert {k: v for k, v in run.items() if k not in consumers} == up


def test_run_keeps_the_seed_the_system_was_started_with(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)

    ran(tmp_path, "up.sh", SEED="3")
    ran(tmp_path, "run.sh", DURATION_S="1")

    up, run = deploy_options(given(bin_dir, "poetry"))
    assert (up["--seed"], run["--seed"]) == ("3", "3")


def test_run_ignores_a_seed_other_than_the_one_the_system_was_started_with(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)

    ran(tmp_path, "up.sh", SEED="3")
    ran(tmp_path, "run.sh", SEED="5", DURATION_S="1")

    _, run = deploy_options(given(bin_dir, "poetry"))
    assert run["--seed"] == "3"


def test_a_start_that_stops_early_removes_its_database_volume(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)
    (bin_dir / "poetry").write_text(
        FAKE.replace(
            'case "$1 $2" in', 'case "$*" in *eemls-deploy*) exit 1 ;; esac\ncase "$1 $2" in'
        ),
        encoding="utf-8",
    )

    done = ran(tmp_path, "up.sh", RUN_ID="r1")

    assert done.returncode != 0
    assert ["volume", "rm", "eemls_metrics-database-r1"] in volume_calls(bin_dir)


def test_a_start_that_succeeds_keeps_its_database_volume(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)

    done = ran(tmp_path, "up.sh", RUN_ID="r1")

    assert done.returncode == 0, done.stdout + done.stderr
    assert ["volume", "rm", "eemls_metrics-database-r1"] not in volume_calls(bin_dir)


def test_up_refuses_a_run_id_whose_database_volume_exists(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)
    (bin_dir / "docker").write_text(
        FAKE.replace('  "volume inspect") exit 1 ;;\n', ""), encoding="utf-8"
    )

    done = ran(tmp_path, "up.sh", RUN_ID="r1")

    assert done.returncode == 1
    assert done.stdout.strip().splitlines() == [
        "run id r1 has a database volume already; pick another RUN_ID or remove the volume"
    ]
    assert ["volume", "rm", "eemls_metrics-database-r1"] not in volume_calls(bin_dir)
    assert "deploy" not in given(bin_dir, "docker")


def test_up_refuses_a_run_id_outside_the_rule_before_anything_is_made(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)

    done = ran(tmp_path, "up.sh", RUN_ID="My_Run")

    assert done.returncode == 1
    assert done.stdout.strip().splitlines() == [
        "run id My_Run must be lowercase letters, digits and hyphens, with no hyphen at either end"
    ]
    assert not (tmp_path / "runs").exists()
    assert not (bin_dir / "docker.args").exists()
    assert not (bin_dir / "poetry.args").exists()


def a_run_is_up(root: Path) -> None:
    out = root / "runs" / "one_host"
    out.mkdir(parents=True)
    (out / "run_id").write_text("r1\n", encoding="utf-8")
    (out / "profiles").write_text("models/profiles.jsonl\n", encoding="utf-8")
    (out / "seed").write_text("0\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("scenarios", "mounted"),
    [
        (None, "experiments/scenarios"),
        ("experiments/archived/scenarios", "experiments/archived/scenarios"),
    ],
)
def test_run_mounts_the_scenario_folder_by_its_full_path(
    tmp_path: Path, scenarios: str | None, mounted: str
) -> None:
    bin_dir = host(tmp_path)
    a_run_is_up(tmp_path)

    ran(tmp_path, "run.sh", DURATION_S="1", **({"SCENARIOS": scenarios} if scenarios else {}))

    (run,) = deploy_options(given(bin_dir, "poetry"))
    assert run["--scenarios"] == f"{tmp_path}/{mounted}"


def test_run_mounts_a_scenario_folder_given_by_a_full_path(tmp_path: Path) -> None:
    bin_dir = host(tmp_path)
    a_run_is_up(tmp_path)

    ran(tmp_path, "run.sh", DURATION_S="1", SCENARIOS="/data/scenarios")

    (run,) = deploy_options(given(bin_dir, "poetry"))
    assert run["--scenarios"] == "/data/scenarios"


def test_an_experiment_runs_its_consumers_on_the_scenarios_it_planned_against(
    tmp_path: Path,
) -> None:
    bin_dir = host(tmp_path)
    (bin_dir / "poetry").write_text(
        FAKE.replace(
            'case "$1 $2" in',
            'case "$2 $3" in "eemls-exp plan") echo "cap 0 r1 cap_0 60.0 smoke" ;; esac\n'
            'case "$1 $2" in',
        ),
        encoding="utf-8",
    )
    (tmp_path / "out" / "cap_0" / "config").mkdir(parents=True)

    ran(
        tmp_path,
        "experiment.sh",
        PLAN="experiments/archived/runs/shuffled_days_seed_3.yaml",
        OUT="out",
    )

    options = deploy_options(given(bin_dir, "poetry"))
    [run] = [one for one in options if "--scenario" in one]
    assert run["--scenarios"] == f"{tmp_path}/experiments/archived/scenarios"


def docker_runs(script: str) -> list[str]:
    joined = script.replace("\\\n", " ")
    return [line for line in joined.splitlines() if "docker run " in line]


@pytest.mark.parametrize("script", sorted((ROOT / "deploy" / "one_host").glob("*.sh")), ids=str)
def test_every_container_a_script_starts_mounts_the_config_read_only(script: Path) -> None:
    for started in docker_runs(script.read_text(encoding="utf-8")):
        assert re.search(r"-v \S+:/opt/eemls/config:ro", started), started


ORDERED = """#!/bin/sh
echo "$(basename "$0") $*" >> "$(dirname "$0")/calls.log"
case "$*" in *"eemls-deploy name"*) eval last=\\${$#}; echo "$last" | tr _ - ;; esac
case "$*" in *"eemls-exp plan"*) echo "cap 0 r1 cap_0 60.0 smoke" ;; esac
case "$*" in *"service ls --filter label="*) printf 'eemls_infer-b1\neemls_broker\n'; exit ;; esac
case "$1 $2" in
  "info --format") echo active ;;
  "ps --filter") echo running ;;
  "volume inspect") exit 1 ;;
  "service ls") echo eemls_consumer-c1-navigation ;;
  "service ps") echo Complete ;;
esac
"""


def ordered(root: Path) -> None:
    bin_dir = host(root)
    for tool in ("docker", "poetry"):
        (bin_dir / tool).write_text(ORDERED, encoding="utf-8")


def logged(root: Path) -> list[str]:
    return (root / "bin" / "calls.log").read_text(encoding="utf-8").splitlines()


def first_at(log: list[str], part: str) -> int:
    return next(at for at, line in enumerate(log) if part in line)


class TestTheObservationServiceOnOneHost:
    def test_up_waits_until_the_observation_service_is_ready(self, tmp_path: Path) -> None:
        ordered(tmp_path)

        done = ran(tmp_path, "up.sh")

        assert done.returncode == 0, done.stdout + done.stderr
        held = logged(tmp_path)
        assert first_at(held, "name=eemls_observation-service --filter health=healthy") < (
            first_at(held, "eemls-warm")
        )

    def test_down_stops_each_service_after_the_ones_that_send_to_it(self, tmp_path: Path) -> None:
        ordered(tmp_path)
        (tmp_path / "runs" / "one_host" / "config").mkdir(parents=True)

        done = ran(tmp_path, "down.sh")

        assert done.returncode == 0, done.stdout + done.stderr
        held = logged(tmp_path)
        order = [
            first_at(held, part)
            for part in (
                "service rm eemls_consumer-c1-navigation",
                "service rm eemls_control-plane",
                "service rm eemls_infer-b1",
                "service rm eemls_observation-service",
                "service rm eemls_data-hub",
                "service rm eemls_broker",
                "service rm eemls_metrics-database",
                "stack rm eemls",
            )
        ]
        assert order == sorted(order)

    def test_an_experiment_stops_the_senders_then_the_observation_service_then_collects(
        self, tmp_path: Path
    ) -> None:
        ordered(tmp_path)
        (tmp_path / "out" / "cap_0" / "config").mkdir(parents=True)

        ran(
            tmp_path,
            "experiment.sh",
            PLAN="experiments/archived/runs/shuffled_days_seed_3.yaml",
            OUT="out",
        )

        held = logged(tmp_path)
        collect = first_at(held, "eemls-collect")
        consumers_done = max(at for at, line in enumerate(held[:collect]) if "service ps" in line)
        between = held[consumers_done:collect]
        control_plane = first_at(between, "service rm eemls_control-plane")
        inference = first_at(between, "service rm eemls_infer-b1")
        observation_service = first_at(between, "service rm eemls_observation")
        assert control_plane < inference < observation_service
