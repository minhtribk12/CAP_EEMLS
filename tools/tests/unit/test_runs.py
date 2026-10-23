from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from eemls_tools.config_tree import ConfigError, load_config
from eemls_tools.experiment_command import main
from eemls_tools.runs import ExperimentPlan, plan_of, planned_runs, rendered_approach
from support.repository import ROOT

TREE = ROOT / "config"
RUNS = ROOT / "experiments" / "runs"
SCENARIOS = ROOT / "experiments" / "scenarios"
PLANS = sorted(RUNS.glob("*.yaml"))


def written(root: Path, body: dict[str, object]) -> Path:
    path = root / "plan.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def minimal(**changes: object) -> dict[str, object]:
    return {
        "description": "two approaches, one seed",
        "scenario": "smoke",
        "approaches": {
            "cap": {"approach": "cap"},
            "cap_star": {"approach": "cap", "contract": "only_general_accuracy"},
        },
        "seeds": [0],
        "duration_s": 60.0,
    } | changes


class TestTheCommittedPlans:
    def test_each_is_named_after_its_scenario(self) -> None:
        assert {path.stem for path in PLANS} == {
            "quick_start",
            "recurring_ten_days",
            "unpredictable_ten_days",
        }
        for path in PLANS:
            assert yaml.safe_load(path.read_text(encoding="utf-8"))["scenario"] == path.stem

    @pytest.mark.parametrize("path", PLANS, ids=lambda one: one.stem)
    def test_every_plan_loads_against_the_committed_tree(self, path: Path) -> None:
        assert plan_of(path, config=load_config(TREE), scenarios=SCENARIOS).approaches

    def test_the_recurring_run_compares_the_approaches_of_the_article(self) -> None:
        held = plan_of(
            RUNS / "recurring_ten_days.yaml", config=load_config(TREE), scenarios=SCENARIOS
        )

        assert set(held.approaches) == {
            "cap",
            "cap_star",
            "cap_minus",
            "scale_reduction",
            "sneakpeek",
            "axiomvision",
            "cocktail",
            "frugalml",
            "fixed_ensemble",
        }
        assert held.approaches["cap_star"].contract == "only_general_accuracy"
        assert held.seeds == tuple(range(10))


class TestWhatAPlanRefuses:
    def test_a_scenario_no_file_holds_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="nowhere"):
            plan_of(
                written(tmp_path, minimal(scenario="nowhere")),
                config=load_config(TREE),
                scenarios=SCENARIOS,
            )

    def test_an_approach_no_file_holds_is_refused_by_approach(self, tmp_path: Path) -> None:
        body = minimal(approaches={"magic": {"approach": "magic"}})

        with pytest.raises(ConfigError, match="magic"):
            plan_of(written(tmp_path, body), config=load_config(TREE), scenarios=SCENARIOS)

    def test_a_contract_no_consumer_states_is_refused_by_approach(self, tmp_path: Path) -> None:
        body = minimal(approaches={"cap": {"approach": "cap", "contract": "imagined"}})

        with pytest.raises(ConfigError, match="imagined"):
            plan_of(written(tmp_path, body), config=load_config(TREE), scenarios=SCENARIOS)

    def test_a_plan_with_no_seed_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="seed"):
            plan_of(
                written(tmp_path, minimal(seeds=[])), config=load_config(TREE), scenarios=SCENARIOS
            )


class TestTheTreeOfOneArm:
    def test_it_runs_the_policy_and_the_contract_of_its_approach(self, tmp_path: Path) -> None:
        plan = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)

        out = rendered_approach(plan, "cap_star", source=TREE, out=tmp_path / "run")
        plane = load_config(out / "config").services.control_plane

        assert (plane.policy, plane.contract) == ("cap", "only_general_accuracy")

    def test_an_override_naming_a_key_that_does_not_exist_is_refused_by_path(
        self, tmp_path: Path
    ) -> None:
        body = minimal(overrides={"services/control_plane.yaml": {"decision_intervals": 5}})
        plan = plan_of(written(tmp_path, body), config=load_config(TREE), scenarios=SCENARIOS)

        with pytest.raises(ConfigError, match="services/control_plane.yaml: decision_intervals"):
            rendered_approach(plan, "cap", source=TREE, out=tmp_path / "run")

    def test_the_same_plan_renders_the_same_files(self, tmp_path: Path) -> None:
        plan = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)
        one = rendered_approach(plan, "cap", source=TREE, out=tmp_path / "one") / "config"
        two = rendered_approach(plan, "cap", source=TREE, out=tmp_path / "two") / "config"

        for path in one.rglob("*.yaml"):
            assert path.read_bytes() == (two / path.relative_to(one)).read_bytes()

    def test_the_manifest_holds_a_hash_of_every_file_it_rendered(self, tmp_path: Path) -> None:
        plan = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)
        out = rendered_approach(plan, "cap", source=TREE, out=tmp_path / "run")

        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        files = {
            str(path.relative_to(out / "config")): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((out / "config").rglob("*.yaml"))
        }

        assert manifest["files"] == files
        assert manifest["approach"] == "cap"
        assert manifest["scenario"] == "smoke"
        assert {"commit", "rendered_at", "seeds", "duration_s"} <= set(manifest)


def autoscaled(**block: object) -> dict[str, object]:
    held = yaml.safe_load((TREE / "models" / "inference_services.yaml").read_text(encoding="utf-8"))
    return {"models/inference_services.yaml": {"autoscale": held["autoscale"] | block}}


class TestAutoscalingInARun:
    def test_a_comparison_may_autoscale(self, tmp_path: Path) -> None:
        body = minimal(overrides=autoscaled(enabled=True))
        plan = plan_of(written(tmp_path, body), config=load_config(TREE), scenarios=SCENARIOS)

        out = rendered_approach(plan, "cap", source=TREE, out=tmp_path / "run")

        assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["autoscale"] is True

    def test_the_manifest_records_autoscaling_off(self, tmp_path: Path) -> None:
        body = minimal(overrides=autoscaled(enabled=False))
        plan = plan_of(written(tmp_path, body), config=load_config(TREE), scenarios=SCENARIOS)

        out = rendered_approach(plan, "cap", source=TREE, out=tmp_path / "run")

        assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["autoscale"] is False


class TestTheCommand:
    def test_it_renders_an_approach_of_a_plan(self, tmp_path: Path) -> None:
        code = main(
            [
                "render",
                str(RUNS / "recurring_ten_days.yaml"),
                "--approach",
                "cap",
                "--out",
                str(tmp_path / "run"),
            ]
        )

        assert code == 0
        assert (tmp_path / "run" / "manifest.json").is_file()

    def test_an_approach_the_plan_does_not_hold_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "render",
                str(RUNS / "recurring_ten_days.yaml"),
                "--approach",
                "oracle",
                "--out",
                str(tmp_path),
            ]
        )

        assert code == 1
        assert "oracle" in capsys.readouterr().err


def test_a_plan_is_a_value(tmp_path: Path) -> None:
    held = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)

    assert isinstance(held, ExperimentPlan)
    with pytest.raises(ValueError, match="frozen"):
        held.duration_s = 1.0  # type: ignore[misc]


class TestTheRunsAPlanPlans:
    def test_one_run_per_approach_and_seed(self, tmp_path: Path) -> None:
        plan = plan_of(
            written(tmp_path, minimal(seeds=[0, 1])), config=load_config(TREE), scenarios=SCENARIOS
        )

        held = planned_runs(plan, name="smoke_check")

        assert [(one.approach, one.seed) for one in held] == [
            ("cap", 0),
            ("cap", 1),
            ("cap_star", 0),
            ("cap_star", 1),
        ]

    def test_each_run_has_an_id_that_can_name_its_volume(self, tmp_path: Path) -> None:
        plan = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)

        ids = [one.run_id for one in planned_runs(plan, name="recurring_ten_days")]

        assert ids == ["recurring-ten-days-cap-s0", "recurring-ten-days-cap-star-s0"]

    def test_each_run_has_a_folder_of_its_scenario_approach_and_seed(self, tmp_path: Path) -> None:
        plan = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)

        (first, _) = planned_runs(plan, name="p")

        assert first.folder == Path("smoke") / "cap" / "seed-0"

    def test_a_run_can_be_narrowed_to_some_approaches_and_seeds(self, tmp_path: Path) -> None:
        plan = plan_of(
            written(tmp_path, minimal(seeds=[0, 1, 2])),
            config=load_config(TREE),
            scenarios=SCENARIOS,
        )

        held = planned_runs(plan, name="p", approaches=["cap_star"], seeds=[2])

        assert [(one.approach, one.seed) for one in held] == [("cap_star", 2)]

    def test_an_approach_or_a_seed_the_plan_does_not_hold_is_refused(self, tmp_path: Path) -> None:
        plan = plan_of(written(tmp_path, minimal()), config=load_config(TREE), scenarios=SCENARIOS)

        with pytest.raises(ConfigError, match="oracle"):
            planned_runs(plan, name="p", approaches=["oracle"])
        with pytest.raises(ConfigError, match="seed 7"):
            planned_runs(plan, name="p", seeds=[7])


class TestThePlanCommand:
    def test_it_prints_one_line_per_run_of_the_approaches_and_seeds_named(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "plan",
                str(RUNS / "recurring_ten_days.yaml"),
                "--approaches",
                "cap,cap_star",
                "--seeds",
                "0",
            ]
        )

        assert code == 0
        assert capsys.readouterr().out.splitlines() == [
            "cap 0 recurring-ten-days-cap-s0 recurring_ten_days/cap/seed-0 18000.0 "
            "recurring_ten_days",
            "cap_star 0 recurring-ten-days-cap-star-s0 recurring_ten_days/cap_star/seed-0 18000.0 "
            "recurring_ten_days",
        ]

    def test_with_nothing_named_it_plans_every_approach_and_seed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["plan", str(RUNS / "recurring_ten_days.yaml")]) == 0
        assert len(capsys.readouterr().out.splitlines()) == 9 * 10

    def test_a_seed_the_plan_does_not_run_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["plan", str(RUNS / "recurring_ten_days.yaml"), "--seeds", "10"]) == 1
        assert "seed 10" in capsys.readouterr().err
