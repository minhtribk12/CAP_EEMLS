from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from eemls.domain.types import FrozenMap, FrozenModel, Identifier, NonNegativeCount, PositiveNumber
from eemls_tools.config_tree import ConfigError, ConfigTree
from eemls_tools.render import rendered_with

PLANE = "services/control_plane.yaml"


class ApproachRun(FrozenModel):
    approach: Identifier
    contract: Identifier | None = None


class ExperimentPlan(FrozenModel):
    description: str
    scenario: Identifier
    approaches: FrozenMap[ApproachRun]
    seeds: tuple[NonNegativeCount, ...]
    duration_s: PositiveNumber
    overrides: dict[str, dict[str, object]] = Field(default_factory=dict)


def plan_of(path: Path, *, config: ConfigTree, scenarios: Path) -> ExperimentPlan:
    try:
        plan = ExperimentPlan.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (ValidationError, yaml.YAMLError, OSError) as refused:
        raise ConfigError(f"{path}: {refused}") from refused
    if not plan.seeds:
        raise ConfigError(f"{path}: the plan names no seed to run")
    if not (scenarios / f"{plan.scenario}.yaml").is_file():
        raise ConfigError(f"{path}: no scenario {plan.scenario} in {scenarios}")
    stated = {name for consumer in config.consumers.values() for name in consumer.contract_variants}
    for name, run in plan.approaches.items():
        if run.approach not in config.approaches:
            raise ConfigError(f"{path}: {name} runs approach {run.approach}, which has no file")
        if run.contract is not None and run.contract not in stated:
            raise ConfigError(
                f"{path}: approach {name} scores contract {run.contract}, which no consumer states"
            )
    return plan


class PlannedRun(FrozenModel):
    approach: Identifier
    seed: NonNegativeCount
    run_id: str
    folder: Path


def planned_runs(
    plan: ExperimentPlan,
    *,
    name: str,
    approaches: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[PlannedRun, ...]:
    chosen_approaches = list(approaches) if approaches is not None else list(plan.approaches)
    chosen_seeds = list(seeds) if seeds is not None else list(plan.seeds)
    unknown = sorted(set(chosen_approaches) - set(plan.approaches))
    if unknown:
        raise ConfigError(f"the plan holds no approach {unknown}, only {sorted(plan.approaches)}")
    for seed in chosen_seeds:
        if seed not in plan.seeds:
            raise ConfigError(f"the plan runs no seed {seed}, only {list(plan.seeds)}")
    return tuple(
        PlannedRun(
            approach=approach,
            seed=seed,
            run_id=f"{name}-{approach}-s{seed}".replace("_", "-").lower(),
            folder=Path(plan.scenario) / approach / f"seed-{seed}",
        )
        for approach in chosen_approaches
        for seed in chosen_seeds
    )


def rendered_approach(plan: ExperimentPlan, approach: str, *, source: Path, out: Path) -> Path:
    if approach not in plan.approaches:
        raise ConfigError(f"the plan holds no approach {approach}, only {sorted(plan.approaches)}")
    chosen = plan.approaches[approach]
    overrides = dict(plan.overrides)
    overrides[PLANE] = {
        **overrides.get(PLANE, {}),
        "policy": chosen.approach,
        "contract": chosen.contract,
    }
    config = out / "config"
    autoscale = rendered_with(source, overrides, config).models.inference_services.autoscale.enabled
    manifest: dict[str, object] = {
        "description": plan.description,
        "scenario": plan.scenario,
        "approach": approach,
        "policy": chosen.approach,
        "contract": chosen.contract,
        "seeds": list(plan.seeds),
        "duration_s": plan.duration_s,
        "autoscale": autoscale,
        "commit": _commit(source),
        "rendered_at": datetime.now(UTC).isoformat(),
        "files": {
            str(path.relative_to(config)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(config.rglob("*.yaml"))
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def _commit(source: Path) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "-C", str(source), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
