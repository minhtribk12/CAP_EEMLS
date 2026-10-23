"""`eemls-exp`: plan an experiment's runs, or render one approach.

eemls-exp plan experiments/runs/recurring_ten_days.yaml --approaches A,B --seeds 0
eemls-exp render experiments/runs/recurring_ten_days.yaml --approach A --out runs/r1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from eemls_tools.config_tree import ConfigError, load_config
from eemls_tools.runs import plan_of, planned_runs, rendered_approach


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-exp")
    parser.add_argument("command", choices=["plan", "render"])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--approach", help="the approach to render")
    parser.add_argument("--out", type=Path, help="where the rendered approach is written")
    parser.add_argument(
        "--approaches", help="the approaches to plan, comma separated; every approach if left out"
    )
    parser.add_argument(
        "--seeds", help="the seeds to plan, comma separated; every seed if left out"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(os.environ.get("EEMLS_CONFIG_DIR", "config"))
    )
    parser.add_argument("--scenarios", type=Path, help="the scenario files, beside the plans")
    args = parser.parse_args(argv)
    if args.command == "render" and (args.approach is None or args.out is None):
        parser.error("render needs --approach and --out")
    scenarios = args.scenarios or args.plan.resolve().parent.parent / "scenarios"
    try:
        plan = plan_of(args.plan, config=load_config(args.config), scenarios=scenarios)
        if args.command == "plan":
            runs = planned_runs(
                plan,
                name=args.plan.stem,
                approaches=args.approaches.split(",") if args.approaches else None,
                seeds=[int(seed) for seed in args.seeds.split(",")] if args.seeds else None,
            )
        else:
            out = rendered_approach(plan, args.approach, source=args.config, out=args.out)
    except ConfigError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    if args.command == "plan":
        for run in runs:
            print(
                run.approach,
                run.seed,
                run.run_id,
                run.folder,
                float(plan.duration_s),
                plan.scenario,
            )
        return 0
    print(f"{out}: the {args.approach} approach of {args.plan.name}, scenario {plan.scenario}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
