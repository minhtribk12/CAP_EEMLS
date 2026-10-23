from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from eemls.config.reading import ConfigError, Reader
from eemls.ensemble_optimization.approaches.baselines.scale_reduction import (
    ScaleReductionSettings,
    scales_for,
    search_space,
)
from eemls.observation_service.profile_updates import profiles_in
from eemls_tools.profiles import PROFILES

ROOT = Path(__file__).resolve().parents[1]
POLICY = "approaches/scale_reduction.yaml"
RATES = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
COLUMNS = (
    "rate",
    "replicas",
    "ensembles",
    "naive",
    "log10 naive",
    "scale reduction",
    "log10 scale reduction",
    "cap per action",
)


class SearchSpaceError(ValueError): ...


def throughputs_given(pairs: Sequence[str]) -> dict[str, list[float]]:
    held: dict[str, list[float]] = {}
    for pair in pairs:
        model, _, values = pair.partition("=")
        try:
            rates = [float(value) for value in values.split(",")]
        except ValueError:
            raise SearchSpaceError(f"{model} has no throughput in {pair!r}") from None
        if not model or any(rate <= 0.0 for rate in rates):
            raise SearchSpaceError(f"{pair!r} is not a model with positive throughputs")
        held[model] = rates
    return held


def throughputs_profiled(path: Path) -> dict[str, list[float]]:
    held: dict[str, list[float]] = {}
    for one in profiles_in(path):
        if one.throughput_rps > 0.0:
            held.setdefault(one.model, []).append(one.throughput_rps)
    return held


def rows(
    throughputs: Mapping[str, Sequence[float]],
    *,
    rates: Sequence[float],
    max_ensemble_size: int,
    device_types: int | None,
) -> list[tuple[str, ...]]:
    services = sum(
        device_types if device_types is not None else len(held) for held in throughputs.values()
    )
    every = [rate for held in throughputs.values() for rate in held]
    table: list[tuple[str, ...]] = []
    for rate in rates:
        space = search_space(
            throughputs,
            rate_per_s=rate,
            max_ensemble_size=max_ensemble_size,
            device_types=device_types,
        )
        scales = scales_for(rate, every)
        table.append(
            (
                f"{rate:g}",
                f"{min(scales)} to {max(scales)}",
                str(space.ensembles),
                str(space.naive),
                f"{math.log10(space.naive):.1f}",
                str(space.scale_reduction),
                f"{math.log10(space.scale_reduction):.1f}",
                str(services),
            )
        )
    return table


def printed(table: Sequence[tuple[str, ...]]) -> str:
    widths = [
        max(len(name), *(len(row[index]) for row in table)) for index, name in enumerate(COLUMNS)
    ]
    lines = [
        " | ".join(name.ljust(width) for name, width in zip(COLUMNS, widths, strict=True)),
        "-+-".join("-" * width for width in widths),
    ]
    lines += [
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in table
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="search_space")
    parser.add_argument("--profiles", type=Path, default=ROOT / PROFILES)
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--throughput", action="append", default=[])
    parser.add_argument("--device-types", type=int, default=None)
    parser.add_argument("--rates", default=",".join(f"{rate:g}" for rate in RATES))
    parser.add_argument("--max-size", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        throughputs = (
            throughputs_given(args.throughput)
            if args.throughput
            else throughputs_profiled(args.profiles)
        )
        rates = [float(rate) for rate in args.rates.split(",")]
    except ValueError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 2
    if not throughputs:
        print("refused: no model has a throughput", file=sys.stderr)
        return 2
    limit = args.max_size
    if limit is None:
        try:
            limit = Reader(args.config).build(ScaleReductionSettings, POLICY).max_ensemble_size
        except ConfigError as refused:
            print(f"refused: {refused}", file=sys.stderr)
            return 2
    print(
        printed(
            rows(
                throughputs,
                rates=rates,
                max_ensemble_size=limit,
                device_types=args.device_types,
            )
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
