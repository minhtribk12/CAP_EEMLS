"""`eemls-config`: validate or render a configuration tree.

eemls-config check config
eemls-config render config --overrides deploy/one_host/overrides.yaml --out runs/x/config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eemls_tools.config_tree import ConfigError, load_config
from eemls_tools.render import rendered


def check(directory: Path) -> int:
    try:
        config = load_config(directory)
    except ConfigError as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1
    print(
        f"{directory}: {len(config.consumers)} consumers, {len(config.approaches)} approaches, "
        f"{len(config.models.model_repository.models)} models, "
        f"{len(config.models.inference_services.services)} inference services"
    )
    return 0


def render(directory: Path, overrides: Path, out: Path) -> int:
    try:
        rendered(directory, overrides, out)
    except ConfigError as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1
    print(f"{out}: {directory} with the keys of {overrides}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-config")
    parser.add_argument("command", choices=["check", "render"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.directory)
    if args.overrides is None or args.out is None:
        parser.error("render needs --overrides and --out")
    return render(args.directory, args.overrides, args.out)


if __name__ == "__main__":
    sys.exit(main())
