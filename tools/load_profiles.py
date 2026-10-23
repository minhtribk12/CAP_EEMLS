"""Load measured profiles into the store under one run id.

Usage:
    EEMLS_DATABASE_PASSWORD=... python3 tools/load_profiles.py --run demo
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from eemls_control_plane.config import load_control_plane_config

from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls.observation_service.profile_updates import profiles_in

ROOT = Path(__file__).resolve().parents[1]
FROM = ROOT / "models" / "profiles.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="load_profiles")
    parser.add_argument("--run", required=True)
    parser.add_argument("--profiles", type=Path, default=FROM)
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    args = parser.parse_args(argv)

    password = os.environ.get("EEMLS_DATABASE_PASSWORD")
    if not password:
        print("refused: EEMLS_DATABASE_PASSWORD is not set", file=sys.stderr)
        return 1
    if not args.profiles.is_file():
        print(f"refused: {args.profiles} is not there; run tools/profile_pool.py", file=sys.stderr)
        return 1
    config = load_control_plane_config(args.config)
    pool = config.models.inference_services.services
    held = [one for one in profiles_in(args.profiles) if one.service_name in pool]
    try:
        database = PostgresDatabase.from_config(
            config.services.metrics_database, password=password, run_id=args.run
        )
    except DatabaseUnavailableError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    database.apply_schema()
    for one in held:
        database.write_profile(one)
    database.flush()
    database.close()
    print(f"{len(held)} profiles stored under run {args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
