"""`eemls-collect`: export one finished run, table by table.

eemls-collect --run-id R --out /opt/eemls/out
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from eemls.config.reading import ConfigError
from eemls.metrics_database import MissingPasswordError, password_from_environment
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls.metrics_database.schema import EXPORTED
from eemls_control_plane.config import load_control_plane_config


def export_run(database: MetricsDatabase, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Sequence[Mapping[str, object]]] = {}
    for table in EXPORTED:
        rows = database.read_rows(table)
        tables[table] = rows
        with (out / f"{table}.jsonl").open("w", encoding="utf-8") as written:
            for row in rows:
                written.write(json.dumps(row, default=str, sort_keys=True) + "\n")
    summary = _summary(tables)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _summary(tables: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    answered = [
        row for row in tables["member_predictions"] if row["status"] == "ok" and row["in_ensemble"]
    ]
    return {
        "rows": {table: len(rows) for table, rows in tables.items()},
        "cache_hit_rate": (
            sum(1 for row in answered if row["cache_hit"]) / len(answered) if answered else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-collect")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        password = password_from_environment()
    except MissingPasswordError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    try:
        settings = load_control_plane_config(args.config).services.metrics_database
        database = PostgresDatabase.from_config(
            settings, password=password, run_id=args.run_id, statement_timeout_ms=0
        )
    except (ConfigError, DatabaseUnavailableError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    try:
        summary = export_run(database, args.out)
    finally:
        database.close()
    print(json.dumps(summary["rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
