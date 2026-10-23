from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from eemls.domain.records import Feedback
from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.testing.records import AT, decision, member, prediction, served
from eemls_control_plane import collect
from eemls_control_plane.collect import export_run

TREE = Path(__file__).resolve().parents[4] / "config"

UNANSWERED: dict[str, object] = {
    "outcome": "failed",
    "predicted_label": None,
    "predicted_confidence": None,
    "t_result_ready": None,
    "t_publish": None,
}


def stored() -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase(run_id="r1")
    for index in range(3):
        arrived = AT + timedelta(seconds=index)
        timing: dict[str, object] = {
            "t_arrive": arrived,
            "t_result_ready": arrived + timedelta(milliseconds=174),
            "t_publish": arrived + timedelta(milliseconds=180),
        }
        database.write_query(
            served(query_id=f"q-{index}", **(timing | (UNANSWERED if index == 2 else {})))
        )
        database.write_member_predictions(
            (
                prediction(
                    query_id=f"q-{index}",
                    member=member("b6", "x86_cpu"),
                    cache_hit=index != 1,
                ),
            )
        )
    database.write_feedback(
        Feedback(
            query_id="q-0",
            consumer_id="c1_navigation",
            true_label="car",
            t_feedback=AT,
            delay_ms=1.0,
        )
    )
    database.write_decision(decision())
    return database


def test_every_table_of_the_run_is_written_one_row_a_line(tmp_path: Path) -> None:
    export_run(stored(), tmp_path)

    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()

    assert [json.loads(line)["query_id"] for line in lines] == ["q-0", "q-1", "q-2"]
    for table in ("member_predictions", "feedback", "decisions", "instances"):
        assert (tmp_path / f"{table}.jsonl").is_file(), table


def test_the_summary_counts_every_table(tmp_path: Path) -> None:
    summary = export_run(stored(), tmp_path)

    assert summary["rows"] == {
        "instances": 0,
        "queries": 3,
        "member_predictions": 3,
        "feedback": 1,
        "decisions": 1,
        "inference_service_profiles": 0,
    }


def test_the_summary_gives_the_cache_hit_rate_of_the_members_that_answered(tmp_path: Path) -> None:
    summary = export_run(stored(), tmp_path)

    assert summary["cache_hit_rate"] == 2 / 3


def test_the_cache_hit_rate_counts_only_the_calls_to_the_members(tmp_path: Path) -> None:
    database = stored()
    database.write_member_predictions(
        tuple(
            prediction(
                query_id=f"q-{index}",
                member=member("b5", "x86_cpu"),
                in_ensemble=False,
                cache_hit=False,
            )
            for index in range(3)
        )
    )

    summary = export_run(database, tmp_path)

    assert summary["cache_hit_rate"] == 2 / 3


def test_the_summary_holds_the_row_counts_and_the_cache_hit_rate_only(tmp_path: Path) -> None:
    summary = export_run(stored(), tmp_path)

    assert set(summary) == {"rows", "cache_hit_rate"}


def test_the_summary_is_written_beside_the_tables(tmp_path: Path) -> None:
    summary = export_run(stored(), tmp_path)

    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary


def test_the_export_reads_without_a_statement_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: dict[str, object] = {}

    def refused(*args: object, **kwargs: object) -> None:
        opened.update(kwargs)
        raise DatabaseUnavailableError("no database")

    monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "example")
    monkeypatch.setattr(PostgresDatabase, "from_config", refused)

    assert collect.main(["--config", str(TREE), "--run-id", "r1", "--out", "unused"]) == 1
    assert opened["statement_timeout_ms"] == 0
