from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from eemls.evaluation_engine.window import Window
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.postgres import DatabaseUnavailableError, reachable
from eemls.metrics_database.schema import CoverageError, table_columns
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.testing.records import decision, prediction, profile, served

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def store() -> MemoryMetricsDatabase:
    held = MemoryMetricsDatabase()
    held.apply_schema()
    held.write_query(served())
    held.write_member_predictions((prediction(),))
    return held


class TestTheInterface:
    def test_the_in_memory_database_is_a_metrics_database(self) -> None:
        assert isinstance(MemoryMetricsDatabase(), MetricsDatabase)

    def test_an_object_missing_one_call_is_not_a_store(self) -> None:
        class Partial:
            def apply_schema(self) -> None: ...

        assert not isinstance(Partial(), MetricsDatabase)

    def test_applying_the_schema_twice_is_allowed(self) -> None:
        held = MemoryMetricsDatabase()

        held.apply_schema()
        held.apply_schema()

        assert held.applied == 2


class TestWhatAWindowReadReturns:
    def test_a_query_without_feedback_has_no_true_label_yet(self) -> None:
        rows = store().read_window("c1_navigation", 500)

        assert len(rows) == 1
        assert rows[0]["true_label"] is None

    def test_a_query_with_feedback_carries_its_true_label(self) -> None:
        from eemls.domain.records import Feedback

        held = store()
        held.write_feedback(
            Feedback(
                query_id="q-1",
                consumer_id="c1_navigation",
                true_label="car",
                t_feedback=NOW,
                delay_ms=15000.0,
            )
        )

        rows = held.read_window("c1_navigation", 500)

        assert rows[0]["true_label"] == "car"

    def test_it_returns_nothing_for_another_consumer(self) -> None:
        assert store().read_window("c2_safety", 500) == []

    def test_it_returns_at_most_the_limit(self) -> None:
        held = MemoryMetricsDatabase()
        for index in range(5):
            held.write_query(served(query_id=f"q-{index}"))

        assert len(held.read_window("c1_navigation", 2)) == 2

    def test_the_rows_it_returns_can_be_evaluated(self) -> None:
        rows = store().read_window("c1_navigation", 500)

        assert len(Window(rows)) == 1

    @pytest.mark.parametrize(
        "field", ["predicted_label", "true_label", "response_time_ms", "condition"]
    )
    def test_every_field_it_returns_is_a_column_a_cdm_may_read(self, field: str) -> None:
        rows = store().read_window("c1_navigation", 500)

        assert field in rows[0]
        assert field in table_columns("queries") or field == "true_label"


class TestWhatElseAStoreHolds:
    def test_it_holds_a_decision(self) -> None:
        held = store()
        held.write_decision(decision())

        assert len(held.decisions) == 1

    def test_it_holds_and_returns_a_profile(self) -> None:
        held = store()
        held.write_profile(profile())

        assert held.read_profiles()[0].model == "b6"

    def test_it_flushes_on_request(self) -> None:
        held = store()
        held.flush()

        assert held.flushes == 1

    def test_asking_for_a_table_the_store_does_not_hold_is_refused(self) -> None:
        with pytest.raises(CoverageError, match="unicorns"):
            table_columns("unicorns")


class TestTheWindowIsWhatAMetricMayRead:
    def test_the_window_returns_every_column_a_metric_may_read(self) -> None:
        from eemls.metrics_database.testing import window_columns

        rows = store().read_window("c1_navigation", 500)

        assert set(rows[0]) == set(window_columns())

    def test_every_field_the_committed_consumers_read_is_in_the_window(self) -> None:
        import yaml

        from eemls.domain.consumer_defined_metrics import CdmSet

        rows = store().read_window("c1_navigation", 500)
        reads: set[str] = set()
        for folder in sorted(path.name for path in (ROOT / "config" / "consumers").iterdir()):
            body = yaml.safe_load(
                (ROOT / "config" / "consumers" / folder / "cdm.yaml").read_text(encoding="utf-8")
            )
            reads |= CdmSet.model_validate({"definitions": body["cdms"]}).attributes()

        assert reads <= set(rows[0])

    def test_every_declared_cdm_can_be_computed_from_a_window_read(self) -> None:
        import yaml

        from eemls.domain.consumer_defined_metrics import CdmSet
        from eemls.domain.records import Feedback
        from eemls.evaluation_engine.aggregation_functions import Undefined
        from eemls.evaluation_engine.engine import EvaluationEngine
        from eemls.evaluation_engine.window import Window

        held = store()
        for label in ("car", "truck", "bus", "pedestrian", "cyclist", "motorcycle"):
            query_id = f"q-{label}"
            held.write_query(served(query_id=query_id, predicted_label=label))
            held.write_feedback(
                Feedback(
                    query_id=query_id,
                    consumer_id="c1_navigation",
                    true_label=label,
                    t_feedback=NOW,
                    delay_ms=1.0,
                )
            )
        body = yaml.safe_load(
            (ROOT / "config" / "consumers" / "c1_navigation" / "cdm.yaml").read_text(
                encoding="utf-8"
            )
        )
        declared = CdmSet.model_validate({"definitions": body["cdms"]})
        window = Window(held.read_window("c1_navigation", 500))

        for name in declared.definitions:
            value = EvaluationEngine(declared).of(name, window)
            assert not isinstance(value, Undefined), f"{name} could not be computed: {value}"


class TestADatabaseThatIsStillStarting:
    def test_a_refusal_inside_the_connect_timeout_is_tried_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        class Opened:
            def close(self) -> None:
                return None

        def connect(*_: object, **__: object) -> Opened:
            attempts.append(1)
            if len(attempts) == 1:
                raise psycopg.OperationalError("server closed the connection unexpectedly")
            return Opened()

        monkeypatch.setattr("eemls.metrics_database.postgres.psycopg.connect", connect)

        reachable("postgresql://u:p@127.0.0.1:1/db", connect_timeout_s=5.0, retry_s=0.0)

        assert len(attempts) == 2

    def test_the_last_reason_is_given_once_the_timeout_has_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def connect(*_: object, **__: object) -> None:
            raise psycopg.OperationalError("password authentication failed for user eemls")

        monkeypatch.setattr("eemls.metrics_database.postgres.psycopg.connect", connect)

        with pytest.raises(DatabaseUnavailableError, match="password authentication failed"):
            reachable("postgresql://u:p@127.0.0.1:1/db", connect_timeout_s=0.0, retry_s=0.0)
