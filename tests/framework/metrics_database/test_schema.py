from __future__ import annotations

import pytest

from eemls.config.consumers import load_consumers
from eemls.config.reading import ConfigError, Reader
from eemls.domain.consumer_defined_metrics import MetricSpecification
from eemls.domain.records import (
    Decision,
    Feedback,
    InferenceServiceProfile,
    Instance,
    RunOwner,
    ServedQuery,
)
from eemls.domain.types import FrozenModel, MemberPrediction
from eemls.metrics_database.schema import (
    INDEXES,
    STORED_FLAT,
    TABLES,
    WRITTEN_BY,
    CoverageError,
    check_every_scored_cdm_can_be_computed,
    field_columns,
    schema_statements,
    table_columns,
)
from eemls.testing.records import decision, prediction, served
from eemls.testing.trees import TREE


class TestTheTables:
    def test_every_table_of_the_plan_exists(self) -> None:
        assert set(TABLES) == {
            "run_owners",
            "instances",
            "queries",
            "member_predictions",
            "feedback",
            "decisions",
            "inference_service_profiles",
        }

    @pytest.mark.parametrize("table", list(TABLES))
    def test_every_table_carries_the_run_it_belongs_to(self, table: str) -> None:
        assert "run_id" in table_columns(table)

    @pytest.mark.parametrize("table", list(TABLES))
    def test_every_table_can_be_created_twice(self, table: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS" in TABLES[table]

    def test_every_index_can_be_created_twice(self) -> None:
        for statement in INDEXES:
            assert "CREATE INDEX IF NOT EXISTS" in statement

    def test_the_whole_schema_is_one_ordered_list_of_statements(self) -> None:
        found = schema_statements()

        assert len(found) == len(TABLES) + len(INDEXES)
        assert found.index(TABLES["queries"]) < found.index(TABLES["member_predictions"])


class TestTheIndexesTheWindowReadsNeed:
    def test_the_queries_of_one_consumer_are_indexed_by_arrival(self) -> None:
        wanted = [index for index in INDEXES if "ON queries " in index]

        assert any("consumer_id" in index and "t_arrive" in index for index in wanted)

    def test_the_feedback_of_one_consumer_is_indexed_by_arrival(self) -> None:
        wanted = [index for index in INDEXES if "ON feedback " in index]

        assert any("consumer_id" in index and "t_feedback" in index for index in wanted)

    @pytest.mark.parametrize("table", ("member_predictions", "feedback"))
    def test_each_joined_table_is_found_by_its_query(self, table: str) -> None:
        primary = [line for line in TABLES[table].splitlines() if "PRIMARY KEY" in line]
        secondary = [index for index in INDEXES if f"ON {table} " in index]

        covered = any("run_id, query_id" in line for line in primary) or any(
            "query_id" in index for index in secondary
        )
        assert covered

    def test_the_decisions_of_one_consumer_are_indexed_by_time(self) -> None:
        wanted = [index for index in INDEXES if "ON decisions " in index]

        assert any("consumer_id" in index and "t" in index for index in wanted)

    def test_the_latest_profile_of_an_inference_service_is_found_by_an_index(self) -> None:
        wanted = [index for index in INDEXES if "inference_service_profiles" in index]

        assert any("instance_id" in index and "window_end" in index for index in wanted)


class TestTheTypesAndTheColumnsAgree:
    @pytest.mark.parametrize(
        ("table", "written"),
        [
            ("run_owners", RunOwner),
            ("instances", Instance),
            ("queries", ServedQuery),
            ("member_predictions", MemberPrediction),
            ("feedback", Feedback),
            ("decisions", Decision),
            ("inference_service_profiles", InferenceServiceProfile),
        ],
    )
    def test_every_field_of_the_type_has_a_column_of_the_same_name(
        self, table: str, written: type[FrozenModel]
    ) -> None:
        assert WRITTEN_BY[table] is written

        held = table_columns(table)
        for name in written.model_fields:
            for column in field_columns(table, name):
                assert column in held, f"{table} has no column for {written.__name__}.{name}"

    @pytest.mark.parametrize(
        ("table", "derived"),
        [("queries", ("response_time_ms", "publish_ms")), ("decisions", ())],
    )
    def test_a_derived_value_is_stored_as_well(self, table: str, derived: tuple[str, ...]) -> None:
        held = table_columns(table)

        for name in derived:
            assert name in held

    def test_the_runtime_context_is_stored_flat_so_no_reader_splits_it(self) -> None:
        held = table_columns("queries")

        for name in (
            "period_name",
            "period_index",
            "condition",
            "weather",
        ):
            assert name in held

    def test_the_place_in_the_simulated_day_is_stored_with_the_query(self) -> None:
        assert "time_of_day" in table_columns("queries")

    @pytest.mark.parametrize("table", list(TABLES))
    def test_no_table_holds_a_column_nothing_writes(self, table: str) -> None:
        written = WRITTEN_BY[table]
        carried = {"run_id"}
        for name in (*written.model_fields, *written.model_computed_fields):
            carried |= set(field_columns(table, name))

        assert set(table_columns(table)) - carried == set()

    def test_a_nested_value_is_stored_flat_and_says_which_columns(self) -> None:
        assert field_columns("queries", "context") == STORED_FLAT[("queries", "context")]
        assert field_columns("member_predictions", "member") == ("service_name", "model", "device")

    def test_asking_for_a_field_no_table_holds_is_refused(self) -> None:
        with pytest.raises(CoverageError, match="queries has no column for unicorn"):
            field_columns("queries", "unicorn")


class TestBuildingARow:
    def test_every_column_of_a_query_row_is_filled(self) -> None:
        from eemls.metrics_database.schema import run_row

        row = run_row("queries", served(), run_id="run-1")

        assert set(row) == set(table_columns("queries"))
        assert row["run_id"] == "run-1"

    def test_the_context_is_expanded_into_its_columns(self) -> None:
        from eemls.metrics_database.schema import run_row

        row = run_row("queries", served(), run_id="run-1")

        assert row["condition"] == "night"
        assert row["condition"] == "night"
        assert row["period_index"] == 3
        assert "context" not in row

    def test_a_derived_value_is_in_the_row(self) -> None:
        from eemls.metrics_database.schema import run_row

        row = run_row("queries", served(), run_id="run-1")

        assert row["response_time_ms"] == pytest.approx(174.0)

    def test_the_member_of_a_prediction_is_expanded(self) -> None:
        from eemls.metrics_database.schema import run_row

        row = run_row("member_predictions", prediction(), run_id="run-1")

        assert row["service_name"] == "infer-b6-jetson_orin_nano"
        assert row["model"] == "b6"
        assert "member" not in row

    def test_a_nested_mapping_becomes_plain_data(self) -> None:
        from eemls.metrics_database.schema import run_row

        row = run_row("decisions", decision(), run_id="run-1")

        assert type(row["cdm_values"]) is dict
        assert row["cdm_values"] == {"accuracy_car": 0.86}
        assert type(row["action"]) is dict

    def test_writing_the_wrong_type_into_a_table_is_refused(self) -> None:
        from eemls.metrics_database.schema import run_row

        with pytest.raises(CoverageError, match="holds Decision, not ServedQuery"):
            run_row("decisions", served(), run_id="run-1")


class TestEveryConsumerCdmCanBeComputed:
    def test_the_committed_consumers_pass(self) -> None:
        check_every_scored_cdm_can_be_computed(load_consumers(Reader(TREE)))

    def test_a_cdm_reading_a_column_no_table_holds_is_refused_by_its_file(self) -> None:
        consumers = load_consumers(Reader(TREE))
        first = sorted(consumers)[0]
        consumer = consumers[first]
        name = sorted(consumer.cdms)[0]
        reading = consumer.cdms[name].model_copy(
            update={
                "specification": MetricSpecification(function="sum", attribute="no_such_column")
            }
        )
        changed = consumer.model_copy(update={"cdms": {**consumer.cdms, name: reading}})

        with pytest.raises(ConfigError, match=f"consumers/{first}/cdm.yaml: .*no_such_column"):
            check_every_scored_cdm_can_be_computed({first: changed})
