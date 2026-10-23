from __future__ import annotations

from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.observation_service.metrics_storage import store_metrics
from eemls.observation_service.observation_service_api import RowBatch
from eemls.testing.records import decision, feedback, prediction, served


def batch() -> RowBatch:
    return RowBatch(
        run_id="test",
        sender="control-plane-1",
        queries=(served(),),
        member_predictions=(prediction(), prediction(in_ensemble=False)),
        feedback=(feedback(),),
        decisions=(decision(),),
    )


class TestStoringABatch:
    def test_every_row_of_the_batch_is_stored(self) -> None:
        database = MemoryMetricsDatabase()

        store_metrics(batch(), database)

        assert list(database.queries) == ["q-1"]
        assert len(database.predictions) == 2
        assert list(database.feedback) == ["q-1"]
        assert list(database.decisions) == ["d-1"]

    def test_it_answers_how_many_rows_it_stored(self) -> None:
        assert store_metrics(batch(), MemoryMetricsDatabase()) == 5

    def test_the_rows_are_committed_before_it_answers(self) -> None:
        database = MemoryMetricsDatabase()

        store_metrics(batch(), database)

        assert database.flushes == 1

    def test_a_batch_sent_twice_is_stored_once(self) -> None:
        database = MemoryMetricsDatabase()

        store_metrics(batch(), database)
        store_metrics(batch(), database)

        assert len(database.read_rows("member_predictions")) == 2
        assert len(database.read_rows("queries")) == 1
