from __future__ import annotations

from eemls.metrics_database.database import MetricsDatabase
from eemls.observation_service.observation_service_api import RowBatch


def store_metrics(batch: RowBatch, database: MetricsDatabase) -> int:
    for query in batch.queries:
        database.write_query(query)
    database.write_member_predictions(batch.member_predictions)
    for returned in batch.feedback:
        database.write_feedback(returned)
    for decision in batch.decisions:
        database.write_decision(decision)
    database.flush()
    return batch.rows
