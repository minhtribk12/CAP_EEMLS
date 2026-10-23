from __future__ import annotations

from eemls.domain.types import AggregatedResult, Query
from eemls.messaging.result_publisher import ResultPublisher


class ResultDelivery:
    def __init__(self, broker: ResultPublisher) -> None:
        self._broker = broker

    def __call__(self, query: Query, result: AggregatedResult) -> None:
        self._broker.publish(
            topic=query.reply_topic,
            payload=result.model_dump_json().encode("utf-8"),
            correlation_id=query.correlation_id,
            consumer_id=query.consumer_id,
        )
