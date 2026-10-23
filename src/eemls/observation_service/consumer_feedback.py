from __future__ import annotations

import logging
import threading
from collections import deque

from eemls.domain.measuring import Measuring
from eemls.domain.records import Feedback
from eemls.domain.types import Query, Timestamp
from eemls.metrics_database.database import MetricsDatabase

logger = logging.getLogger(__name__)


class ReturnedLabels:
    def __init__(
        self,
        database: MetricsDatabase,
        *,
        remember: int,
        monitoring: Measuring,
    ) -> None:
        self._database = database
        self._remember = remember
        self._monitoring = monitoring
        self._served: dict[str, tuple[str, Timestamp]] = {}
        self._order: dict[str, deque[str]] = {}
        self._labeled: set[str] = set()
        self._lock = threading.Lock()

    def arrived(self, query: Query) -> None:
        with self._lock:
            self._served[query.query_id] = (query.consumer_id, query.t_arrive)
            order = self._order.setdefault(query.consumer_id, deque())
            order.append(query.query_id)
            while len(order) > self._remember:
                forgotten = order.popleft()
                self._served.pop(forgotten, None)
                self._labeled.discard(forgotten)

    def label(self, query_id: str, *, consumer_id: str, true_label: str, at: Timestamp) -> bool:
        with self._lock:
            reason = self._refusal(query_id, consumer_id, at)
            if reason is None:
                _, arrived = self._served[query_id]
                self._labeled.add(query_id)
        if reason is not None:
            self._monitoring.count(
                "eemls_feedback_ignored_total", consumer=consumer_id, reason=reason
            )
            logger.info(f"error_type=feedback_ignored reason={reason} query_id={query_id}")
            return False
        self._database.write_feedback(
            Feedback(
                query_id=query_id,
                consumer_id=consumer_id,
                true_label=true_label,
                t_feedback=at,
                delay_ms=(at - arrived).total_seconds() * 1000.0,
            )
        )
        self._monitoring.count("eemls_feedback_total", consumer=consumer_id)
        return True

    def _refusal(self, query_id: str, consumer_id: str, at: Timestamp) -> str | None:
        held = self._served.get(query_id)
        if held is None:
            return "no_such_query"
        served_to, arrived = held
        if served_to != consumer_id:
            return "another_consumer"
        if query_id in self._labeled:
            return "already_labeled"
        if at < arrived:
            return "before_the_query_arrived"
        return None
