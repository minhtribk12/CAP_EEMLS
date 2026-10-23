from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from eemls.image_catalogue import Catalogue
from eemls.interfaces.control_plane_api import ClassifyRequest, ReturnedLabel
from eemls.messaging.result_listener import Delivered
from eemls.monitoring.metrics import Monitoring
from eemls_consumer.feedback import LabelsToReturn
from eemls_consumer.query_schedule import Schedule, sampled_label
from eemls_consumer.results import Pending

logger = logging.getLogger(__name__)


class QuerySender:
    def __init__(
        self,
        *,
        consumer_id: str,
        reply_topic: str,
        schedule: Schedule,
        catalogue: Catalogue,
        pending: Pending,
        labels_to_return: LabelsToReturn,
        control_plane: httpx.Client,
        send_window: int,
        feedback_batch: int,
        decision_interval_queries: int,
        draw: Callable[[], float],
        identifiers: Callable[[], str],
        monitoring: Monitoring,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._consumer_id = consumer_id
        self._reply_topic = reply_topic
        self._schedule = schedule
        self._catalogue = catalogue
        self._pending = pending
        self._labels_to_return = labels_to_return
        self._control_plane = control_plane
        self._send_window = send_window
        self._feedback_batch = feedback_batch
        self._decision_interval_queries = decision_interval_queries
        self._draw = draw
        self._identifiers = identifiers
        self._monitoring = monitoring
        self._clock = clock
        self._started_at = clock()
        self._called_for = 0
        self.accepted = 0

    def begin(self) -> None:
        self._started_at = self._clock()

    def elapsed(self) -> float:
        return self._clock() - self._started_at

    def tick(self) -> int:
        due = self._schedule.due_by(self.elapsed()) - self._called_for
        sent = 0
        for _ in range(due):
            self._called_for += 1
            request, label = self._next_request(self._schedule.planned_at(self._called_for))
            kept = self._labels_to_return.kept()
            if self._pending.in_flight >= self._send_window:
                self._monitoring.count("eemls_client_dropped_total", consumer=self._consumer_id)
                continue
            accepted_before = self.accepted
            self._send(request, label, kept, self.elapsed())
            sent += 1
            if self.accepted > accepted_before and (
                self.accepted % self._decision_interval_queries == 0
            ):
                self.returned()
        return sent

    def result(self, delivered: Delivered) -> None:
        answered = self._pending.arrived(
            correlation_id=delivered.correlation_id, at_s=self.elapsed()
        )
        self._monitoring.count(
            "eemls_client_results_total", consumer=self._consumer_id, result=answered.outcome
        )
        if answered.round_trip is None:
            logger.warning(
                f"stage_tag=result process_tag=consumer_client outcome={answered.outcome} "
                f"consumer={self._consumer_id} topic={delivered.topic}"
            )
            return
        self._monitoring.observe(
            "eemls_client_round_trip_seconds",
            seconds=answered.round_trip.round_trip_ms / 1000.0,
            consumer=self._consumer_id,
        )

    def expire(self) -> int:
        gone = self._pending.expired(now_s=self.elapsed())
        for _ in gone:
            self._monitoring.count(
                "eemls_client_results_total", consumer=self._consumer_id, result="lost"
            )
        return len(gone)

    def returned(self) -> int:
        return self._handed_back(self._labels_to_return.due(after_queries=self.accepted))

    def returned_the_rest(self) -> int:
        return self._handed_back(self._labels_to_return.rest())

    def _handed_back(self, due: tuple[ReturnedLabel, ...]) -> int:
        handed = 0
        for start in range(0, len(due), self._feedback_batch):
            batch = due[start : start + self._feedback_batch]
            if not self._hand_over(batch):
                self._lost_labels(len(due) - handed)
                return handed
            handed += len(batch)
        return handed

    def close(self) -> None:
        self._control_plane.close()

    def _next_request(self, planned_s: float) -> tuple[ClassifyRequest, str]:
        quality = self._schedule.image_quality_at(planned_s)
        period = self._schedule.period_at(planned_s)
        label = sampled_label(period.class_mix, self._draw())
        request = ClassifyRequest(
            consumer_id=self._consumer_id,
            image_id=self._catalogue.pick(
                label=label,
                condition=period.condition,
                image_quality=quality,
                draw=self._draw(),
            ),
            reply_topic=self._reply_topic,
            correlation_id=self._identifiers(),
            image_quality=quality,
            time_of_day=self._schedule.time_of_day_at(planned_s),
        )
        return request, label

    def _send(self, request: ClassifyRequest, label: str, kept: bool, now: float) -> None:
        self._pending.sent(correlation_id=request.correlation_id, at_s=now)
        try:
            answer = self._control_plane.post("/classify", json=request.model_dump(mode="json"))
        except httpx.HTTPError as refused:
            self._not_sent(request, f"the control plane did not answer: {refused}")
            return
        if answer.status_code == httpx.codes.SERVICE_UNAVAILABLE:
            self._pending.withdrawn(correlation_id=request.correlation_id)
            self._monitoring.count(
                "eemls_client_requests_total", consumer=self._consumer_id, outcome="refused"
            )
            return
        if answer.status_code != httpx.codes.ACCEPTED:
            self._not_sent(request, f"the control plane answered {answer.status_code}")
            return
        self._accept(request, label, kept, now, answer)

    def _accept(
        self,
        request: ClassifyRequest,
        label: str,
        kept: bool,
        now: float,
        answer: httpx.Response,
    ) -> None:
        try:
            body = answer.json()
        except ValueError:
            self._not_sent(request, "the control plane accepted without a readable answer")
            return
        query_id = body.get("query_id") if isinstance(body, dict) else None
        if not isinstance(query_id, str) or not query_id:
            self._not_sent(request, "the control plane accepted without a query id")
            return
        self._labels_to_return.recorded(
            query_id=query_id, true_label=label, at_query=self.accepted, kept=kept
        )
        self.accepted += 1
        self._monitoring.count(
            "eemls_client_requests_total", consumer=self._consumer_id, outcome="accepted"
        )

    def _not_sent(self, request: ClassifyRequest, detail: str) -> None:
        self._pending.withdrawn(correlation_id=request.correlation_id)
        self._monitoring.count(
            "eemls_client_requests_total", consumer=self._consumer_id, outcome="failed"
        )
        logger.error(
            f"stage_tag=send process_tag=consumer_client error_type=classify_failed "
            f"consumer={self._consumer_id} recovery_action=query_dropped detail={detail}"
        )

    def _hand_over(self, batch: tuple[ReturnedLabel, ...]) -> bool:
        body = {
            "consumer_id": self._consumer_id,
            "labels": [one.model_dump(mode="json") for one in batch],
        }
        try:
            answer = self._control_plane.post("/feedback", json=body)
        except httpx.HTTPError as refused:
            logger.error(
                f"stage_tag=feedback process_tag=consumer_client error_type=feedback_not_taken "
                f"consumer={self._consumer_id} recovery_action=labels_dropped detail={refused}"
            )
            return False
        if answer.status_code // 100 != 2:
            logger.error(
                f"stage_tag=feedback process_tag=consumer_client error_type=feedback_refused "
                f"consumer={self._consumer_id} recovery_action=labels_dropped "
                f"detail=the control plane answered {answer.status_code}"
            )
            return False
        self._monitoring.count(
            "eemls_feedback_total", amount=len(batch), consumer=self._consumer_id
        )
        return True

    def _lost_labels(self, count: int) -> None:
        self._monitoring.count(
            "eemls_feedback_ignored_total",
            amount=count,
            consumer=self._consumer_id,
            reason="not_taken",
        )
        logger.error(
            f"stage_tag=feedback process_tag=consumer_client error_type=labels_dropped "
            f"consumer={self._consumer_id} labels={count} recovery_action=none"
        )
