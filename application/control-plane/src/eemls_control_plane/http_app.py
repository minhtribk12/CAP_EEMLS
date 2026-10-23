from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi import Query as QueryParam
from pydantic import BaseModel, ConfigDict

from eemls.domain.measuring import Measuring
from eemls.domain.records import Decision
from eemls.domain.types import EnsembleDeployment, Query
from eemls.domain.waiting import all_finished
from eemls.ensemble_optimization.decision_loop import DecisionLoop, utc_now
from eemls.interfaces.control_plane_api import ClassifyRequest, ReturnedLabel
from eemls.metrics_database.database import MetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.monitoring.metrics_endpoint import service_app
from eemls.observation_service.consumer_feedback import ReturnedLabels
from eemls.orchestration.ensemble_deployments import EnsembleDeployments, NoSuchConsumerError
from eemls.orchestration.query_forwarding import QueryForwarding

logger = logging.getLogger(__name__)

MOST_DECISIONS = 500


def identifiers() -> str:
    return uuid.uuid4().hex


class Accepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str


class FeedbackBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consumer_id: str
    labels: tuple[ReturnedLabel, ...]


class FeedbackAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kept: int
    ignored: int


class EnsembleView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consumer_id: str
    deployment_id: str
    n_members: int
    members: tuple[MemberView, ...]


class MemberView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str
    model: str
    device: str


class Decisions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: tuple[Decision, ...]


class ControlPlaneEndpoints:
    def __init__(
        self,
        *,
        instance_id: str,
        query_forwarding: QueryForwarding,
        deployments: EnsembleDeployments,
        database: MetricsDatabase,
        orchestrator: DecisionLoop,
        monitoring: Measuring,
        max_concurrent_queries_per_consumer: int,
        decision_interval_queries: int,
        remember_served: int,
        now: Callable[[], datetime] = utc_now,
        identifiers: Callable[[], str] = identifiers,
    ) -> None:
        self.instance_id = instance_id
        self._query_forwarding = query_forwarding
        self._deployments = deployments
        self._database = database
        self._orchestrator = orchestrator
        self._monitoring = monitoring
        self._max_concurrent_queries_per_consumer = max_concurrent_queries_per_consumer
        self._interval = decision_interval_queries
        self._now = now
        self._identifiers = identifiers
        self._returned_labels = ReturnedLabels(
            database, remember=remember_served, monitoring=monitoring
        )
        self._in_flight: dict[str, int] = {}
        self._since_decision: dict[str, int] = {}
        self._answering: set[asyncio.Task[None]] = set()
        self._deciding_for: set[str] = set()

    def serves(self, consumer_id: str) -> bool:
        return consumer_id in self._deployments.consumers()

    def ensemble(self, consumer_id: str) -> EnsembleDeployment:
        return self._deployments.current(consumer_id)

    def accept(self, asked: ClassifyRequest) -> Query | None:
        """Accept one query and stamp its arrival, or None at the consumer's limit."""
        held = self._in_flight.get(asked.consumer_id, 0)
        if held >= self._max_concurrent_queries_per_consumer:
            self._monitoring.count(
                "eemls_queries_total", consumer=asked.consumer_id, outcome="refused"
            )
            return None
        arrived = time.perf_counter()
        query = Query(
            query_id=self._identifiers(),
            consumer_id=asked.consumer_id,
            image_id=asked.image_id,
            reply_topic=asked.reply_topic,
            correlation_id=asked.correlation_id,
            t_arrive=self._now(),
            image_quality=asked.image_quality,
            time_of_day=asked.time_of_day,
        )
        self._returned_labels.arrived(query)
        self._hold(asked.consumer_id, held + 1)
        answering = asyncio.create_task(self._answer(query, arrived))
        self._answering.add(answering)
        answering.add_done_callback(self._answering.discard)
        return query

    def label(self, batch: FeedbackBatch) -> FeedbackAccepted:
        at = self._now()
        kept = sum(
            self._returned_labels.label(
                one.query_id,
                consumer_id=batch.consumer_id,
                true_label=one.true_label,
                at=at,
            )
            for one in batch.labels
        )
        return FeedbackAccepted(kept=kept, ignored=len(batch.labels) - kept)

    def decide(self, consumer_id: str) -> None:
        try:
            self._orchestrator.decide(consumer_id)
        except Exception as refused:
            self._monitoring.count("eemls_decision_failures_total", consumer=consumer_id)
            logger.error(
                f"stage_tag=decide process_tag=control_plane error_type=decision_failed "
                f"consumer={consumer_id} recovery_action=the_run_keeps_serving "
                f"detail_type={type(refused).__name__} detail={refused}"
            )

    def decisions(self, limit: int) -> Decisions:
        return Decisions(decisions=tuple(self._database.read_decisions(limit)))

    async def drain(self) -> None:
        await all_finished(self._answering)
        await self._query_forwarding.drain()

    async def _answer(self, query: Query, arrived: float) -> None:
        try:
            await self._query_forwarding.answer(query, arrived_at=arrived)
        except Exception as refused:
            logger.error(
                f"stage_tag=answer process_tag=control_plane error_type=query_not_answered "
                f"consumer={query.consumer_id} query_id={query.query_id} "
                f"recovery_action=the_row_is_stored_without_a_result "
                f"detail_type={type(refused).__name__} detail={refused}"
            )
        finally:
            self._done(query.consumer_id)
        if self._due(query.consumer_id):
            await self.decide_when_free(query.consumer_id)

    async def decide_when_free(self, consumer_id: str) -> None:
        if consumer_id in self._deciding_for:
            self._monitoring.count("eemls_decisions_skipped_total", consumer=consumer_id)
            return
        self._deciding_for.add(consumer_id)
        try:
            await asyncio.to_thread(self.decide, consumer_id)
        finally:
            self._deciding_for.discard(consumer_id)

    def _due(self, consumer_id: str) -> bool:
        counted = self._since_decision.get(consumer_id, 0) + 1
        if counted < self._interval:
            self._since_decision[consumer_id] = counted
            return False
        self._since_decision[consumer_id] = 0
        return True

    def _hold(self, consumer_id: str, held: int) -> None:
        self._in_flight[consumer_id] = held
        self._monitoring.gauge("eemls_queries_in_progress", value=held, consumer=consumer_id)

    def _done(self, consumer_id: str) -> None:
        held = max(self._in_flight.get(consumer_id, 1) - 1, 0)
        self._hold(consumer_id, held)


def build_app(serving: ControlPlaneEndpoints, monitoring: Monitoring) -> FastAPI:
    app = service_app("eemls control plane", monitoring, lambda: True)

    @app.post("/classify", status_code=status.HTTP_202_ACCEPTED, response_model=Accepted)
    async def classify(asked: ClassifyRequest) -> Accepted:
        _served(serving, asked.consumer_id)
        query = serving.accept(asked)
        if query is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"{asked.consumer_id} is already waiting for all the queries it may have in flight",
            )
        return Accepted(query_id=query.query_id)

    @app.post("/feedback", response_model=FeedbackAccepted)
    async def feedback(batch: FeedbackBatch) -> FeedbackAccepted:
        _served(serving, batch.consumer_id)
        return await asyncio.to_thread(serving.label, batch)

    @app.get("/ensembles/{consumer_id}", response_model=EnsembleView)
    def ensembles(consumer_id: str) -> EnsembleView:
        try:
            deployment = serving.ensemble(consumer_id)
        except NoSuchConsumerError as refused:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(refused)) from refused
        return EnsembleView(
            consumer_id=consumer_id,
            deployment_id=deployment.deployment_id,
            n_members=len(deployment.members),
            members=tuple(
                MemberView(service_name=one.service_name, model=one.model, device=one.device)
                for one in deployment.members
            ),
        )

    @app.get("/decisions", response_model=Decisions)
    async def decisions(
        limit: int = QueryParam(default=100, ge=0, le=MOST_DECISIONS),
    ) -> Decisions:
        return await asyncio.to_thread(serving.decisions, limit)

    return app


def _served(serving: ControlPlaneEndpoints, consumer_id: str) -> None:
    if not serving.serves(consumer_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"this run does not serve {consumer_id}")
