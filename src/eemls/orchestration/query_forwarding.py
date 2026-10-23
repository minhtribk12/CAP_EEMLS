from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from eemls.domain.cost import Charge, DeviceRate, charge_of
from eemls.domain.measuring import Measuring, milliseconds_since
from eemls.domain.records import ServedQuery
from eemls.domain.types import (
    AggregatedResult,
    EnsembleMember,
    Identifier,
    MemberPrediction,
    MemberStatus,
    Milliseconds,
    NumberFormat,
    PositiveCount,
    Query,
    Ratio,
    RuntimeContext,
)
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.database import MetricsDatabase
from eemls.observation_service.idle_service_profiling import IdleServiceProfiling
from eemls.observation_service.operating_conditions import Conditioning
from eemls.orchestration.aggregation import AggregationError, aggregate_predictions
from eemls.orchestration.ensemble_deployments import EnsembleDeployments

logger = logging.getLogger(__name__)

Publishing = Callable[[Query, AggregatedResult], None]
Contexting = Callable[[str], RuntimeContext]


class MemberAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    query_id: str | None = None
    instance_id: Identifier
    service_name: Identifier
    node: Identifier
    replica: PositiveCount
    predicted_label: Identifier
    confidence: Ratio
    fetch_ms: Milliseconds
    inference_ms: Milliseconds
    total_ms: Milliseconds
    execution_provider: Identifier
    number_format: NumberFormat
    cache_hit: bool

    @model_validator(mode="after")
    def _the_parts_fit_inside_the_whole(self) -> MemberAnswer:
        if self.fetch_ms + self.inference_ms > self.total_ms + 1e-6:
            raise ValueError(
                f"the fetch and the run take {self.fetch_ms + self.inference_ms} ms, "
                f"longer than the {self.total_ms} ms they sit in"
            )
        if self.inference_ms <= 0.0:
            raise ValueError("the member answered but the model took no time to run")
        return self


@dataclass(frozen=True, slots=True)
class MemberOutcome:
    member: EnsembleMember
    status: MemberStatus
    waited_ms: float
    answer: MemberAnswer | None
    in_ensemble: bool


class QueryForwarding:
    def __init__(
        self,
        *,
        deployments: EnsembleDeployments,
        database: MetricsDatabase,
        monitoring: Measuring,
        rates: Mapping[str, DeviceRate],
        runtime_context: Contexting,
        condition_of: Conditioning,
        publishing: Publishing,
        member_port: int,
        member_timeout_ms: float,
        transport: httpx.AsyncBaseTransport | None = None,
        probe_fraction: float,
        max_connections: int,
    ) -> None:
        self._deployments = deployments
        self._database = database
        self._monitoring = monitoring
        self._rates = rates
        self._runtime_context = runtime_context
        self._condition_of = condition_of
        self._publishing = publishing
        self._member_port = member_port
        self._budget_s = member_timeout_ms / 1000.0
        self._client = httpx.AsyncClient(
            timeout=self._budget_s,
            transport=transport,
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )
        self._probing = IdleServiceProfiling(
            database=database, fraction=probe_fraction, asking=self._outside
        )

    async def answer(self, query: Query, *, arrived_at: float) -> AggregatedResult:
        """Call every member and return the result; `arrived_at` is a perf_counter reading."""
        deployment = self._deployments.current(query.consumer_id)
        context = await self._in_context(query)
        handing = time.perf_counter()
        calls = [self._ask(query, one) for one in deployment.members]
        self._probe(query)
        reported = await asyncio.gather(*calls)
        wait_ms = milliseconds_since(handing)
        voting = time.perf_counter()
        rows = tuple(self._row(query, one) for one in reported)
        charge = charge_of(
            [(one.member.device, one.total_ms) for one in rows if one.status == "ok"], self._rates
        )
        try:
            result = aggregate_predictions(
                query_id=query.query_id,
                deployment_id=deployment.deployment_id,
                n_members=len(deployment.members),
                predictions=rows,
            )
        except AggregationError:
            await self._store(
                _served(
                    query,
                    context,
                    charge,
                    wait_ms,
                    milliseconds_since(voting),
                    result=None,
                    ready=_at(query, arrived_at),
                    published=None,
                ),
                rows,
            )
            raise
        aggregate_ms = milliseconds_since(voting)
        ready = _at(query, arrived_at)
        self._monitoring.observe(
            "eemls_response_time_seconds",
            seconds=(ready - query.t_arrive).total_seconds(),
            consumer=query.consumer_id,
        )
        published = self._publish(query, result, arrived_at)
        await self._store(
            _served(
                query,
                context,
                charge,
                wait_ms,
                aggregate_ms,
                result=result,
                ready=ready,
                published=published,
            ),
            rows,
        )
        return result

    async def drain(self) -> None:
        await self._probing.drain()

    async def close(self) -> None:
        await self.drain()
        await self._client.aclose()

    def _probe(self, query: Query) -> None:
        if not self._probing.due(query.consumer_id):
            return
        serving = {
            name
            for consumer_id in self._deployments.consumers()
            for name in self._deployments.current(consumer_id).service_names
        }
        self._probing.start(
            query,
            [
                one
                for one in self._deployments.available_inference_services
                if one.service_name not in serving
            ],
        )

    async def _outside(
        self, query: Query, idle: Sequence[EnsembleMember]
    ) -> tuple[MemberPrediction, ...]:
        reported = await asyncio.gather(*(self._ask(query, one, in_ensemble=False) for one in idle))
        return tuple(self._row(query, one) for one in reported)

    async def _in_context(self, query: Query) -> RuntimeContext:
        held = self._runtime_context(query.consumer_id)
        return held.model_copy(update={"condition": await self._condition_of(query.image_id)})

    async def _ask(
        self, query: Query, member: EnsembleMember, *, in_ensemble: bool = True
    ) -> MemberOutcome:
        asked = time.perf_counter()
        status, answer = await self._reply(query, member)
        waited_ms = milliseconds_since(asked)
        called = "true" if in_ensemble else "false"
        self._monitoring.observe(
            "eemls_member_call_seconds",
            seconds=waited_ms / 1000.0,
            model=member.model,
            device=member.device,
            in_ensemble=called,
        )
        self._monitoring.count(
            "eemls_member_calls_total",
            model=member.model,
            device=member.device,
            status=status,
            in_ensemble=called,
        )
        return MemberOutcome(
            member=member,
            status=status,
            waited_ms=waited_ms,
            answer=answer,
            in_ensemble=in_ensemble,
        )

    async def _reply(
        self, query: Query, member: EnsembleMember
    ) -> tuple[MemberStatus, MemberAnswer | None]:
        url = f"http://{member.service_name}:{self._member_port}/predict"
        try:
            with anyio.fail_after(self._budget_s):
                answered = await self._client.post(
                    url, json={"image_id": query.image_id, "query_id": query.query_id}
                )
        except (httpx.TimeoutException, TimeoutError) as late:
            return self._unheard(member, query, "timeout", late)
        except (httpx.HTTPError, httpx.InvalidURL, RuntimeError) as refused:
            return self._unheard(member, query, "error", refused)
        if answered.status_code == httpx.codes.SERVICE_UNAVAILABLE:
            return self._unheard(member, query, "refused", answered.status_code)
        if answered.status_code != httpx.codes.OK:
            return self._unheard(member, query, "error", answered.status_code)
        try:
            answer = MemberAnswer.model_validate(answered.json())
        except (ValidationError, ValueError) as unreadable:
            return self._unheard(member, query, "error", unreadable)
        return self._addressed(query, member, answer)

    def _addressed(
        self, query: Query, member: EnsembleMember, answer: MemberAnswer
    ) -> tuple[MemberStatus, MemberAnswer | None]:
        if answer.service_name != member.service_name:
            return self._unheard(
                member, query, "error", f"{answer.service_name} answered for {member.service_name}"
            )
        if answer.query_id is not None and answer.query_id != query.query_id:
            return self._unheard(member, query, "error", f"the answer belongs to {answer.query_id}")
        return "ok", answer

    def _unheard(
        self, member: EnsembleMember, query: Query, status: MemberStatus, detail: object
    ) -> tuple[MemberStatus, None]:
        logger.warning(
            f"stage_tag=member_call process_tag=gateway error_type={status} "
            f"service_name={member.service_name} query_id={query.query_id} "
            f"recovery_action=the_ensemble_answers_without_this_member detail={detail}"
        )
        return status, None

    def _row(self, query: Query, reported: MemberOutcome) -> MemberPrediction:
        answer = reported.answer
        if answer is None:
            return MemberPrediction(
                query_id=query.query_id,
                member=reported.member,
                in_ensemble=reported.in_ensemble,
                predicted_label=None,
                confidence=None,
                fetch_ms=0.0,
                inference_ms=0.0,
                total_ms=reported.waited_ms,
                status=reported.status,
            )
        return MemberPrediction(
            query_id=query.query_id,
            instance_id=answer.instance_id,
            member=reported.member,
            node=answer.node,
            replica=answer.replica,
            in_ensemble=reported.in_ensemble,
            predicted_label=answer.predicted_label,
            confidence=answer.confidence,
            fetch_ms=answer.fetch_ms,
            inference_ms=answer.inference_ms,
            total_ms=answer.total_ms,
            provider=answer.execution_provider,
            number_format=answer.number_format,
            cache_hit=answer.cache_hit,
            status=reported.status,
        )

    def _publish(self, query: Query, result: AggregatedResult, arrived: float) -> datetime | None:
        with self._monitoring.timer("eemls_publish_seconds", consumer=query.consumer_id):
            try:
                self._publishing(query, result)
            except Exception as refused:
                logger.error(
                    f"stage_tag=publish process_tag=gateway error_type=result_not_published "
                    f"query_id={query.query_id} detail_type={type(refused).__name__} "
                    f"recovery_action=row_is_still_stored detail={refused}"
                )
                return None
        return _at(query, arrived)

    async def _store(self, served: ServedQuery, rows: Sequence[MemberPrediction]) -> None:
        self._monitoring.count(
            "eemls_queries_total", consumer=served.consumer_id, outcome=served.outcome
        )
        await asyncio.to_thread(self._write_rows, served, rows)

    def _write_rows(self, served: ServedQuery, rows: Sequence[MemberPrediction]) -> None:
        try:
            self._database.write_query(served)
            self._database.write_member_predictions(rows)
        except WriteError as refused:
            logger.error(
                f"stage_tag=store process_tag=gateway error_type=rows_not_written "
                f"query_id={served.query_id} recovery_action=the_store_holds_them "
                f"detail={refused}"
            )


def member_connections(
    *,
    max_concurrent_queries_per_consumer: int,
    consumers: int,
    max_ensemble_size: int,
    probe_fraction: float,
    available: int,
) -> int:
    queries = max_concurrent_queries_per_consumer * consumers
    return queries * max_ensemble_size + math.ceil(queries * probe_fraction) * available


def _at(query: Query, arrived: float) -> datetime:
    return query.t_arrive + timedelta(milliseconds=milliseconds_since(arrived))


def _served(
    query: Query,
    context: RuntimeContext,
    charge: Charge | None,
    wait_ms: float,
    aggregate_ms: float,
    *,
    result: AggregatedResult | None,
    ready: datetime,
    published: datetime | None,
) -> ServedQuery:
    return ServedQuery(
        query_id=query.query_id,
        consumer_id=query.consumer_id,
        image_id=query.image_id,
        dataset=query.dataset,
        image_quality=query.image_quality,
        time_of_day=query.time_of_day,
        t_arrive=query.t_arrive,
        t_result_ready=ready,
        wait_ms=wait_ms,
        aggregate_ms=aggregate_ms,
        t_publish=published,
        context=context,
        deployment_id=result.deployment_id if result else None,
        n_members=result.n_members if result else None,
        n_answered=result.n_answered if result else None,
        predicted_label=result.predicted_label if result else None,
        predicted_confidence=result.confidence if result else None,
        cost_per_1k=charge.cost_per_1k if charge else None,
        throughput_rps=charge.throughput_rps if charge else None,
        outcome="accepted" if result is not None else "failed",
    )
