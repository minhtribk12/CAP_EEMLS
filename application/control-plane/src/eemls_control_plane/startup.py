from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime

import httpx

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.records import RunOwner
from eemls.domain.types import EnsembleMember
from eemls.ensemble_optimization.approach import ConsumerSpecification
from eemls.ensemble_optimization.approach_loading import StartupError, policy_of
from eemls.ensemble_optimization.decision_loop import DecisionLoop, utc_now
from eemls.evaluation_engine.consumer_evaluation import window_size_for
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.recent_windows import RecentWindows
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.operating_conditions import Conditioning
from eemls.observation_service.profile_updates import ProfileUpdating
from eemls.observation_service.runtime_context import ServingContext
from eemls.orchestration.ensemble_deployments import EnsembleDeployments
from eemls.orchestration.query_forwarding import Publishing, QueryForwarding, member_connections
from eemls_control_plane.config import ControlPlaneView
from eemls_control_plane.http_app import ControlPlaneEndpoints

logger = logging.getLogger(__name__)


def recent_windows_of(
    database: MetricsDatabase, consumers: Mapping[str, ConsumerSpecification], policy_window: int
) -> RecentWindows:
    keep = max(window_size_for(one.cdms, at_least=policy_window) for one in consumers.values())
    recent = RecentWindows(database, keep=keep)
    recent.seed(consumers)
    return recent


def remembered_queries(config: ControlPlaneView) -> int:
    settings = config.services.control_plane
    longest = max(
        contract.feedback.delay_intervals
        for consumer in config.consumers.values()
        for contract in (consumer.contract, *consumer.contract_variants.values())
    )
    return (
        longest * settings.decision_interval_queries + settings.max_concurrent_queries_per_consumer
    )


def members_of(config: ControlPlaneView) -> tuple[EnsembleMember, ...]:
    return tuple(
        EnsembleMember(service_name=name, model=service.model, device=service.device_type)
        for name, service in sorted(config.models.inference_services.services.items())
    )


def consumers_of(config: ControlPlaneView) -> Mapping[str, ConsumerSpecification]:
    name = config.services.control_plane.contract
    return {
        consumer_id: ConsumerSpecification(
            contract=held.scored_under(name), cdms=CdmSet(definitions=held.cdms)
        )
        for consumer_id, held in config.consumers.items()
    }


def claim_the_run(database: MetricsDatabase, *, instance_id: str, at: datetime) -> None:
    owner = database.claim_run(RunOwner(instance_id=instance_id, claimed_at=at))
    if owner != instance_id:
        raise StartupError(
            f"{owner} already owns this run and holds its ensembles, so {instance_id} "
            "would decide beside it; one instance owns a run"
        )


def check_every_service_is_profiled(config: ControlPlaneView, database: MetricsDatabase) -> None:
    held = {(profile.model, profile.device) for profile in database.read_profiles()}
    unprofiled = sorted(
        name
        for name, service in config.models.inference_services.services.items()
        if (service.model, service.device_type) not in held
    )
    if unprofiled:
        raise StartupError(
            f"no measured profile for {unprofiled}; a service with no profile is never selected"
        )


def register_the_initial_ensembles(
    config: ControlPlaneView, deployments: EnsembleDeployments
) -> None:
    for consumer_id, consumer in sorted(config.consumers.items()):
        if not consumer.initial_ensemble:
            raise StartupError(
                f"consumers/{consumer_id}/initial_ensemble.yaml states no ensemble to start on"
            )
        deployments.register(consumer_id=consumer_id, service_names=consumer.initial_ensemble)


def build_control_plane(
    config: ControlPlaneView,
    *,
    database: MetricsDatabase,
    instance_id: str,
    monitoring: Monitoring,
    publishing: Publishing,
    runtime_context: ServingContext,
    condition_of: Conditioning,
    now: Callable[[], datetime] = utc_now,
    seed: int,
    transport: httpx.AsyncBaseTransport | None = None,
    profiles: ProfileUpdating,
) -> ControlPlaneEndpoints:
    settings = config.services.control_plane
    approach = config.approaches[settings.policy]
    claim_the_run(database, instance_id=instance_id, at=now())
    check_every_service_is_profiled(config, database)
    consumers = consumers_of(config)
    recent = recent_windows_of(database, consumers, approach.evaluation_window)
    pool = members_of(config)
    deployments = EnsembleDeployments(
        available_inference_services=pool,
        max_ensemble_size=approach.max_ensemble_size,
    )
    register_the_initial_ensembles(config, deployments)
    query_forwarding = QueryForwarding(
        deployments=deployments,
        database=recent,
        monitoring=monitoring,
        rates=config.cost.rates(),
        runtime_context=runtime_context,
        condition_of=condition_of,
        publishing=publishing,
        member_port=config.services.inference_service.port,
        member_timeout_ms=float(settings.member_timeout_ms),
        transport=transport,
        probe_fraction=settings.probe_fraction,
        max_connections=member_connections(
            max_concurrent_queries_per_consumer=settings.max_concurrent_queries_per_consumer,
            consumers=len(consumers),
            max_ensemble_size=approach.max_ensemble_size,
            probe_fraction=settings.probe_fraction,
            available=len(pool),
        ),
    )
    serving = ControlPlaneEndpoints(
        instance_id=instance_id,
        query_forwarding=query_forwarding,
        deployments=deployments,
        database=recent,
        orchestrator=DecisionLoop(
            deployments=deployments,
            database=recent,
            policy=policy_of(approach, classes=config.datasets.classes, seed=seed),
            consumers=consumers,
            monitoring=monitoring,
            runtime_context=runtime_context,
            now=now,
            policy_window=approach.evaluation_window,
            profiles=profiles,
        ),
        monitoring=monitoring,
        max_concurrent_queries_per_consumer=settings.max_concurrent_queries_per_consumer,
        decision_interval_queries=settings.decision_interval_queries,
        remember_served=remembered_queries(config),
        now=now,
    )
    logger.info(
        f"stage_tag=bootstrap process_tag=control_plane instance_id={instance_id} "
        f"policy={settings.policy} contract={settings.contract or 'own'} "
        f"consumers={sorted(consumers)} pool={len(pool)}"
    )
    return serving
