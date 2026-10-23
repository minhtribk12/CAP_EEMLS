from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

from eemls.config.reading import ConfigError
from eemls.ensemble_optimization.approach_loading import StartupError
from eemls.messaging.broker import Factory, client_of
from eemls.messaging.result_publisher import ResultPublisher
from eemls.metrics_database import MissingPasswordError, password_from_environment
from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls.metrics_database.schema import check_every_scored_cdm_can_be_computed
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.metrics_reporting import MetricsReporter, ReportingDatabase
from eemls.observation_service.operating_conditions import ConditionLookup, ConditionLookupFactory
from eemls.observation_service.profile_updates import ProfileUpdatesClient
from eemls.observation_service.runtime_context import RuntimeContextReader
from eemls.orchestration.result_delivery import ResultDelivery
from eemls_control_plane.config import ControlPlaneView, load_control_plane_config
from eemls_control_plane.http_app import build_app
from eemls_control_plane.startup import build_control_plane

logger = logging.getLogger(__name__)


def load_checked_config(directory: Path | None) -> ControlPlaneView:
    config = load_control_plane_config(directory)
    check_every_scored_cdm_can_be_computed(config.consumers)
    return config


def monitoring_of(config: ControlPlaneView) -> Monitoring:
    return Monitoring.from_config(config.services.monitoring)


def database_of(
    config: ControlPlaneView, *, password: str, run_id: str, monitoring: Monitoring
) -> PostgresDatabase:
    return PostgresDatabase.from_config(
        config.services.metrics_database,
        password=password,
        run_id=run_id,
        monitoring=monitoring,
    )


def publisher_of(
    config: ControlPlaneView, *, instance_id: str, factory: Factory = client_of
) -> ResultPublisher:
    settings = config.services.broker
    return ResultPublisher(
        broker=settings,
        client_id=f"{settings.client_id_prefix}-{instance_id}",
        topic_prefix=config.services.control_plane.result_topic_prefix,
        factory=factory,
    )


def metrics_reporter_of(
    config: ControlPlaneView,
    *,
    instance_id: str,
    run_id: str,
    monitoring: Monitoring,
    transport: httpx.BaseTransport | None = None,
) -> MetricsReporter:
    database = config.services.metrics_database
    return MetricsReporter(
        base_url=str(config.services.observation_service.url),
        run_id=run_id,
        sender=instance_id,
        monitoring=monitoring,
        buffer_rows=database.buffer_rows,
        flush_interval_ms=float(database.flush_interval_ms),
        request_timeout_ms=float(config.services.observation_service.request_timeout_ms),
        transport=transport,
    )


def profile_updates_of(
    config: ControlPlaneView,
    database: ReportingDatabase,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ProfileUpdatesClient:
    return ProfileUpdatesClient(
        database=database,
        base_url=str(config.services.observation_service.url),
        interval_s=config.services.control_plane.profile_update_interval_s,
        request_timeout_ms=float(config.services.observation_service.request_timeout_ms),
        transport=transport,
    )


def condition_lookup_of(
    config: ControlPlaneView, *, factory: ConditionLookupFactory = ConditionLookup
) -> ConditionLookup:
    settings = config.services.control_plane
    return factory(
        base_url=str(settings.data_hub_url), timeout_ms=float(settings.member_timeout_ms)
    )


def runtime_context_reader_of(config: ControlPlaneView) -> RuntimeContextReader:
    return RuntimeContextReader(dominant_classes=config.datasets.dominant_classes)


def application(
    config: ControlPlaneView, *, instance_id: str, run_id: str, password: str, seed: int
) -> FastAPI:
    monitoring = monitoring_of(config)
    stored = database_of(config, password=password, run_id=run_id, monitoring=monitoring)
    stored.apply_schema()
    database = ReportingDatabase(
        stored,
        metrics_reporter_of(config, instance_id=instance_id, run_id=run_id, monitoring=monitoring),
    )
    profiles = profile_updates_of(config, database)
    broker = publisher_of(config, instance_id=instance_id)
    broker.start()
    conditions = condition_lookup_of(config)
    serving = build_control_plane(
        config,
        database=database,
        instance_id=instance_id,
        monitoring=monitoring,
        publishing=ResultDelivery(broker),
        runtime_context=runtime_context_reader_of(config),
        condition_of=conditions,
        seed=seed,
        profiles=profiles,
    )
    app = build_app(serving, monitoring)

    @contextlib.asynccontextmanager
    async def stopping(_: FastAPI) -> AsyncIterator[None]:
        yield
        await serving.drain()
        await conditions.close()
        broker.close()
        profiles.close()
        database.close()
        logger.info(
            f"stage_tag=stopped process_tag=control_plane instance_id={instance_id} "
            f"results_not_sent={broker.unsent_on_close}"
        )

    app.router.lifespan_context = stopping
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-control-plane")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--instance", default=os.environ.get("EEMLS_INSTANCE_ID", "control-plane-1")
    )
    parser.add_argument("--run", default=os.environ.get("EEMLS_RUN_ID", "unnamed"))
    parser.add_argument("--seed", type=int, default=os.environ.get("EEMLS_SEED") or None)
    args = parser.parse_args(argv)
    if args.seed is None:
        print("refused: EEMLS_SEED is not set and no --seed is given", file=sys.stderr)
        return 1
    try:
        password = password_from_environment()
    except MissingPasswordError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    try:
        config = load_checked_config(args.config)
        app = application(
            config,
            instance_id=args.instance,
            run_id=args.run,
            password=password,
            seed=args.seed,
        )
    except (ConfigError, StartupError, DatabaseUnavailableError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    settings = config.services.control_plane
    logger.info(
        f"stage_tag=serving process_tag=control_plane instance_id={args.instance} "
        f"run_id={args.run} port={settings.port}"
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
