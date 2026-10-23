from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from eemls.config.reading import ConfigError
from eemls.messaging.broker import Factory, client_of
from eemls.messaging.instance_announcements import InstanceAnnouncementListener
from eemls.messaging.topics import instances_of_run
from eemls.metrics_database import MissingPasswordError, password_from_environment
from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.profile_updates import ProfileSettings
from eemls.observation_service.running_instances import InstanceRecords, instance_recorder
from eemls_observation_service.config import (
    ObservationServiceView,
    load_observation_service_config,
)
from eemls_observation_service.http_app import build_app

logger = logging.getLogger(__name__)


def announcement_listener_of(
    config: ObservationServiceView,
    *,
    instance_id: str,
    run_id: str,
    database: InstanceRecords,
    factory: Factory = client_of,
) -> InstanceAnnouncementListener:
    settings = config.services.broker
    return InstanceAnnouncementListener(
        broker=settings,
        client_id=f"{settings.client_id_prefix}-{instance_id}-instances",
        topic_filter=instances_of_run(prefix=settings.instance_topic_prefix, run_id=run_id),
        on_announcement=instance_recorder(database, clock=lambda: datetime.now(UTC)),
        factory=factory,
    )


def profile_settings_of(config: ObservationServiceView) -> Callable[[datetime], ProfileSettings]:
    settings = config.services.observation_service
    rates = config.cost.rates()

    def at(end: datetime) -> ProfileSettings:
        return ProfileSettings.ending_at(
            end,
            window_s=settings.profile_window_s,
            rates=rates,
            min_labeled_per_class=settings.profile_min_labeled_per_class,
            min_labeled_overall=settings.profile_min_labeled_overall,
        )

    return at


def application(
    config: ObservationServiceView,
    *,
    instance_id: str,
    run_id: str,
    password: str,
    factory: Factory = client_of,
) -> FastAPI:
    monitoring = Monitoring.from_config(config.services.monitoring)
    database = PostgresDatabase.from_config(
        config.services.metrics_database,
        password=password,
        run_id=run_id,
        monitoring=monitoring,
    )
    database.apply_schema()
    instances = announcement_listener_of(
        config, instance_id=instance_id, run_id=run_id, database=database, factory=factory
    )
    instances.start()
    app = build_app(
        database=database,
        run_id=run_id,
        settings_at=profile_settings_of(config),
        monitoring=monitoring,
    )

    @contextlib.asynccontextmanager
    async def stopping(_: FastAPI) -> AsyncIterator[None]:
        yield
        instances.close()
        database.close()
        logger.info(f"stage_tag=stopped process_tag=observation_service instance_id={instance_id}")

    app.router.lifespan_context = stopping
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-observation-service")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--instance", default=os.environ.get("EEMLS_INSTANCE_ID", "observation-service-1")
    )
    parser.add_argument("--run", default=os.environ.get("EEMLS_RUN_ID", "unnamed"))
    args = parser.parse_args(argv)
    try:
        password = password_from_environment()
    except MissingPasswordError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    try:
        config = load_observation_service_config(args.config)
        app = application(config, instance_id=args.instance, run_id=args.run, password=password)
    except (ConfigError, DatabaseUnavailableError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    settings = config.services.observation_service
    logger.info(
        f"stage_tag=serving process_tag=observation_service instance_id={args.instance} "
        f"run_id={args.run} port={settings.port}"
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
