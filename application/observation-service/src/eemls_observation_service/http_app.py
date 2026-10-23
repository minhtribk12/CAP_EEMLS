from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

import psycopg
from fastapi import FastAPI, HTTPException, status

from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.database import MetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.monitoring.metrics_endpoint import service_app
from eemls.observation_service.metrics_storage import store_metrics
from eemls.observation_service.observation_service_api import (
    ProfileUpdateAnswer,
    ProfileUpdateRequest,
    RowBatch,
    Stored,
)
from eemls.observation_service.profile_updates import ProfileSettings, update_profiles

logger = logging.getLogger(__name__)

REFUSED = (WriteError, psycopg.Error)


def build_app(
    *,
    database: MetricsDatabase,
    run_id: str,
    settings_at: Callable[[datetime], ProfileSettings],
    monitoring: Monitoring,
) -> FastAPI:
    app = service_app("eemls observation service", monitoring, lambda: True)
    lock = threading.Lock()

    @app.post("/rows", response_model=Stored)
    def rows(batch: RowBatch) -> Stored:
        if batch.run_id != run_id:
            logger.error(
                f"stage_tag=store process_tag=observation_service error_type=another_run "
                f"run_id={batch.run_id} sender={batch.sender} recovery_action=refused"
            )
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"this service stores run {run_id}, not {batch.run_id}"
            )
        try:
            with lock:
                stored = store_metrics(batch, database)
        except REFUSED as refused:
            logger.error(
                f"stage_tag=store process_tag=observation_service error_type=rows_not_written "
                f"sender={batch.sender} rows={batch.rows} recovery_action=the_sender_holds_them "
                f"detail={refused}"
            )
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(refused)) from refused
        return Stored(stored=stored)

    @app.post("/profiles/update", response_model=ProfileUpdateAnswer)
    def update(request: ProfileUpdateRequest) -> ProfileUpdateAnswer:
        try:
            with lock:
                updated = update_profiles(database, settings_at(request.at))
        except REFUSED as refused:
            logger.error(
                f"stage_tag=profile process_tag=observation_service "
                f"error_type=profile_rebuild_failed at={request.at.isoformat()} "
                f"recovery_action=the_held_profiles_are_kept detail={refused}"
            )
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(refused)) from refused
        return ProfileUpdateAnswer(profiles=updated)

    return app
