from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, FastAPI, Response, status

from eemls.monitoring.metrics import Monitoring


def metrics_router(monitoring: Monitoring) -> APIRouter:
    router = APIRouter()

    @router.get(monitoring.path, include_in_schema=False)
    def metrics() -> Response:
        body, content_type = monitoring.exposition()
        return Response(content=body, media_type=content_type)

    return router


def service_app(title: str, monitoring: Monitoring, ready: Callable[[], bool]) -> FastAPI:
    app = FastAPI(title=title, docs_url=None)
    app.include_router(metrics_router(monitoring))

    @app.get("/healthz")
    def healthz() -> Response:
        return Response(status_code=status.HTTP_200_OK)

    @app.get("/readyz")
    def readyz() -> Response:
        if not ready():
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(status_code=status.HTTP_200_OK)

    return app
