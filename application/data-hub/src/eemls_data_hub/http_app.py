from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict

from eemls.domain.types import DEFAULT_CONDITION, DEFAULT_IMAGE_QUALITY
from eemls.interfaces.data_hub_api import ImageMetadata
from eemls.monitoring.metrics import Monitoring
from eemls.monitoring.metrics_endpoint import service_app
from eemls_data_hub.image_store import ImageStore, NoSuchImageError, StoredImage, StoreError

logger = logging.getLogger(__name__)
JPEG = "image/jpeg"


class Listing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    condition: str
    image_quality: str
    image_ids: tuple[str, ...]


def build_app(store: ImageStore, monitoring: Monitoring) -> FastAPI:
    app = service_app("eemls data hub", monitoring, lambda: len(store) > 0)

    @app.get("/images/{image_id}/meta", response_model=ImageMetadata)
    def meta(image_id: str) -> ImageMetadata:
        held = _held_or_404(store, image_id, monitoring)
        return ImageMetadata(
            image_id=held.image_id,
            label=held.label,
            dataset=held.dataset,
            original_label=held.original_label,
            condition=held.condition,
            image_quality=held.image_quality,
        )

    @app.get("/images/{image_id}")
    def image(image_id: str) -> Response:
        _held_or_404(store, image_id, monitoring)
        try:
            body = store.bytes_of(image_id)
        except StoreError as refused:
            monitoring.count("eemls_images_served_total", outcome="failed")
            logger.error(
                f"error_type=image_unreadable image_id={image_id} "
                f"recovery_action=none detail={refused}"
            )
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(refused)) from refused
        monitoring.count("eemls_images_served_total", outcome="accepted")
        return Response(content=body, media_type=JPEG)

    @app.get("/images", response_model=Listing)
    def listing(
        label: str = Query(...),
        condition: str = Query(DEFAULT_CONDITION),
        image_quality: str = Query(DEFAULT_IMAGE_QUALITY),
    ) -> Listing:
        return Listing(
            label=label,
            condition=condition,
            image_quality=image_quality,
            image_ids=tuple(
                store.ids(label=label, condition=condition, image_quality=image_quality)
            ),
        )

    @app.get("/labels")
    def labels() -> dict[str, tuple[str, ...]]:
        return {"labels": tuple(store.labels())}

    @app.get("/conditions")
    def conditions() -> dict[str, tuple[str, ...]]:
        return {"conditions": tuple(store.conditions())}

    return app


def _held_or_404(store: ImageStore, image_id: str, monitoring: Monitoring) -> StoredImage:
    try:
        return store.meta(image_id)
    except NoSuchImageError as refused:
        monitoring.count("eemls_images_served_total", outcome="refused")
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(refused)) from refused
