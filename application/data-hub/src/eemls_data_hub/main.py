"""`eemls-data-hub`: serves the images."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from eemls.config.reading import ConfigError
from eemls.monitoring.metrics import Monitoring
from eemls_data_hub.config import DataHubView, load_data_hub_config
from eemls_data_hub.http_app import build_app
from eemls_data_hub.image_store import ImageStore, StoreError

logger = logging.getLogger(__name__)


def build_data_hub(argv: list[str] | None = None) -> tuple[FastAPI, DataHubView]:
    parser = argparse.ArgumentParser(prog="eemls-data-hub")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_data_hub_config(args.config)
    settings = config.services.data_hub
    root = args.images or Path(settings.image_root)
    store = ImageStore.opened(root, max_image_bytes=settings.max_image_bytes)
    monitoring = Monitoring.from_config(config.services.monitoring)
    logger.info(
        f"stage_tag=started process_tag=data_hub images={len(store)} "
        f"classes={list(store.labels())} root={root}"
    )
    return build_app(store, monitoring), config


def main(argv: list[str] | None = None) -> int:
    try:
        app, config = build_data_hub(argv)
    except (ConfigError, StoreError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    settings = config.services.data_hub
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
