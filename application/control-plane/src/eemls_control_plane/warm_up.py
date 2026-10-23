"""`eemls-warm`: send the same images to every pool service before a run.

eemls-warm --images 50 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from eemls.config.reading import ConfigError
from eemls.image_catalogue import Catalogue, CatalogueError
from eemls_control_plane.config import ControlPlaneView, load_control_plane_config

logger = logging.getLogger(__name__)


def warm_up(
    config: ControlPlaneView,
    image_ids: Sequence[str],
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, int]:
    port = config.services.inference_service.port
    timeout_s = config.services.control_plane.member_timeout_ms / 1000.0
    answered: dict[str, int] = {}
    with httpx.Client(transport=transport, timeout=timeout_s) as client:
        for name in sorted(config.models.inference_services.services):
            answered[name] = 0
            for image_id in image_ids:
                try:
                    reply = client.post(
                        f"http://{name}:{port}/predict",
                        json={"image_id": image_id, "query_id": f"warm-{image_id}"},
                    )
                except httpx.HTTPError as refused:
                    logger.warning(
                        f"stage_tag=warm process_tag=warm error_type=warm_request_failed "
                        f"service={name} image_id={image_id} recovery_action=next_image "
                        f"detail={refused}"
                    )
                    continue
                answered[name] += reply.status_code == httpx.codes.OK
    return answered


def drawn_image_ids(
    config: ControlPlaneView, catalogue: Catalogue, *, images: int, seed: int
) -> list[str]:
    draws = random.Random(seed)  # noqa: S311
    return [
        catalogue.draw(dominant_classes=config.datasets.dominant_classes, draws=draws)
        for _ in range(images)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-warm")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        config = load_control_plane_config(args.config)
        with httpx.Client(base_url=str(config.services.control_plane.data_hub_url)) as hub:
            catalogue = Catalogue.read(hub)
        answered = warm_up(
            config, drawn_image_ids(config, catalogue, images=args.images, seed=args.seed)
        )
    except (ConfigError, CatalogueError, httpx.HTTPError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    cold = sorted(name for name, count in answered.items() if count < args.images)
    print(f"warmed {len(answered)} services with {args.images} images; short: {cold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
