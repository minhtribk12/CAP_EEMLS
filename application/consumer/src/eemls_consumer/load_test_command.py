"""`eemls-load`: drive one consumer at rising rates and report each rate.

eemls-load --consumer c3_surveillance --rates 10,20,50,100 --step-s 30 --run-id R
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eemls.config.reading import ConfigError
from eemls.image_catalogue import Catalogue, CatalogueError
from eemls.interfaces.control_plane_api import ClassifyRequest
from eemls.messaging.topics import result_topic
from eemls.metrics_database import MissingPasswordError, password_from_environment
from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls_consumer.config import LoadTestView, load_load_test_config
from eemls_consumer.load_test import (
    Report,
    Step,
    StepOutcome,
    reported,
    run_load_test,
    steps_of,
    table,
)

CLIENT_ID = "load"


def rates_of(written: str) -> list[float]:
    """Rates from a comma-separated list, in requests per second."""
    try:
        rates = [float(one) for one in written.split(",")]
    except ValueError as refused:
        raise ValueError(f"rates must be numbers separated by commas, not {written!r}") from refused
    if not rates:
        raise ValueError("rates name no rate")
    return rates


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eemls-load")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--consumer", required=True)
    parser.add_argument("--rates", required=True, help="requests a second, e.g. 10,20,50,100")
    parser.add_argument("--step-s", type=float, default=30.0)
    parser.add_argument("--run-id", required=True, help="the run the control plane stores under")
    parser.add_argument("--out", type=Path, help="where to write the reports as JSON")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        password = password_from_environment()
    except MissingPasswordError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    try:
        config = load_load_test_config(args.config)
        if args.consumer not in config.consumers:
            raise ConfigError(f"{args.consumer} is not a consumer of this configuration")
        steps = steps_of(rates_of(args.rates), step_s=args.step_s)
        sent = asyncio.run(_sent(config, args.consumer, steps, args.seed))
        reports = _reports(config, password, args.run_id, args.consumer, sent)
    except (
        ConfigError,
        CatalogueError,
        DatabaseUnavailableError,
        ValueError,
        httpx.HTTPError,
    ) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    print(table(reports), end="")
    if args.out is not None:
        args.out.write_text(
            json.dumps([asdict(one) for one in reports], indent=2) + "\n", encoding="utf-8"
        )
    return 0


async def _sent(
    config: LoadTestView, consumer_id: str, steps: tuple[Step, ...], seed: int
) -> list[StepOutcome]:
    services = config.services
    with httpx.Client(base_url=str(services.control_plane.data_hub_url)) as hub:
        catalogue = Catalogue.read(hub)
    draws = random.Random(seed)  # noqa: S311
    classes = config.datasets.dominant_classes
    topic = result_topic(
        prefix=services.consumer.reply_topic_prefix,
        consumer_id=consumer_id,
        client_id=CLIENT_ID,
    )
    async with httpx.AsyncClient(
        base_url=str(services.consumer.control_plane_url),
        limits=httpx.Limits(max_connections=services.consumer.send_window),
        timeout=services.consumer.result_timeout_ms / 1000.0,
    ) as plane:

        async def post(index: int) -> int:
            request = ClassifyRequest(
                consumer_id=consumer_id,
                image_id=catalogue.draw(dominant_classes=classes, draws=draws),
                reply_topic=topic,
                correlation_id=uuid.uuid4().hex,
            )
            answer = await plane.post("/classify", json=request.model_dump(mode="json"))
            return answer.status_code

        return await run_load_test(
            steps,
            post=post,
            clock=time.monotonic,
            sleep=asyncio.sleep,
            wall=lambda: datetime.now(UTC),
            max_in_flight=services.consumer.send_window,
        )


def _reports(
    config: LoadTestView, password: str, run_id: str, consumer_id: str, sent: list[StepOutcome]
) -> list[Report]:
    database = PostgresDatabase.from_config(
        config.services.metrics_database, password=password, run_id=run_id
    )
    try:
        return [
            reported(
                one,
                database.read_served(consumer_id=consumer_id, since=one.since, until=one.until),
                database.read_member_answers(since=one.since, until=one.until),
            )
            for one in sent
        ]
    finally:
        database.close()


if __name__ == "__main__":
    sys.exit(main())
