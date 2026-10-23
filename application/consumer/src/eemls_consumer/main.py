from __future__ import annotations

import argparse
import logging
import random
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

from eemls.config.reading import ConfigError, Reader
from eemls.config.services import ConsumerServiceConfig
from eemls.domain.contract import ConsumerContract
from eemls.domain.random_seeds import consumer_seed
from eemls.image_catalogue import Catalogue, CatalogueError
from eemls.messaging.result_listener import ResultListener
from eemls.messaging.topics import TopicError, result_topic
from eemls.monitoring.metrics import Monitoring
from eemls.monitoring.metrics_endpoint import metrics_router
from eemls_consumer.config import ConsumerView, load_consumer_config
from eemls_consumer.feedback import LabelsToReturn
from eemls_consumer.query_schedule import Schedule
from eemls_consumer.query_sender import QuerySender
from eemls_consumer.results import Pending

logger = logging.getLogger(__name__)


def label_delay_queries(contract: ConsumerContract, consumer: ConsumerServiceConfig) -> int:
    return contract.feedback.delay_intervals * consumer.decision_interval_queries


def result_timeout_s(config: ConsumerView) -> float:
    return config.services.consumer.result_timeout_ms / 1000.0


def schedule_of(path: Path, *, request_rate_per_s: float) -> Schedule:
    held = Reader(path.parent).read_file(path.name)
    return Schedule.model_validate({**held, "request_rate_per_s": request_rate_per_s})


def random_generators_of(seed: int, consumer_id: str) -> tuple[random.Random, random.Random]:
    own = consumer_seed(seed, consumer_id)
    return random.Random(own), random.Random(f"labels-{own}")  # noqa: S311


def build_consumer(
    config: ConsumerView, *, consumer_id: str, client_id: str, scenario: Path, seed: int
) -> tuple[QuerySender, ResultListener, Monitoring]:
    settings = config.services
    contract = config.contracts[consumer_id]
    schedule = schedule_of(scenario, request_rate_per_s=contract.request_rate_per_s)
    monitoring = Monitoring.from_config(settings.monitoring)
    topic = result_topic(
        prefix=settings.consumer.reply_topic_prefix,
        consumer_id=consumer_id,
        client_id=client_id,
    )
    with httpx.Client(base_url=str(settings.consumer.data_hub_url)) as hub:
        catalogue = Catalogue.read(hub)
    pending = Pending(timeout_ms=float(settings.consumer.result_timeout_ms))
    draws, labels = random_generators_of(seed, consumer_id)
    client = QuerySender(
        consumer_id=consumer_id,
        reply_topic=topic,
        schedule=schedule,
        catalogue=catalogue,
        pending=pending,
        labels_to_return=LabelsToReturn(
            fraction=contract.feedback.fraction,
            label_delay_queries=label_delay_queries(contract, settings.consumer),
            draw=labels.random,
        ),
        control_plane=httpx.Client(base_url=str(settings.consumer.control_plane_url)),
        send_window=settings.consumer.send_window,
        feedback_batch=settings.consumer.feedback_batch,
        decision_interval_queries=settings.consumer.decision_interval_queries,
        draw=draws.random,
        identifiers=lambda: uuid.uuid4().hex,
        monitoring=monitoring,
    )
    listener = ResultListener(
        broker=settings.broker,
        client_id=f"{settings.broker.client_id_prefix}-{consumer_id}-{client_id}",
        topic=topic,
        on_result=client.result,
    )
    return client, listener, monitoring


def metrics_app(monitoring: Monitoring) -> FastAPI:
    app = FastAPI(title="eemls consumer", docs_url=None)
    app.include_router(metrics_router(monitoring))
    return app


def start_metrics_server(monitoring: Monitoring, port: int) -> threading.Thread:
    server = uvicorn.Server(
        uvicorn.Config(metrics_app(monitoring), host="0.0.0.0", port=port, log_level="warning")  # noqa: S104
    )
    thread = threading.Thread(target=server.run, name="eemls-consumer-metrics", daemon=True)
    thread.start()
    return thread


def run(
    client: QuerySender,
    *,
    duration_s: float,
    result_timeout_s: float,
    interval_s: float,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    client.begin()
    while client.elapsed() < duration_s:
        client.tick()
        client.expire()
        sleeper(interval_s)
    client.returned_the_rest()
    while client.elapsed() < duration_s + result_timeout_s:
        client.expire()
        sleeper(interval_s)
    return client.accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eemls-consumer")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--consumer", required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--client-id", default=f"client-{socket.gethostname()}")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        config = load_consumer_config(args.config)
        if args.consumer not in config.contracts:
            raise ConfigError(f"{args.consumer} is not a consumer of this configuration")
        client, listener, monitoring = build_consumer(
            config,
            consumer_id=args.consumer,
            client_id=args.client_id,
            scenario=args.scenario,
            seed=args.seed,
        )
    except (
        ConfigError,
        ValidationError,
        CatalogueError,
        TopicError,
        ValueError,
        OSError,
    ) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1

    start_metrics_server(monitoring, config.services.consumer.metrics_port)
    listener.start()
    waited_s = config.services.consumer.subscription_wait_s
    if not listener.wait_until_ready(waited_s):
        listener.close()
        client.close()
        print(f"refused: the result subscription was not granted in {waited_s} s", file=sys.stderr)
        return 1
    logger.info(
        f"stage_tag=started process_tag=consumer_client consumer={args.consumer} "
        f"client={args.client_id} duration_s={args.duration_s}"
    )
    try:
        answered = run(
            client,
            duration_s=args.duration_s,
            result_timeout_s=result_timeout_s(config),
            interval_s=config.services.consumer.tick_ms / 1000.0,
        )
    finally:
        listener.close()
        client.close()
    print(f"{answered} queries accepted of {args.consumer} in {args.duration_s} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
