from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import onnxruntime
import uvicorn
from fastapi import FastAPI

from eemls.config.reading import ConfigError
from eemls.domain.records import Instance
from eemls.messaging.broker import Factory, client_of
from eemls.messaging.instance_announcements import Announcer
from eemls.messaging.topics import instance_topic
from eemls.monitoring.metrics import Monitoring
from eemls_inference_service.backends.backend import BackendError
from eemls_inference_service.config import InferenceView, load_inference_config
from eemls_inference_service.data_hub_client import DataHubClient
from eemls_inference_service.http_app import InferenceEndpoints, build_app
from eemls_inference_service.model_loading import load

logger = logging.getLogger(__name__)
MODELS = Path("models")


def monitoring_of(config: InferenceView) -> Monitoring:
    return Monitoring.from_config(config.services.monitoring)


def providers_for(config: InferenceView, service_name: str) -> tuple[str, ...]:
    device_type = config.models.inference_services.services[service_name].device_type
    execution_provider = config.cost.devices[device_type].execution_provider
    return config.services.inference_service.providers[execution_provider]


def inference_endpoints_of(
    config: InferenceView,
    *,
    service_name: str,
    instance_id: str,
    node: str,
    replica: int,
    models: Path,
    monitoring: Monitoring,
) -> InferenceEndpoints:
    services = config.models.inference_services.services
    if service_name not in services:
        raise ConfigError(
            f"models/inference_services.yaml declares no {service_name}, only {sorted(services)}"
        )
    service = services[service_name]
    repository = config.models.model_repository.models
    settings = config.services.inference_service
    chain = providers_for(config, service_name)
    return InferenceEndpoints(
        instance_id=instance_id,
        service_name=service_name,
        model=service.model,
        device=service.device_type,
        node=node,
        replica=replica,
        number_format=settings.number_format,
        backend=load(
            service.model,
            repository[service.model],
            under=models,
            providers=chain,
            session_threads=settings.session_threads,
            warmup_requests=settings.warmup_requests,
        ),
        images=DataHubClient(
            base_url=str(settings.data_hub_url),
            timeout_ms=settings.image_fetch_timeout_ms,
            cache_images=settings.image_cache_images,
            max_image_bytes=settings.max_image_bytes,
        ),
        max_concurrent_requests=settings.max_concurrent_requests,
        monitoring=monitoring,
    )


def instance_of(
    config: InferenceView,
    *,
    service_name: str,
    instance_id: str,
    node: str,
    replica: int,
    provider: str,
    image: str,
    started_at: datetime,
) -> Instance:
    service = config.models.inference_services.services[service_name]
    return Instance(
        instance_id=instance_id,
        service_name=service_name,
        model=service.model,
        device=service.device_type,
        node=node,
        replica=replica,
        provider=provider,
        number_format=config.services.inference_service.number_format,
        runtime_version=onnxruntime.__version__,
        image_tag=image,
        config_hash=hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest(),
        started_at=started_at,
    )


def announcer_of(
    config: InferenceView, instance: Instance, *, run_id: str, factory: Factory = client_of
) -> Announcer:
    settings = config.services.broker
    return Announcer(
        broker=settings,
        client_id=f"{settings.client_id_prefix}-{instance.instance_id}",
        topic=instance_topic(
            prefix=settings.instance_topic_prefix,
            run_id=run_id,
            instance_id=instance.instance_id,
        ),
        instance=instance,
        factory=factory,
    )


def build_inference_service(
    argv: list[str] | None = None,
) -> tuple[FastAPI, InferenceView, Announcer]:
    parser = argparse.ArgumentParser(prog="eemls-inference-service")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--models", type=Path, default=Path(os.environ.get("EEMLS_MODELS_DIR", MODELS))
    )
    parser.add_argument("--service", default=os.environ.get("EEMLS_SERVICE_NAME", ""))
    parser.add_argument("--instance", default=os.environ.get("EEMLS_INSTANCE_ID", ""))
    parser.add_argument("--node", default=os.environ.get("EEMLS_NODE", ""))
    parser.add_argument("--replica", type=int, default=int(os.environ.get("EEMLS_REPLICA", "1")))
    parser.add_argument("--run", default=os.environ.get("EEMLS_RUN_ID", "unnamed"))
    parser.add_argument("--image", default=os.environ.get("EEMLS_IMAGE", "unnamed"))
    args = parser.parse_args(argv)

    config = load_inference_config(args.config)
    monitoring = monitoring_of(config)
    serving = inference_endpoints_of(
        config,
        service_name=args.service,
        instance_id=args.instance,
        node=args.node,
        replica=args.replica,
        models=args.models,
        monitoring=monitoring,
    )
    logger.info(
        f"stage_tag=started process_tag=inference service={serving.service_name} "
        f"instance={serving.instance_id} provider={serving.backend.provider()}"
    )
    instance = instance_of(
        config,
        service_name=serving.service_name,
        instance_id=serving.instance_id,
        node=serving.node,
        replica=serving.replica,
        provider=serving.backend.provider(),
        image=args.image,
        started_at=datetime.now(UTC),
    )
    return build_app(serving, monitoring), config, announcer_of(config, instance, run_id=args.run)


def main(argv: list[str] | None = None) -> int:
    try:
        app, config, announcer = build_inference_service(argv)
    except (ConfigError, BackendError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    settings = config.services.inference_service
    announcer.start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")  # noqa: S104
    finally:
        announcer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
