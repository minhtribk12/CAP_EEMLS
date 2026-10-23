from __future__ import annotations

from typing import Literal

from pydantic import HttpUrl, model_validator

from eemls.config.schema import (
    FrozenMap,
    FrozenModel,
    NonNegativeCount,
    Port,
    PositiveCount,
    PositiveNumber,
    Ratio,
    UrlPath,
)

PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


class ControlPlaneConfig(FrozenModel):
    host: str
    port: Port
    policy: str
    contract: str | None = None
    max_ensemble_size: PositiveCount
    decision_interval_queries: PositiveCount
    periods_of_the_day: PositiveCount
    member_timeout_ms: PositiveCount
    probe_fraction: Ratio
    profile_update_interval_s: PositiveNumber
    max_concurrent_queries_per_consumer: PositiveCount
    data_hub_url: HttpUrl
    result_topic_prefix: str


class BrokerConfig(FrozenModel):
    host: str
    port: Port
    qos: Literal[1, 2]
    keepalive_s: PositiveCount
    session_expiry_s: PositiveCount
    max_queued_results: PositiveCount
    client_id_prefix: str
    instance_topic_prefix: str
    reconnect_min_s: PositiveNumber
    reconnect_max_s: PositiveNumber
    shutdown_flush_s: PositiveNumber

    @model_validator(mode="after")
    def _a_stop_waits_through_one_reconnect(self) -> BrokerConfig:
        if self.shutdown_flush_s < self.reconnect_max_s:
            raise ValueError(
                f"a stop waiting {self.shutdown_flush_s} s gives up before the library retries, "
                f"which it does after at most {self.reconnect_max_s} s"
            )
        return self

    @model_validator(mode="after")
    def _a_session_outlives_a_dropped_connection(self) -> BrokerConfig:
        if self.session_expiry_s <= self.keepalive_s:
            raise ValueError(
                f"a session expiring in {self.session_expiry_s} s does not outlive a connection "
                f"the broker gives up on after {self.keepalive_s} s"
            )
        return self

    @model_validator(mode="after")
    def _backoff_is_ordered(self) -> BrokerConfig:
        if self.reconnect_min_s > self.reconnect_max_s:
            raise ValueError(
                f"reconnect_min_s {self.reconnect_min_s} is above "
                f"reconnect_max_s {self.reconnect_max_s}"
            )
        return self


class MetricsDatabaseConfig(FrozenModel):
    backend: Literal["postgres"]
    host: str
    port: Port
    database: str
    user: str
    buffer_rows: PositiveCount
    flush_interval_ms: PositiveCount
    statement_timeout_ms: PositiveCount
    max_database_connections: PositiveCount

    def dsn(self, password: str) -> str:
        return f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.database}"


class MonitoringConfig(FrozenModel):
    enabled: bool
    metrics_path: UrlPath
    histogram_buckets_ms: tuple[PositiveNumber, ...]
    fast_histogram_buckets_ms: tuple[PositiveNumber, ...]

    @model_validator(mode="after")
    def _buckets_rise(self) -> MonitoringConfig:
        for name, edges in (
            ("histogram_buckets_ms", self.histogram_buckets_ms),
            ("fast_histogram_buckets_ms", self.fast_histogram_buckets_ms),
        ):
            if not edges:
                raise ValueError(f"{name} is empty")
            if list(edges) != sorted(edges):
                raise ValueError(f"{name} is not in rising order")
        return self


class InferenceConfig(FrozenModel):
    port: Port
    max_concurrent_requests: PositiveCount
    session_threads: PositiveCount
    warmup_requests: PositiveCount
    number_format: Literal["float32", "float16"]
    providers: FrozenMap[tuple[str, ...]]
    image_cache_images: NonNegativeCount
    image_fetch_timeout_ms: PositiveCount
    max_image_bytes: PositiveCount
    data_hub_url: HttpUrl

    @model_validator(mode="after")
    def _providers_are_known(self) -> InferenceConfig:
        for execution_provider, chain in self.providers.items():
            if not chain:
                raise ValueError(f"providers.{execution_provider} names no provider")
            unknown = [name for name in chain if name not in PROVIDERS]
            if unknown:
                raise ValueError(
                    f"providers.{execution_provider} names unknown providers {unknown}"
                )
        return self


class ConsumerServiceConfig(FrozenModel):
    control_plane_url: HttpUrl
    data_hub_url: HttpUrl
    decision_interval_queries: PositiveCount
    send_window: PositiveCount
    tick_ms: PositiveNumber
    result_timeout_ms: PositiveCount
    subscription_wait_s: PositiveNumber
    reply_topic_prefix: str
    feedback_batch: PositiveCount
    metrics_port: Port


class DataHubConfig(FrozenModel):
    host: str
    port: Port
    image_root: str
    max_image_bytes: PositiveCount


class ObservationServiceConfig(FrozenModel):
    url: HttpUrl
    host: str
    port: Port
    request_timeout_ms: PositiveCount
    profile_window_s: PositiveNumber
    profile_min_labeled_per_class: PositiveCount
    profile_min_labeled_overall: PositiveCount
