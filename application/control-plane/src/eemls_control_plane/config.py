from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from eemls.config.consumers import ConsumerConfig, check_the_cdms, load_consumers, stated_by
from eemls.config.cost import CostConfig
from eemls.config.datasets import DatasetClasses
from eemls.config.models import InferenceServicesConfig
from eemls.config.reading import (
    ConfigError,
    Reader,
    check_the_device_types_are_priced,
    read_services,
    root_of,
)
from eemls.config.schema import (
    ExtraFieldsIgnored,
    FrozenMap,
    FrozenModel,
    Port,
)
from eemls.config.services import (
    BrokerConfig,
    ControlPlaneConfig,
    MetricsDatabaseConfig,
    MonitoringConfig,
    ObservationServiceConfig,
)
from eemls.ensemble_optimization.approach import ApproachSettings
from eemls.ensemble_optimization.approach_loading import load_approaches

FILES_READ: frozenset[str] = frozenset(
    {
        "services/control_plane.yaml",
        "services/metrics_database.yaml",
        "services/monitoring.yaml",
        "services/observation_service.yaml",
        "services/broker.yaml",
        "services/inference_service.yaml",
        "consumers",
        "approaches",
        "models/inference_services.yaml",
        "cost/prices.yaml",
        "datasets/sources.yaml",
    }
)


class InferencePort(ExtraFieldsIgnored):
    port: Port


class ControlPlaneViewServices(FrozenModel):
    control_plane: ControlPlaneConfig
    metrics_database: MetricsDatabaseConfig
    monitoring: MonitoringConfig
    observation_service: ObservationServiceConfig
    broker: BrokerConfig
    inference_service: InferencePort


class ControlPlaneViewModels(FrozenModel):
    inference_services: InferenceServicesConfig


class ControlPlaneView(FrozenModel):
    services: ControlPlaneViewServices
    consumers: FrozenMap[ConsumerConfig]
    approaches: FrozenMap[ApproachSettings]
    models: ControlPlaneViewModels
    datasets: DatasetClasses
    cost: CostConfig


def load_control_plane_config(directory: Path | str | None = None) -> ControlPlaneView:
    reader = Reader(root_of(directory))
    config = ControlPlaneView(
        services=read_services(reader, ControlPlaneViewServices),
        consumers=load_consumers(reader),
        approaches=load_approaches(reader),
        models=ControlPlaneViewModels(
            inference_services=reader.build(
                InferenceServicesConfig, "models/inference_services.yaml"
            )
        ),
        datasets=reader.build(DatasetClasses, "datasets/sources.yaml"),
        cost=reader.build(CostConfig, "cost/prices.yaml"),
    )
    check_the_control_plane_files(
        config.services.control_plane,
        config.approaches,
        config.consumers,
        config.models.inference_services,
        config.datasets.dominant_classes,
        config.datasets.classes,
        config.cost.devices,
    )
    return config


def check_the_control_plane_files(
    control_plane: ControlPlaneConfig,
    approaches: Mapping[str, ApproachSettings],
    consumers: Mapping[str, ConsumerConfig],
    inference_services: InferenceServicesConfig,
    dominant_classes: tuple[str, ...],
    classes: tuple[str, ...],
    devices: Mapping[str, object],
) -> None:
    chosen = control_plane.policy
    if chosen not in approaches:
        raise ConfigError(
            f"services/control_plane.yaml: policy {chosen} has no file in approaches/"
        )
    limit = control_plane.max_ensemble_size
    for name, approach in approaches.items():
        if approach.max_ensemble_size > limit:
            raise ConfigError(
                f"approaches/{name}.yaml: max_ensemble_size {approach.max_ensemble_size} is above "
                f"the control plane limit {limit}"
            )
    periods = control_plane.periods_of_the_day
    dominant = len(dominant_classes)
    if periods != dominant:
        raise ConfigError(
            f"services/control_plane.yaml: periods_of_the_day {periods} is not the "
            f"{dominant} dominant classes of the dataset, and a period is named by one"
        )
    _check_the_chosen_contract_is_stated(control_plane, consumers)
    _check_the_starting_ensembles_can_be_served(
        control_plane, approaches, consumers, inference_services
    )
    _check_a_label_lands_inside_the_window_that_scores_it(control_plane, approaches, consumers)
    _check_the_prioritized_classes_are_held(consumers, set(classes))
    check_the_cdms(consumers, set(classes))
    check_the_device_types_are_priced(inference_services, devices, "cost/prices.yaml")


def _check_the_prioritized_classes_are_held(
    consumers: Mapping[str, ConsumerConfig], known: set[str]
) -> None:
    for consumer_id, consumer in consumers.items():
        for which, held in stated_by(consumer):
            unknown = [name for name in held.prioritized_classes if name not in known]
            if unknown:
                raise ConfigError(
                    f"consumers/{consumer_id}/{which}: prioritized_classes names {unknown}, "
                    "which the dataset does not hold"
                )


def _check_the_chosen_contract_is_stated(
    control_plane: ControlPlaneConfig, consumers: Mapping[str, ConsumerConfig]
) -> None:
    name = control_plane.contract
    if name is None:
        return
    if not any(name in consumer.contract_variants for consumer in consumers.values()):
        raise ConfigError(f"services/control_plane.yaml: contract {name} is stated by no consumer")


def _check_the_starting_ensembles_can_be_served(
    control_plane: ControlPlaneConfig,
    approaches: Mapping[str, ApproachSettings],
    consumers: Mapping[str, ConsumerConfig],
    inference_services: InferenceServicesConfig,
) -> None:
    pool = set(inference_services.services)
    chosen = control_plane.policy
    limit = approaches[chosen].max_ensemble_size
    for consumer_id, consumer in consumers.items():
        where = f"consumers/{consumer_id}/initial_ensemble.yaml"
        unknown = [name for name in consumer.initial_ensemble if name not in pool]
        if unknown:
            raise ConfigError(f"{where}: names {unknown}, which the pool does not hold")
        if len(consumer.initial_ensemble) > limit:
            raise ConfigError(
                f"{where}: names {len(consumer.initial_ensemble)} services, above the "
                f"max_ensemble_size of {limit} that approaches/{chosen}.yaml states"
            )


def _check_a_label_lands_inside_the_window_that_scores_it(
    control_plane: ControlPlaneConfig,
    approaches: Mapping[str, ApproachSettings],
    consumers: Mapping[str, ConsumerConfig],
) -> None:
    interval = control_plane.decision_interval_queries
    approach_windows = [approach.evaluation_window for approach in approaches.values()]
    for consumer_id, consumer in consumers.items():
        windows = [one.window for one in consumer.cdms.values() if one.window] + [
            window for window in approach_windows if window
        ]
        for contract in (consumer.contract, *consumer.contract_variants.values()):
            delay = contract.feedback.delay_intervals * interval
            narrowest = min(windows, default=None)
            if narrowest is not None and delay >= narrowest:
                raise ConfigError(
                    f"consumers/{consumer_id}: a label returns {delay} queries late "
                    f"({contract.feedback.delay_intervals} intervals of {interval}), and the "
                    f"narrowest window that scores it holds {narrowest}"
                )
