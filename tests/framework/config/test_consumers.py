from __future__ import annotations

import pytest

from eemls.config.consumers import ConsumerConfig, check_the_cdms, load_consumers
from eemls.config.datasets import DatasetClasses
from eemls.config.reading import ConfigError, Reader
from eemls.domain.consumer_defined_metrics import MetricSpecification
from eemls.testing.trees import TREE

READER = Reader(TREE)
CONSUMERS = load_consumers(READER)
CLASSES = set(READER.build(DatasetClasses, "datasets/sources.yaml").classes)
FIRST = sorted(CONSUMERS)[0]


def with_cdm(consumer: ConsumerConfig, name: str, specification: object) -> ConsumerConfig:
    changed = consumer.cdms[name].model_copy(
        update={"specification": MetricSpecification.model_validate(specification)}
    )
    return consumer.model_copy(update={"cdms": {**consumer.cdms, name: changed}})


def scored(consumer: ConsumerConfig) -> str:
    return sorted(consumer.contract.objectives)[0]


class TestTheCdms:
    def test_the_committed_consumers_pass(self) -> None:
        check_the_cdms(CONSUMERS, CLASSES)

    def test_a_scored_cdm_answering_one_value_per_group_is_refused(self) -> None:
        consumer = CONSUMERS[FIRST]
        grouped = with_cdm(
            consumer, scored(consumer), {"function": "count", "group_by": "true_label"}
        )

        with pytest.raises(ConfigError, match="answers one value per group"):
            check_the_cdms({FIRST: grouped}, CLASSES)

    def test_one_cdm_id_declared_two_ways_is_refused(self) -> None:
        consumer = CONSUMERS[FIRST]
        other = with_cdm(consumer, scored(consumer), {"function": "count"})

        with pytest.raises(ConfigError, match="declared differently"):
            check_the_cdms({FIRST: consumer, "other": other}, CLASSES)

    def test_a_weighted_class_the_dataset_does_not_hold_is_refused(self) -> None:
        consumer = CONSUMERS[FIRST]
        weighing = with_cdm(
            consumer,
            scored(consumer),
            {"function": "weighted_count", "group_by": "true_label", "weights": {"dragon": 1.0}},
        )

        with pytest.raises(ConfigError, match="weighs \\['dragon'\\]"):
            check_the_cdms({FIRST: weighing}, CLASSES)
