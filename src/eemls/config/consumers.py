from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field, model_validator

from eemls.config.reading import ConfigError, Reader, validate
from eemls.config.schema import FrozenMap, FrozenModel
from eemls.domain.consumer_defined_metrics import CdmDefinition, CdmSet
from eemls.domain.contract import ConsumerContract


class CdmFileConfig(FrozenModel):
    cdms: CdmSet

    @model_validator(mode="before")
    @classmethod
    def _the_file_is_the_set(cls, data: object) -> object:
        if isinstance(data, dict) and isinstance(data.get("cdms"), dict):
            return {**data, "cdms": {"definitions": data["cdms"]}}
        return data


class InitialEnsembleConfig(FrozenModel):
    members: tuple[str, ...]

    @model_validator(mode="after")
    def _an_ensemble_holds_at_least_one_service_once(self) -> InitialEnsembleConfig:
        if not self.members:
            raise ValueError("members is empty, and an ensemble keeps at least one service")
        repeated = sorted({one for one in self.members if self.members.count(one) > 1})
        if repeated:
            raise ValueError(f"members names {repeated} more than once")
        return self


class ConsumerConfig(FrozenModel):
    contract: ConsumerContract
    cdms: FrozenMap[CdmDefinition]
    contract_variants: FrozenMap[ConsumerContract] = Field(default_factory=dict)
    initial_ensemble: tuple[str, ...] = ()

    def scored_under(self, name: str | None) -> ConsumerContract:
        if name is None:
            return self.contract
        return self.contract_variants.get(name, self.contract)


def load_consumers(reader: Reader) -> dict[str, ConsumerConfig]:
    directory = reader.root / "consumers"
    if not directory.is_dir():
        raise ConfigError("consumers: directory is missing")
    consumers: dict[str, ConsumerConfig] = {}
    for folder in sorted(path for path in directory.iterdir() if path.is_dir()):
        contract_name = f"consumers/{folder.name}/contract.yaml"
        cdm_name = f"consumers/{folder.name}/cdm.yaml"
        contract_body = reader.read_file(contract_name)
        contract = validate(ConsumerContract, contract_body, contract_name)
        cdm_file = reader.build(CdmFileConfig, cdm_name)
        variants = load_contract_variants(reader, folder.name)
        consumer = ConsumerConfig(
            contract=contract,
            cdms=cdm_file.cdms.definitions,
            contract_variants=variants,
            initial_ensemble=load_initial_ensemble(reader, folder.name),
        )
        for name, held in stated_by(consumer):
            if held.consumer_id != folder.name:
                raise ConfigError(
                    f"consumers/{folder.name}/{name}: consumer_id {held.consumer_id} "
                    f"does not match the folder {folder.name}"
                )
        check_objectives_are_declared(consumer, folder.name)
        consumers[folder.name] = consumer
    if not consumers:
        raise ConfigError("consumers: no consumer folder found")
    return consumers


def load_contract_variants(reader: Reader, folder: str) -> dict[str, ConsumerContract]:
    directory = reader.root / "consumers" / folder
    found: dict[str, ConsumerContract] = {}
    for path in sorted(directory.glob("contract_*.yaml")):
        name = f"consumers/{folder}/{path.name}"
        found[path.stem.removeprefix("contract_")] = validate(
            ConsumerContract, reader.read_file(name), name
        )
    return found


def stated_by(consumer: ConsumerConfig) -> tuple[tuple[str, ConsumerContract], ...]:
    return (
        ("contract.yaml", consumer.contract),
        *((f"contract_{key}.yaml", one) for key, one in consumer.contract_variants.items()),
    )


def load_initial_ensemble(reader: Reader, folder: str) -> tuple[str, ...]:
    name = f"consumers/{folder}/initial_ensemble.yaml"
    if not (reader.root / "consumers" / folder / "initial_ensemble.yaml").is_file():
        return ()
    return reader.build(InitialEnsembleConfig, name).members


def check_objectives_are_declared(consumer: ConsumerConfig, consumer_id: str) -> None:
    for name, held in stated_by(consumer):
        which = f"consumers/{consumer_id}/{name}"
        undeclared = [cdm for cdm in held.objectives if cdm not in consumer.cdms]
        if undeclared:
            raise ConfigError(f"{which}: objectives name undeclared CDMs {sorted(undeclared)}")
        required = [level.cdm for level in held.requirements if level.cdm not in consumer.cdms]
        if required:
            raise ConfigError(f"{which}: requirements name undeclared CDMs {sorted(required)}")


def check_the_cdms(consumers: Mapping[str, ConsumerConfig], classes: set[str]) -> None:
    _check_a_scored_cdm_answers_one_number(consumers)
    _check_one_cdm_id_means_one_metric(consumers)
    _check_the_weighted_classes_are_held(consumers, classes)


def _check_a_scored_cdm_answers_one_number(consumers: Mapping[str, ConsumerConfig]) -> None:
    for consumer_id, consumer in consumers.items():
        for which, held in stated_by(consumer):
            for name in held.objectives:
                specification = consumer.cdms[name].specification
                if specification.answers_one_value_per_group():
                    raise ConfigError(
                        f"consumers/{consumer_id}/{which}: {name} groups by "
                        f"{specification.group_by}, so it answers one value per group"
                    )


def _check_one_cdm_id_means_one_metric(consumers: Mapping[str, ConsumerConfig]) -> None:
    seen: dict[str, tuple[str, CdmDefinition]] = {}
    for consumer_id, consumer in consumers.items():
        for name, definition in consumer.cdms.items():
            first = seen.setdefault(name, (consumer_id, definition))
            if first[1] != definition:
                raise ConfigError(
                    f"consumers/{consumer_id}/cdm.yaml: {name} is declared differently by "
                    f"{first[0]}, so one id would mean two metrics"
                )


def _check_the_weighted_classes_are_held(
    consumers: Mapping[str, ConsumerConfig], known: set[str]
) -> None:
    for consumer_id, consumer in consumers.items():
        for metric, definition in consumer.cdms.items():
            unknown = sorted(_weighted_classes(definition) - known)
            if unknown:
                raise ConfigError(
                    f"consumers/{consumer_id}/cdm.yaml: {metric} weighs {unknown}, "
                    "which the dataset does not hold"
                )


def _weighted_classes(definition: CdmDefinition) -> set[str]:
    found: set[str] = set()
    pending = [definition.specification]
    while pending:
        node = pending.pop()
        if node.weights:
            found |= set(node.weights)
        pending.extend(node.inputs)
    return found
