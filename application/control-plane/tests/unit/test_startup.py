from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from eemls.config.reading import ConfigError
from eemls.domain.types import AggregatedResult, EnsembleMember, Query, Row
from eemls.ensemble_optimization import approach_loading
from eemls.ensemble_optimization.approach import ApproachSettings, Policy
from eemls.ensemble_optimization.approach_loading import (
    StartupError,
    policy_of,
)
from eemls.ensemble_optimization.testing.approaches import installed
from eemls.interfaces.control_plane_api import ClassifyRequest
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.profile_updates import ProfileUpdatesClient, ProfileUpdating
from eemls.observation_service.testing import (
    FixedContext,
    InProcessObservationService,
    conditioned,
    in_process_profile_updates,
)
from eemls.orchestration.ensemble_deployments import DeploymentError
from eemls.orchestration.testing import Answers, Members
from eemls.testing.records import AT, prediction, profile, served
from eemls.testing.trees import TREE, committed_tree
from eemls_control_plane.config import ControlPlaneView, load_control_plane_config
from eemls_control_plane.http_app import ControlPlaneEndpoints
from eemls_control_plane.startup import (
    build_control_plane,
    check_every_service_is_profiled,
    consumers_of,
    members_of,
    remembered_queries,
)

POOL = {
    "infer-b3-pi5": ("b3", "raspberry_pi5"),
    "infer-b6-orin": ("b6", "jetson_orin_nano"),
    "infer-b5-pi5": ("b5", "raspberry_pi5"),
}
STARTS_ON = ("infer-b3-pi5", "infer-b6-orin")


def nothing_published(query: Query, result: AggregatedResult) -> None: ...


def asked() -> ClassifyRequest:
    return ClassifyRequest(
        consumer_id="c1_navigation",
        image_id="img-7",
        reply_topic="eemls/results/c1_navigation/client-1",
        correlation_id="corr-1",
    )


@pytest.fixture
def tree(tmp_path: Path) -> Iterator[Path]:
    target = committed_tree(tmp_path / "config")
    repository = target / "models" / "model_repository.yaml"
    body = yaml.safe_load(repository.read_text(encoding="utf-8"))
    body["models"] = {
        model: {
            "hugging_face_id": f"onnx-community/{model}",
            "base_model": model,
            "onnx_file": "model.onnx",
            "sha256": "b" * 64,
            "input_height": 224,
            "input_width": 224,
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
            "classes": ["car"],
            "memory_mb": 20.0,
            "variant_of": None,
        }
        for model, _ in POOL.values()
    }
    repository.write_text(yaml.safe_dump(body), encoding="utf-8")
    pool = target / "models" / "inference_services.yaml"
    held = yaml.safe_load(pool.read_text(encoding="utf-8"))
    pool.write_text(
        yaml.safe_dump(
            {
                "autoscale": held["autoscale"],
                "services": {
                    name: {"model": model, "device_type": device, "replicas": 1}
                    for name, (model, device) in POOL.items()
                },
            }
        ),
        encoding="utf-8",
    )
    for folder in sorted((target / "consumers").iterdir()):
        (folder / "initial_ensemble.yaml").write_text(
            yaml.safe_dump({"members": list(STARTS_ON)}), encoding="utf-8"
        )
    plane = target / "services" / "control_plane.yaml"
    held = yaml.safe_load(plane.read_text(encoding="utf-8"))
    held["policy"] = "keep_the_ensemble"
    plane.write_text(yaml.safe_dump(held), encoding="utf-8")
    yield target


class Records(MemoryMetricsDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.limits: list[int] = []

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        self.limits.append(limit)
        return super().read_window(consumer_id, limit)

    def read_member_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        self.limits.append(limit)
        return super().read_member_window(consumer_id, limit)


def profiles() -> tuple[object, ...]:
    return tuple(
        profile(model=model, device=device, service_name=name, instance_id=f"i-{index}")
        for index, (name, (model, device)) in enumerate(sorted(POOL.items()))
    )


def stored() -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase()
    for one in profiles():
        database.write_profile(one)  # type: ignore[arg-type]
    return database


def build(
    config: ControlPlaneView,
    database: MemoryMetricsDatabase,
    *,
    instance_id: str = "control-plane-1",
    now: Callable[[], datetime] = lambda: AT,
    transport: httpx.AsyncBaseTransport | None = None,
    profiles: ProfileUpdating | None = None,
    seed: int = 0,
) -> ControlPlaneEndpoints:
    return build_control_plane(
        config,
        database=database,
        instance_id=instance_id,
        monitoring=Monitoring.from_config(config.services.monitoring),
        publishing=nothing_published,
        runtime_context=FixedContext(),
        condition_of=conditioned(),
        now=now,
        transport=transport,
        seed=seed,
        profiles=profiles
        if profiles is not None
        else in_process_profile_updates(
            database,
            rates=config.cost.rates(),
            window_s=config.services.observation_service.profile_window_s,
            interval_s=config.services.control_plane.profile_update_interval_s,
            min_labeled_per_class=config.services.observation_service.profile_min_labeled_per_class,
            min_labeled_overall=config.services.observation_service.profile_min_labeled_overall,
        ),
    )


def with_plane_settings(tree: Path, **changes: object) -> None:
    plane = tree / "services" / "control_plane.yaml"
    body = yaml.safe_load(plane.read_text(encoding="utf-8"))
    plane.write_text(yaml.safe_dump(body | changes), encoding="utf-8")


class TestWhatItBuilds:
    def test_it_builds_the_control_plane_of_its_instance(self, tree: Path) -> None:
        config = load_control_plane_config(tree)

        serving = build(config, stored())

        assert serving.instance_id == "control-plane-1"

    def test_every_consumer_starts_on_the_ensemble_it_declared(self, tree: Path) -> None:
        config = load_control_plane_config(tree)

        serving = build(config, stored())

        for consumer_id in config.consumers:
            held = serving.ensemble(consumer_id)
            assert tuple(one.service_name for one in held.members) == STARTS_ON, consumer_id

    def test_the_pool_is_every_service_the_catalogue_holds(self, tree: Path) -> None:
        held = members_of(load_control_plane_config(tree))

        assert tuple(one.service_name for one in held) == tuple(sorted(POOL))

    def test_it_serves_the_consumers_under_the_contract_the_run_names(self, tree: Path) -> None:
        plane = tree / "services" / "control_plane.yaml"
        body = yaml.safe_load(plane.read_text(encoding="utf-8"))
        body["contract"] = "only_general_accuracy"
        plane.write_text(yaml.safe_dump(body), encoding="utf-8")

        held = consumers_of(load_control_plane_config(tree))

        for consumer_id, consumer in held.items():
            assert set(consumer.contract.objectives) == {"general_accuracy"}, consumer_id

    def test_without_a_named_contract_each_consumer_is_scored_on_its_own(self, tree: Path) -> None:
        held = consumers_of(load_control_plane_config(tree))

        assert "miss_rate_cyclist_as_motorcycle" in held["c2_safety"].contract.objectives


class TestWhoOwnsTheRun:
    def test_the_first_instance_owns_it(self, tree: Path) -> None:
        database = stored()

        build(load_control_plane_config(tree), database, instance_id="control-plane-1")

        assert database.owner is not None
        assert database.owner.instance_id == "control-plane-1"

    def test_a_second_instance_is_refused_and_told_who_owns_it(self, tree: Path) -> None:
        config, database = load_control_plane_config(tree), stored()
        build(config, database, instance_id="control-plane-1")

        with pytest.raises(StartupError, match="control-plane-1 already owns this run"):
            build(config, database, instance_id="control-plane-2")

    def test_the_owner_starting_again_is_not_refused(self, tree: Path) -> None:
        config, database = load_control_plane_config(tree), stored()
        build(config, database, instance_id="control-plane-1")

        assert build(config, database, instance_id="control-plane-1").instance_id == (
            "control-plane-1"
        )


class TestWhatItRefusesToStart:
    def test_a_service_with_no_measured_profile_is_refused_by_name(self, tree: Path) -> None:
        database = MemoryMetricsDatabase()
        database.write_profile(profiles()[0])  # type: ignore[arg-type]

        with pytest.raises(StartupError, match="infer-b5-pi5"):
            check_every_service_is_profiled(load_control_plane_config(tree), database)

    def test_a_consumer_with_no_starting_ensemble_is_refused_by_file(self, tree: Path) -> None:
        (tree / "consumers" / "c2_safety" / "initial_ensemble.yaml").unlink()

        with pytest.raises(StartupError, match="consumers/c2_safety/initial_ensemble.yaml"):
            build(load_control_plane_config(tree), stored())

    def test_a_policy_with_no_file_at_all_is_refused_by_name(self, tree: Path) -> None:
        with_plane_settings(tree, policy="nobody")

        with pytest.raises(ConfigError, match="nobody"):
            load_control_plane_config(tree)


class TestTheApproachItServes:
    def test_an_installed_approach_is_built_and_run_by_the_orchestrator(
        self, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        site = tmp_path / "site"
        site.mkdir()
        installed(site, monkeypatch)
        (tree / "approaches" / "fake.yaml").write_text(
            yaml.safe_dump({"kind": "fake", "max_ensemble_size": 3, "patience": 2}),
            encoding="utf-8",
        )
        with_plane_settings(tree, policy="fake")
        database = stored()
        serving = build(load_control_plane_config(tree), database)

        serving.decide("c1_navigation")

        assert [one.policy for one in database.decisions.values()] == ["fake"]

    def test_a_fixed_ensemble_is_still_served(self) -> None:
        config = load_control_plane_config(TREE)

        held = policy_of(config.approaches["keep_the_ensemble"], classes=config.datasets.classes)

        assert held.name == "keep_the_ensemble"

    def test_the_window_it_reads_is_the_one_its_own_file_declares(self, tree: Path) -> None:
        plane = tree / "services" / "control_plane.yaml"
        body = yaml.safe_load(plane.read_text(encoding="utf-8"))
        body["policy"] = "cap"
        plane.write_text(yaml.safe_dump(body), encoding="utf-8")
        cap = tree / "approaches" / "cap.yaml"
        asked = yaml.safe_load(cap.read_text(encoding="utf-8"))
        asked["evaluation_window"] = 700
        cap.write_text(yaml.safe_dump(asked), encoding="utf-8")
        database = Records()
        for one in profiles():
            database.write_profile(one)  # type: ignore[arg-type]
        config = load_control_plane_config(tree)

        build(config, database)

        assert database.limits == [700, 700] * len(config.consumers)

    def test_a_decision_reads_its_windows_without_asking_the_store(self, tree: Path) -> None:
        database = Records()
        for one in profiles():
            database.write_profile(one)  # type: ignore[arg-type]
        serving = build(load_control_plane_config(tree), database)
        database.limits.clear()

        serving.decide("c1_navigation")

        assert database.limits == []

    def test_the_ensemble_size_the_approach_states_is_the_one_enforced(self, tree: Path) -> None:
        config = load_control_plane_config(tree)
        narrow = config.approaches["keep_the_ensemble"].model_copy(update={"max_ensemble_size": 1})
        held = config.model_copy(
            update={"approaches": {**config.approaches, "keep_the_ensemble": narrow}}
        )

        with pytest.raises(DeploymentError, match="at most 1"):
            build(held, stored())

        assert config.services.control_plane.max_ensemble_size == 3


class TestTheSeedTheRunGives:
    def test_the_approach_is_built_on_the_seed_of_the_run(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        given: list[int] = []
        building = approach_loading.policy_of

        def policy_of(
            settings: ApproachSettings, *, classes: Sequence[str], seed: int = 0
        ) -> Policy:
            given.append(seed)
            return building(settings, classes=classes, seed=seed)

        monkeypatch.setattr("eemls_control_plane.startup.policy_of", policy_of)
        config = load_control_plane_config(tree)

        build(config, stored(), seed=4)

        assert given == [4]


class TestWhatItReadsFromTheTree:
    def test_the_device_rates_come_from_the_prices_file(self, tree: Path) -> None:
        config = load_control_plane_config(tree)

        held = config.cost.rates()

        assert set(held) == set(config.cost.devices)
        for name, rate in held.items():
            assert rate.power_watts == config.cost.devices[name].peak_power_watts
            assert rate.energy_price_per_kwh == config.cost.energy_price_per_kwh


class TestTheServicesOutsideTheEnsemble:
    async def test_they_are_asked_at_the_fraction_the_control_plane_file_sets(
        self, tree: Path
    ) -> None:
        with_plane_settings(tree, probe_fraction=1.0)
        members = Members({name: Answers(label="car") for name in POOL})
        serving = build(load_control_plane_config(tree), stored(), transport=members)

        serving.accept(asked())
        await serving.drain()

        assert "infer-b5-pi5" in members.asked

    async def test_none_are_asked_at_a_fraction_of_none(self, tree: Path) -> None:
        with_plane_settings(tree, probe_fraction=0.0)
        members = Members({name: Answers(label="car") for name in POOL})
        serving = build(load_control_plane_config(tree), stored(), transport=members)

        serving.accept(asked())
        await serving.drain()

        assert "infer-b5-pi5" not in members.asked


class TestTheProfilesItUpdates:
    def test_it_decides_on_the_profiles_it_is_handed(self, tree: Path) -> None:
        with_plane_settings(tree, profile_update_interval_s=30)
        config = load_control_plane_config(tree)
        database = stored()
        updated = stored()
        started = AT + timedelta(minutes=10)
        arrived = started - timedelta(seconds=20)
        updated.write_query(
            served(
                query_id="q-0",
                t_arrive=arrived,
                t_result_ready=arrived + timedelta(milliseconds=174),
            )
        )
        updated.write_member_predictions(
            (
                prediction(
                    query_id="q-0",
                    member=EnsembleMember(
                        service_name="infer-b5-pi5", model="b5", device="raspberry_pi5"
                    ),
                    in_ensemble=False,
                ),
            )
        )
        service = InProcessObservationService(
            updated,
            run_id="r1",
            rates=config.cost.rates(),
            window_s=120.0,
            min_labeled_per_class=1,
            min_labeled_overall=1,
        )
        clock = iter([started, started + timedelta(seconds=10), started + timedelta(seconds=40)])
        serving = build(
            config,
            database,
            now=lambda: next(clock),
            profiles=ProfileUpdatesClient(
                database=database,
                base_url="http://observation-service:8083",
                interval_s=30.0,
                request_timeout_ms=500.0,
                transport=service,
            ),
        )

        serving.decide("c1_navigation")
        serving.decide("c1_navigation")

        latest = {one.service_name: one for one in updated.read_profiles()}
        assert latest["infer-b5-pi5"].window_end == started + timedelta(seconds=40)
        assert len(database.read_profiles()) == len(POOL)


class TestTheQueriesItRemembersForTheirLabels:
    def test_a_consumer_remembers_every_query_its_labels_can_still_name(self, tree: Path) -> None:
        assert remembered_queries(load_control_plane_config(tree)) == 15 * 5 + 64

    def test_a_variant_contract_with_a_longer_delay_is_remembered_for(self, tree: Path) -> None:
        contract = tree / "consumers" / "c2_safety" / "contract_only_general_accuracy.yaml"
        held = yaml.safe_load(contract.read_text(encoding="utf-8"))
        held["feedback"]["delay_intervals"] = 40
        contract.write_text(yaml.safe_dump(held), encoding="utf-8")

        assert remembered_queries(load_control_plane_config(tree)) == 40 * 5 + 64
