from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from eemls.config.reading import Reader
from eemls.config.services import MonitoringConfig
from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.contract import ConsumerContract, Requirement
from eemls.domain.cost import DeviceRate
from eemls.domain.records import Decision, Feedback
from eemls.domain.types import Action, EnsembleMember, Row
from eemls.ensemble_optimization.approach import (
    ConsumerSpecification,
    Observation,
    Policy,
    PolicyDecision,
)
from eemls.ensemble_optimization.approaches.baselines.keep_the_ensemble import KeepTheEnsemble
from eemls.ensemble_optimization.decision_loop import (
    DecisionLoop,
    NoContractError,
    utc_now,
)
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.profile_updates import ProfileUpdating
from eemls.observation_service.runtime_context import RuntimeContextReader, ServingContext
from eemls.observation_service.testing import FixedContext, in_process_profile_updates
from eemls.orchestration.ensemble_deployments import EnsembleDeployments
from eemls.orchestration.testing import AVAILABLE
from eemls.testing.records import AT, context, member, prediction, profile, served
from eemls.testing.trees import TREE

MONITORING = Reader(TREE).build(MonitoringConfig, "services/monitoring.yaml")


def committed_rates() -> Mapping[str, DeviceRate]:
    prices = yaml.safe_load((TREE / "cost" / "prices.yaml").read_text(encoding="utf-8"))
    return {
        name: DeviceRate(
            power_watts=device["peak_power_watts"],
            hardware_price=device["hardware_price_eur"],
            lifetime_years=prices["device_lifetime_years"],
            energy_price_per_kwh=prices["energy_price_per_kwh"],
            eur_per_cost_unit=prices["eur_per_cost_unit"],
        )
        for name, device in prices["devices"].items()
    }


CDMS = CdmSet.model_validate(
    {
        "definitions": {
            "accuracy_car": {
                "measured": "per_window",
                "window": 500,
                "unit": "ratio",
                "description": "Accuracy on car queries",
                "specification": {
                    "function": "ratio",
                    "inputs": [
                        {
                            "function": "count",
                            "scopes": [
                                {"attribute": "true_label", "value": "car"},
                                {"attribute": "predicted_label", "same_as": "true_label"},
                            ],
                        },
                        {
                            "function": "count",
                            "scopes": [{"attribute": "true_label", "value": "car"}],
                        },
                    ],
                },
            }
        }
    }
)

CONTRACT = ConsumerContract.model_validate(
    {
        "consumer_id": "c1_navigation",
        "description": "A consumer that scores accuracy on cars",
        "prioritized_classes": ["car"],
        "request_rate_per_s": 20.0,
        "objectives": {"accuracy_car": {"weight": 10.0}},
        "cost_weight": 1.0,
        "cost_range_max_per_1k": 30.0,
        "feedback": {"fraction": 0.5, "delay_intervals": 15},
    }
)


class Decides:
    name = "test_policy"

    def __init__(self, decided: PolicyDecision | None = None) -> None:
        self.decided = decided or PolicyDecision(action=Action(kind="keep"))
        self.saw: list[Observation] = []

    def decide(self, observed: Observation) -> PolicyDecision:
        self.saw.append(observed)
        return self.decided


class Accounts:
    name = "accounts"

    def decide(self, observed: Observation) -> PolicyDecision:
        contributing = time.perf_counter()
        contribution_ms = (time.perf_counter() - contributing) * 1000.0
        selecting = time.perf_counter()
        select_ms = (time.perf_counter() - selecting) * 1000.0
        return PolicyDecision(
            action=Action(kind="keep"),
            candidates=7,
            contribution_ms=contribution_ms,
            select_ms=select_ms,
        )


class Reads(MemoryMetricsDatabase):
    def __init__(self, *, spent_ms: float) -> None:
        super().__init__()
        self.spent_ms = spent_ms

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        until = time.perf_counter() + self.spent_ms / 1000.0
        while time.perf_counter() < until:
            pass
        return super().read_window(consumer_id, limit)


class ReadsAnswers(MemoryMetricsDatabase):
    def __init__(self, *, spent_ms: float) -> None:
        super().__init__()
        self.spent_ms = spent_ms

    def read_member_answers(self, *, since: datetime, until: datetime) -> Sequence[Row]:
        ending = time.perf_counter() + self.spent_ms / 1000.0
        while time.perf_counter() < ending:
            pass
        return super().read_member_answers(since=since, until=until)


class Writes:
    name = "writes"

    def decide(self, observed: Observation) -> PolicyDecision:
        with contextlib.suppress(TypeError):
            observed.cdm_values["accuracy_car"] = 0.99  # type: ignore[index]
        return PolicyDecision(action=Action(kind="keep"))


class Built:
    def __init__(
        self,
        orchestrator: DecisionLoop,
        deployments: EnsembleDeployments,
        database: MemoryMetricsDatabase,
        policy: Policy,
        monitoring: Monitoring,
    ) -> None:
        self.orchestrator = orchestrator
        self.deployments = deployments
        self.database = database
        self.policy = policy
        self.monitoring = monitoring

    def names(self, consumer_id: str = "c1_navigation") -> tuple[str, ...]:
        return tuple(one.service_name for one in self.deployments.current(consumer_id).members)

    @property
    def shown(self) -> list[Observation]:
        assert isinstance(self.policy, Decides)
        return self.policy.saw

    def decisions(self) -> list[Decision]:
        return list(self.database.decisions.values())


def held_profiles(database: MemoryMetricsDatabase) -> ProfileUpdating:
    return in_process_profile_updates(
        database,
        rates=committed_rates(),
        window_s=120.0,
        interval_s=30.0,
        min_labeled_per_class=1,
        min_labeled_overall=1,
    )


def built(
    *,
    policy: Policy | None = None,
    serving: tuple[str, ...] = ("b5-cpu", "b6-gpu"),
    consumers: tuple[str, ...] = ("c1_navigation",),
    queries: int = 4,
    available: tuple[EnsembleMember, ...] = AVAILABLE,
    answered: bool = True,
    labeled: bool = True,
    database: MemoryMetricsDatabase | None = None,
    cdms: CdmSet = CDMS,
    contract: ConsumerContract = CONTRACT,
    now: Callable[[], datetime] | None = None,
    policy_window: int = 0,
    runtime_context: ServingContext | None = None,
    profiles: ProfileUpdating | None = None,
) -> Built:
    deployments = EnsembleDeployments(available_inference_services=available, max_ensemble_size=3)
    for consumer_id in consumers:
        deployments.register(consumer_id=consumer_id, service_names=serving)
    database = database if database is not None else MemoryMetricsDatabase()
    for consumer_id in consumers:
        for index in range(queries):
            arrived = AT + timedelta(seconds=index)
            answer: dict[str, object] = (
                {
                    "t_result_ready": arrived + timedelta(milliseconds=174),
                    "predicted_label": "car" if index % 2 == 0 else "bus",
                }
                if answered
                else {
                    "outcome": "failed",
                    "t_result_ready": None,
                    "predicted_label": None,
                    "predicted_confidence": None,
                    "deployment_id": None,
                    "n_members": None,
                    "n_answered": None,
                    "wait_ms": None,
                    "aggregate_ms": None,
                    "cost_per_1k": None,
                    "throughput_rps": None,
                }
            )
            database.write_query(
                served(
                    query_id=f"q-{consumer_id}-{index}",
                    consumer_id=consumer_id,
                    t_arrive=arrived,
                    **answer,
                )
            )
            database.write_member_predictions(
                (prediction(query_id=f"q-{consumer_id}-{index}", member=member()),)
            )
            if not labeled:
                continue
            database.write_feedback(
                Feedback(
                    query_id=f"q-{consumer_id}-{index}",
                    consumer_id=consumer_id,
                    true_label="car",
                    t_feedback=arrived,
                    delay_ms=1.0,
                )
            )
    monitoring = Monitoring.from_config(MONITORING)
    held = policy or Decides()
    orchestrator = DecisionLoop(
        deployments=deployments,
        database=database,
        policy=held,
        consumers={one: ConsumerSpecification(contract=contract, cdms=cdms) for one in consumers},
        monitoring=monitoring,
        runtime_context=runtime_context if runtime_context is not None else FixedContext(),
        now=now or (lambda: AT),
        policy_window=policy_window,
        profiles=profiles if profiles is not None else held_profiles(database),
    )
    return Built(orchestrator, deployments, database, held, monitoring)


class TestOneDecision:
    def test_a_decision_is_stored(self) -> None:
        held = built()

        decision = held.orchestrator.decide("c1_navigation")

        assert held.decisions() == [decision]

    def test_a_decision_names_the_policy_that_took_it(self) -> None:
        held = built(policy=KeepTheEnsemble())

        assert held.orchestrator.decide("c1_navigation").policy == "keep_the_ensemble"

    def test_every_decision_stores_a_row_of_its_own(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")
        held.orchestrator.decide("c1_navigation")
        held.orchestrator.decide("c1_navigation")

        assert len({one.decision_id for one in held.decisions()}) == 3


class TestNoChange:
    def test_keeping_the_ensemble_is_a_decision_and_is_stored(self) -> None:
        held = built(policy=KeepTheEnsemble())

        decision = held.orchestrator.decide("c1_navigation")

        assert (decision.action.kind, decision.accepted) == ("keep", True)
        assert decision.deployment_before == decision.deployment_after
        assert decision.reason

    def test_the_ensemble_is_where_it_was(self) -> None:
        held = built(policy=KeepTheEnsemble())

        held.orchestrator.decide("c1_navigation")

        assert held.names() == ("b5-cpu", "b6-gpu")


class TestAnActionTheEnsembleTakes:
    def test_the_member_joins_and_the_decision_records_both_versions(self) -> None:
        held = built(policy=Decides(PolicyDecision(action=Action(kind="add", member_in="b7-gpu"))))

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.accepted
        assert held.names() == ("b5-cpu", "b6-gpu", "b7-gpu")
        assert decision.deployment_before != decision.deployment_after

    def test_what_the_policy_accounted_for_is_stored(self) -> None:
        held = built(policy=Accounts())

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.candidates == 7
        assert decision.scored is None
        assert decision.contribution_ms is not None and decision.contribution_ms > 0.0
        assert decision.select_ms is not None and decision.select_ms > 0.0

    def test_a_policy_claiming_more_time_than_it_took_still_leaves_a_decision(self) -> None:
        held = built(
            policy=Decides(
                PolicyDecision(
                    action=Action(kind="keep"),
                    candidates=7,
                    contribution_ms=500.0,
                    select_ms=250.0,
                )
            )
        )

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.candidates == 7
        assert (decision.contribution_ms, decision.select_ms) == (None, None)

    def test_the_count_of_scored_deployments_is_stored_beside_the_candidates(self) -> None:
        held = built(
            policy=Decides(PolicyDecision(action=Action(kind="keep"), candidates=19, scored=8))
        )

        decision = held.orchestrator.decide("c1_navigation")

        assert (decision.candidates, decision.scored) == (19, 8)


class TestAnActionTheEnsembleRefuses:
    @pytest.mark.parametrize(
        "action",
        [
            Action(kind="add", member_in="b5-cpu"),
            Action(kind="add", member_in="unknown-gpu"),
            Action(kind="remove", member_out="b7-gpu"),
        ],
    )
    def test_it_is_stored_with_the_reason_and_changes_nothing(self, action: Action) -> None:
        held = built(policy=Decides(PolicyDecision(action=action)))
        before = held.names()

        decision = held.orchestrator.decide("c1_navigation")

        assert not decision.accepted
        assert decision.reason
        assert decision.deployment_before == decision.deployment_after
        assert held.names() == before

    def test_the_action_the_policy_wanted_is_what_is_stored(self) -> None:
        held = built(
            policy=Decides(PolicyDecision(action=Action(kind="add", member_in="unknown-gpu")))
        )

        decision = held.orchestrator.decide("c1_navigation")

        assert (decision.action.kind, decision.action.member_in) == ("add", "unknown-gpu")


class TestWhatThePolicySees:
    def test_it_is_shown_the_ensemble_serving_and_the_services_available(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        observed = held.shown[0]
        assert observed.deployment == held.deployments.current("c1_navigation")
        assert observed.available == held.deployments.available_inference_services

    def test_it_is_shown_what_each_member_itself_answered(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        observed = held.shown[0]
        assert len(observed.member_window) == 4
        assert {str(row["query_id"]) for row in observed.member_window} == {
            f"q-c1_navigation-{index}" for index in range(4)
        }

    def test_the_member_window_reaches_back_as_far_as_the_cdms_declare(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        observed = held.shown[0]
        assert len(observed.window) >= len({str(row["query_id"]) for row in observed.member_window})

    def test_it_is_shown_the_profile_of_every_inference_service(self) -> None:
        database = MemoryMetricsDatabase()
        database.write_profile(profile(service_name="b5-cpu", instance_id="i-1"))
        database.write_profile(profile(service_name="b6-gpu", instance_id="i-2"))
        held = built(database=database)

        held.orchestrator.decide("c1_navigation")

        observed = held.shown[0]
        assert set(observed.profiles) == {"b5-cpu", "b6-gpu"}
        assert observed.profiles["b5-cpu"].confidence_by_class

    def test_the_newest_profile_of_a_service_with_two_copies_is_the_one_shown(self) -> None:
        database = MemoryMetricsDatabase()
        database.write_profile(
            profile(service_name="b5-cpu", instance_id="i-1", general_accuracy=0.60)
        )
        database.write_profile(
            profile(
                service_name="b5-cpu",
                instance_id="i-2",
                window_start=AT + timedelta(minutes=10),
                window_end=AT + timedelta(minutes=15),
                general_accuracy=0.90,
            )
        )
        held = built(database=database)

        held.orchestrator.decide("c1_navigation")

        shown = held.shown[0].profiles["b5-cpu"]
        assert shown.general_accuracy == pytest.approx(0.90)
        assert shown.instance_id == "i-2"

    def test_it_is_shown_the_profile_the_observation_service_updated(self) -> None:
        database = MemoryMetricsDatabase()
        database.write_profile(
            profile(
                service_name="infer-b6-jetson_orin_nano",
                window_start=AT - timedelta(minutes=10),
                window_end=AT - timedelta(minutes=5),
                general_accuracy=0.30,
            )
        )
        clock = iter([AT, AT + timedelta(seconds=60)])
        held = built(
            database=database,
            available=(*AVAILABLE, member()),
            now=lambda: next(clock),
            profiles=held_profiles(database),
        )

        held.orchestrator.decide("c1_navigation")
        held.orchestrator.decide("c1_navigation")

        assert held.shown[0].profiles["infer-b6-jetson_orin_nano"].general_accuracy == 0.30
        assert held.shown[1].profiles["infer-b6-jetson_orin_nano"].general_accuracy == 1.0

    def test_a_service_with_no_profile_is_simply_absent(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        assert held.shown[0].profiles == {}

    def test_it_is_shown_the_values_read_from_the_window_and_what_they_scored(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        observed = held.shown[0]
        assert observed.cdm_values["accuracy_car"] == pytest.approx(0.5)
        assert observed.score is not None

    def test_the_context_is_named_from_the_window_it_just_read(self) -> None:
        held = built(
            queries=3, runtime_context=RuntimeContextReader(dominant_classes=("car", "bus"))
        )

        held.orchestrator.decide("c1_navigation")

        assert held.shown[0].context.period_name == "car"
        assert held.shown[0].context.period_index == 1

    def test_the_context_the_decision_carries_is_the_one_the_policy_was_shown(self) -> None:
        held = built(
            queries=3, runtime_context=RuntimeContextReader(dominant_classes=("car", "bus"))
        )

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.context == held.shown[0].context

    def test_a_consumer_does_not_take_the_period_another_consumer_named(self) -> None:
        database = MemoryMetricsDatabase()
        database.write_query(
            served(
                query_id="q-c2_safety-late",
                consumer_id="c2_safety",
                t_arrive=AT + timedelta(minutes=1),
                t_result_ready=AT + timedelta(minutes=1, milliseconds=174),
                predicted_label="bus",
            )
        )
        held = built(
            queries=3,
            consumers=("c1_navigation", "c2_safety"),
            database=database,
            runtime_context=RuntimeContextReader(dominant_classes=("car", "bus")),
        )

        held.orchestrator.decide("c1_navigation")
        held.orchestrator.decide("c2_safety")

        navigation, safety = (one.context for one in held.shown)
        assert navigation.period_name == "car"
        assert safety.period_name == "unnamed"

    def test_it_is_shown_the_runtime_context_of_the_moment(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        assert held.shown[0].context == context()


class TestThePoolUnderAContractWithRequirements:
    def test_every_service_of_the_pool_is_shown_with_its_profile_and_may_join(self) -> None:
        database = MemoryMetricsDatabase()
        for one in AVAILABLE:
            database.write_profile(
                profile(service_name=one.service_name, instance_id=f"i-{one.service_name}")
            )
        database.write_member_predictions(
            tuple(
                prediction(
                    query_id=f"q-c1_navigation-{index}",
                    member=AVAILABLE[3],
                    in_ensemble=False,
                    predicted_label="bus",
                )
                for index in range(4)
            )
        )
        contract = CONTRACT.model_copy(
            update={
                "requirements": (
                    Requirement(
                        cdm="accuracy_car",
                        operator=">=",
                        target_value=0.8,
                    ),
                )
            }
        )
        held = built(
            database=database,
            contract=contract,
            policy=Decides(PolicyDecision(action=Action(kind="add", member_in="b7-gpu"))),
        )

        decision = held.orchestrator.decide("c1_navigation")

        pool = {one.service_name for one in AVAILABLE}
        assert {one.service_name for one in held.shown[0].available} == pool
        assert set(held.shown[0].profiles) == pool
        assert decision.accepted
        assert held.names() == ("b5-cpu", "b6-gpu", "b7-gpu")


class TestWhatThePolicySeesOfTheServicesOutsideTheEnsemble:
    def test_their_answers_are_not_in_the_member_window_it_is_shown(self) -> None:
        database = MemoryMetricsDatabase()
        database.write_member_predictions(
            (prediction(query_id="q-c1_navigation-0", member=AVAILABLE[3], in_ensemble=False),)
        )
        held = built(database=database)

        held.orchestrator.decide("c1_navigation")

        assert all(row["in_ensemble"] for row in held.shown[0].member_window)
        assert len(held.shown[0].member_window) == 4


class TestWhatIsStoredAboutTheWindow:
    def test_the_values_the_window_answered_are_on_the_row(self) -> None:
        held = built()

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.cdm_values["accuracy_car"] == pytest.approx(0.5)

    def test_the_score_of_the_window_is_on_the_row(self) -> None:
        held = built()

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.score is not None


class TestTheEdgeOfTheWindow:
    def test_a_consumer_with_no_query_yet_still_decides(self) -> None:
        held = built(queries=0)

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.accepted
        assert decision.score is None
        assert dict(decision.cdm_values) == {}

    def test_a_window_shorter_than_the_one_declared_is_read_as_it_is(self) -> None:
        held = built(queries=2)

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.cdm_values["accuracy_car"] == pytest.approx(0.5)

    def test_a_policy_shown_an_empty_window_is_shown_it_empty(self) -> None:
        held = built(queries=0)

        held.orchestrator.decide("c1_navigation")

        observed = held.shown[0]
        assert len(observed.window) == 0
        assert observed.score is None


class TestWhatDecidingCost:
    def test_reading_the_metrics_is_measured_apart_from_reading_the_store(self) -> None:
        slow = Reads(spent_ms=5.0)
        held = built(database=slow)

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.cdm_ms is not None
        assert decision.cdm_ms < slow.spent_ms
        assert decision.decide_ms > slow.spent_ms

    def test_updating_the_profiles_is_not_counted_in_the_decision(self) -> None:
        slow = ReadsAnswers(spent_ms=200.0)
        clock = iter([AT, AT + timedelta(seconds=40)])
        held = built(
            database=slow,
            now=lambda: next(clock),
            profiles=in_process_profile_updates(
                slow,
                rates=committed_rates(),
                window_s=120.0,
                interval_s=30.0,
                min_labeled_per_class=1,
                min_labeled_overall=1,
            ),
        )

        held.orchestrator.decide("c1_navigation")
        decision = held.orchestrator.decide("c1_navigation")

        assert decision.decide_ms < slow.spent_ms

    def test_the_whole_decision_is_published_as_well_as_stored(self) -> None:
        held = built()

        held.orchestrator.decide("c1_navigation")

        assert "eemls_decide_seconds_count" in held.monitoring.exposition()[0]


class TestANumberAPolicyCannotHaveMeasured:
    @pytest.mark.parametrize("part", [-1.0, float("nan"), float("inf"), "soon"])
    def test_a_time_that_is_no_length_of_time_is_left_off_the_row(self, part: object) -> None:
        held = built(
            policy=Decides(
                PolicyDecision(
                    action=Action(kind="add", member_in="b7-gpu"),
                    contribution_ms=part,  # type: ignore[arg-type]
                )
            )
        )

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.contribution_ms is None
        assert decision.accepted
        assert held.names() == ("b5-cpu", "b6-gpu", "b7-gpu")

    @pytest.mark.parametrize("counted", [-1, 7.5, True])
    def test_a_count_of_candidates_that_is_no_count_is_stored_as_none_of_them(
        self, counted: object
    ) -> None:
        held = built(policy=Decides(PolicyDecision(action=Action(kind="keep"), candidates=counted)))  # type: ignore[arg-type]

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.candidates == 0

    @pytest.mark.parametrize("counted", [-1, 7.5, True])
    def test_a_count_of_scored_deployments_that_is_no_count_is_left_off_the_row(
        self, counted: object
    ) -> None:
        held = built(policy=Decides(PolicyDecision(action=Action(kind="keep"), scored=counted)))  # type: ignore[arg-type]

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.scored is None

    def test_the_decision_is_stored_and_the_ensemble_moved_all_the_same(self) -> None:
        held = built(
            policy=Decides(
                PolicyDecision(action=Action(kind="add", member_in="b7-gpu"), contribution_ms=-1.0)
            )
        )

        held.orchestrator.decide("c1_navigation")

        assert len(held.decisions()) == 1
        assert held.names() == ("b5-cpu", "b6-gpu", "b7-gpu")


def narrowed(window: int) -> CdmSet:
    held = CDMS.model_dump()
    held["definitions"]["accuracy_car"]["window"] = window
    return CdmSet.model_validate(held)


class TestHowFarBackItReads:
    def test_the_approach_is_shown_its_own_window_and_a_cdm_keeps_its_own(self) -> None:
        held = built(queries=3, cdms=narrowed(2), policy_window=500)

        decision = held.orchestrator.decide("c1_navigation")

        assert len(held.shown[0].window) == 3
        assert len(held.shown[0].member_window) == 3
        assert decision.cdm_values["accuracy_car"] == pytest.approx(0.5)

    def test_a_metric_declared_over_two_queries_reads_the_two_most_recent(self) -> None:
        held = built(queries=3, cdms=narrowed(2))

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.cdm_values["accuracy_car"] == pytest.approx(0.5)

    def test_the_same_metric_over_every_query_answers_for_every_query(self) -> None:
        held = built(queries=3)

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.cdm_values["accuracy_car"] == pytest.approx(2 / 3)


class TestWhatTheWindowScored:
    def test_the_score_is_the_weighted_objective_less_the_weighted_cost(self) -> None:
        held = built()

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.score == pytest.approx(10.0 * 0.5 - 1.0 * (10.5 / 30.0))


class TestAWindowWithNoTrueLabelYet:
    def test_the_queries_are_there_before_the_labels_and_it_still_decides(self) -> None:
        held = built(labeled=False)

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.accepted
        assert decision.score is None

    def test_the_policy_is_shown_the_window_without_a_score(self) -> None:
        held = built(labeled=False)

        held.orchestrator.decide("c1_navigation")

        assert len(held.shown[0].window) == 4
        assert held.shown[0].score is None


class TestTheClockItKeepsWhenNobodyGivesItOne:
    def test_the_instant_a_decision_carries_has_a_time_zone(self) -> None:
        before = datetime.now(UTC)
        held = built(now=utc_now)

        decision = held.orchestrator.decide("c1_navigation")

        assert before <= decision.t <= datetime.now(UTC)


class TestAWindowNobodyWasChargedFor:
    def test_a_window_of_queries_nobody_answered_still_decides(self) -> None:
        held = built(answered=False)

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.accepted
        assert decision.score is None
        assert dict(decision.cdm_values) == {}

    def test_one_consumer_it_cannot_price_does_not_silence_the_others(self) -> None:
        held = built(consumers=("c1_navigation", "c2_safety"), answered=False)

        decisions = [held.orchestrator.decide(one) for one in ("c1_navigation", "c2_safety")]

        assert [one.consumer_id for one in decisions] == ["c1_navigation", "c2_safety"]


class TestWhatThePolicyCannotChange:
    def test_a_policy_that_writes_into_what_it_was_shown_changes_no_row(self) -> None:
        held = built(policy=Writes())

        decision = held.orchestrator.decide("c1_navigation")

        assert decision.cdm_values["accuracy_car"] == pytest.approx(0.5)

    def test_it_is_shown_the_instant_the_decision_carries(self) -> None:
        held = built()

        decision = held.orchestrator.decide("c1_navigation")

        assert held.shown[0].t == decision.t


class TestAConsumerItDoesNotKnow:
    def test_a_consumer_that_declared_a_contract_but_is_served_by_nothing_is_refused(
        self,
    ) -> None:
        deployments = EnsembleDeployments(
            available_inference_services=AVAILABLE, max_ensemble_size=3
        )
        deployments.register(consumer_id="c1_navigation", service_names=("b5-cpu",))

        with pytest.raises(NoContractError, match="c2_safety"):
            DecisionLoop(
                deployments=deployments,
                database=MemoryMetricsDatabase(),
                policy=KeepTheEnsemble(),
                consumers={
                    one: ConsumerSpecification(contract=CONTRACT, cdms=CDMS)
                    for one in ("c1_navigation", "c2_safety")
                },
                monitoring=Monitoring.from_config(MONITORING),
                runtime_context=FixedContext(),
                policy_window=0,
                profiles=held_profiles(MemoryMetricsDatabase()),
            )


class TestWhatItPublishes:
    def test_the_decision_the_ensemble_size_and_the_values_appear(self) -> None:
        held = built(policy=KeepTheEnsemble())

        held.orchestrator.decide("c1_navigation")

        body = held.monitoring.exposition()[0]
        assert (
            'eemls_decisions_total{accepted="true",action="keep",consumer="c1_navigation",'
            'policy="keep_the_ensemble"} 1.0' in body
        )
        assert 'eemls_ensemble_size{consumer="c1_navigation"} 2.0' in body
        assert 'eemls_cdm_value{consumer="c1_navigation",metric="accuracy_car"} 0.5' in body
