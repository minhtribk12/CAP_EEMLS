from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, tzinfo
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from eemls.domain.errors import EemlsError
from eemls.domain.records import Decision, Feedback, InferenceServiceProfile, Instance, ServedQuery
from eemls.domain.types import (
    Action,
    AggregatedResult,
    EnsembleDeployment,
    EnsembleMember,
    FrozenMap,
    FrozenModel,
    MemberPrediction,
    Query,
    RuntimeContext,
)
from eemls.orchestration.ensemble_deployments import EnsembleDeployments

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


class MetricWeights(FrozenModel):
    by_class: FrozenMap[object]
    by_consumer: FrozenMap[FrozenMap[float]]


def query_at(when: datetime) -> Query:
    return Query(
        query_id="q-1",
        consumer_id="c1_navigation",
        image_id="img-7",
        reply_topic="eemls/results/c1_navigation",
        correlation_id="corr-1",
        t_arrive=when,
    )


def decision_with(**changes: object) -> Decision:
    fields: dict[str, object] = {
        "decision_id": "d-1",
        "consumer_id": "c1_navigation",
        "t": NOW,
        "policy": "cap",
        "action": Action(kind="keep"),
        "context": context(),
        "accepted": False,
        "reason": "no candidate cleared the margin",
        "deployment_before": "v-1",
        "deployment_after": "v-1",
        "score": 0.81,
        "cdm_values": {"accuracy_car": 0.86},
        "candidates": 5,
        "scored": None,
        "decide_ms": 38.0,
        "cdm_ms": 20.0,
        "contribution_ms": 10.0,
        "select_ms": 5.0,
    }
    return Decision.model_validate(fields | changes)


def prediction_with(**changes: object) -> MemberPrediction:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "instance_id": "i-1",
        "member": member(),
        "node": "orin-1",
        "replica": 1,
        "in_ensemble": True,
        "predicted_label": "car",
        "confidence": 0.9,
        "fetch_ms": 2.0,
        "inference_ms": 18.0,
        "total_ms": 21.0,
        "provider": "CUDAExecutionProvider",
        "number_format": "float32",
        "cache_hit": True,
        "status": "ok",
    }
    return MemberPrediction.model_validate(fields | changes)


def profile_with(**changes: object) -> InferenceServiceProfile:
    fields: dict[str, object] = {
        "model": "b6",
        "device": "jetson_orin_nano",
        "node": "orin-1",
        "instance_id": "i-1",
        "service_name": "infer-b6-orin",
        "window_start": NOW,
        "window_end": NOW.replace(minute=5),
        "mean_ms": 42.0,
        "p50_ms": 40.0,
        "p95_ms": 61.0,
        "max_ms": 88.0,
        "throughput_rps": 23.5,
        "n_queries": 500,
        "energy_wh_per_1k": 1.8,
        "cost_per_1k": 3.5,
        "general_accuracy": 0.81,
        "accuracy_by_class": {"car": 0.9, "truck": 0.72},
        "confidence_by_class": {"car": 0.88, "truck": 0.64},
        "source": "measured",
    }
    return InferenceServiceProfile.model_validate(fields | changes)


def member(model: str = "b6", device: str = "jetson_orin_nano") -> EnsembleMember:
    return EnsembleMember(service_name=f"infer-{model}-{device}", model=model, device=device)


def context() -> RuntimeContext:
    return RuntimeContext(
        period_name="evening_rush", period_index=3, condition="night", weather="fog"
    )


def query() -> Query:
    return Query(
        query_id="q-1",
        consumer_id="c1_navigation",
        image_id="img-7",
        reply_topic="eemls/results/c1_navigation",
        correlation_id="corr-1",
        t_arrive=NOW,
    )


class TestNothingCanBeChanged:
    def test_a_field_cannot_be_changed(self) -> None:
        with pytest.raises(ValidationError, match="frozen"):
            query().consumer_id = "c2_safety"  # type: ignore[misc]

    def test_a_nested_field_cannot_be_changed(self) -> None:
        version = EnsembleDeployment(deployment_id="v-1", members=(member(), member("b3")))

        with pytest.raises(ValidationError, match="frozen"):
            version.members[0].model = "b8"  # type: ignore[misc]

    def test_a_member_cannot_be_added_to_a_version(self) -> None:
        version = EnsembleDeployment(deployment_id="v-1", members=(member(),))

        with pytest.raises(AttributeError):
            version.members.append(member("b3"))  # type: ignore[attr-defined]


class TestTheRuntimeContext:
    def test_a_condition_given_as_anything_but_its_own_value_is_refused(self) -> None:
        given = RuntimeContext.model_validate(
            {
                "period_name": "evening_rush",
                "period_index": 3,
                "condition": "night",
                "weather": "fog",
            }
        )

        assert given.condition == "night"

    @pytest.mark.parametrize(
        ("condition", "weather"), [("dusk", "fog"), ("night", "hail"), ("", "fog")]
    )
    def test_a_condition_or_weather_outside_the_set_is_refused(
        self, condition: str, weather: str
    ) -> None:
        with pytest.raises(ValidationError):
            RuntimeContext(
                period_name="evening_rush",
                period_index=3,
                condition=condition,
                weather=weather,
            )

    def test_an_event_is_optional_and_absent_by_default(self) -> None:
        assert context().event_kind is None
        assert context().event_id is None


class TestAnEnsembleVersion:
    def test_two_versions_of_the_same_members_in_any_order_are_equal(self) -> None:
        first = EnsembleDeployment(deployment_id="v-1", members=(member("a"), member("b")))
        second = EnsembleDeployment(deployment_id="v-1", members=(member("b"), member("a")))

        assert first == second
        assert first.members == second.members

    def test_a_repeated_member_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="more than once"):
            EnsembleDeployment(deployment_id="v-1", members=(member(), member()))

    def test_an_empty_version_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no member"):
            EnsembleDeployment(deployment_id="v-1", members=())

    def test_it_reports_how_many_members_serve(self) -> None:
        version = EnsembleDeployment(deployment_id="v-1", members=(member("a"), member("b")))

        assert version.n_members == 2


class TestAnAction:
    def test_doing_nothing_names_no_member(self) -> None:
        assert Action(kind="keep").member_in is None

    @pytest.mark.parametrize(
        ("kind", "fields", "refusal"),
        [
            ("add", {}, "member_in"),
            ("remove", {}, "member_out"),
            ("replace", {"member_in": "a"}, "member_out"),
            ("keep", {"member_in": "a"}, "names no member_in"),
        ],
    )
    def test_an_action_missing_what_it_needs_is_refused(
        self, kind: str, fields: dict[str, str], refusal: str
    ) -> None:
        with pytest.raises(ValidationError, match=refusal):
            Action(kind=kind, **fields)

    def test_a_kind_outside_the_five_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Action(kind="grow")

    def test_selecting_an_ensemble_names_the_members_that_will_serve(self) -> None:
        held = Action(kind="select", members=("infer-a", "infer-c"))

        assert held.members == ("infer-a", "infer-c")
        assert (held.member_in, held.member_out) == (None, None)

    def test_selecting_no_member_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="members"):
            Action(kind="select")

    def test_selecting_while_naming_one_member_to_change_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="member_in"):
            Action(kind="select", members=("infer-a",), member_in="infer-c")

    @pytest.mark.parametrize("kind", ["keep", "add", "remove", "replace"])
    def test_an_action_that_changes_one_member_names_no_whole_ensemble(self, kind: str) -> None:
        fields = {
            "add": {"member_in": "a"},
            "remove": {"member_out": "a"},
            "keep": {},
            "replace": {"member_in": "a", "member_out": "b"},
        }[kind]
        with pytest.raises(ValidationError, match="names no members"):
            Action(kind=kind, members=("infer-a",), **fields)

    def test_selecting_one_member_twice_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="more than once"):
            Action(kind="select", members=("infer-a", "infer-a"))


class TestAPrediction:
    def test_a_confidence_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            MemberPrediction(
                query_id="q-1",
                instance_id="i-1",
                member=member(),
                node="orin-1",
                replica=1,
                in_ensemble=True,
                predicted_label="car",
                confidence=1.4,
                fetch_ms=2.0,
                inference_ms=18.0,
                total_ms=21.0,
                provider="CUDAExecutionProvider",
                number_format="float32",
                cache_hit=True,
                status="ok",
            )

    def test_a_negative_duration_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            MemberPrediction(
                query_id="q-1",
                instance_id="i-1",
                member=member(),
                node="orin-1",
                replica=1,
                in_ensemble=True,
                predicted_label="car",
                confidence=0.9,
                fetch_ms=-1.0,
                inference_ms=18.0,
                total_ms=21.0,
                provider="CUDAExecutionProvider",
                number_format="float32",
                cache_hit=True,
                status="ok",
            )

    def test_a_member_that_failed_carries_no_label(self) -> None:
        prediction = MemberPrediction(
            query_id="q-1",
            instance_id="i-1",
            member=member(),
            node="orin-1",
            replica=1,
            in_ensemble=True,
            predicted_label=None,
            confidence=None,
            fetch_ms=2.0,
            inference_ms=0.0,
            total_ms=500.0,
            provider="CUDAExecutionProvider",
            number_format="float32",
            cache_hit=False,
            status="timeout",
        )

        assert prediction.status == "timeout"
        assert prediction.predicted_label is None

    def test_a_member_that_answered_must_carry_a_label(self) -> None:
        with pytest.raises(ValidationError, match="ok but carries no label"):
            MemberPrediction(
                query_id="q-1",
                instance_id="i-1",
                member=member(),
                node="orin-1",
                replica=1,
                in_ensemble=True,
                predicted_label=None,
                confidence=None,
                fetch_ms=2.0,
                inference_ms=18.0,
                total_ms=21.0,
                provider="CUDAExecutionProvider",
                number_format="float32",
                cache_hit=True,
                status="ok",
            )


class TestAVote:
    def test_it_names_the_version_that_produced_it(self) -> None:
        result = AggregatedResult(
            query_id="q-1",
            predicted_label="car",
            confidence=0.82,
            deployment_id="v-1",
            n_members=3,
            n_answered=2,
        )

        assert result.deployment_id == "v-1"
        assert (result.n_members, result.n_answered) == (3, 2)

    def test_a_result_from_no_member_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            AggregatedResult(
                query_id="q-1",
                predicted_label="car",
                confidence=0.82,
                deployment_id="v-1",
                n_members=0,
                n_answered=0,
            )

    def test_more_members_answering_than_serve_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="more than serve"):
            AggregatedResult(
                query_id="q-1",
                predicted_label="car",
                confidence=0.82,
                deployment_id="v-1",
                n_members=2,
                n_answered=3,
            )


class TestFeedback:
    def test_it_carries_how_late_it_was(self) -> None:
        feedback = Feedback(
            query_id="q-1",
            consumer_id="c1_navigation",
            true_label="truck",
            t_feedback=NOW,
            delay_ms=15000.0,
        )

        assert feedback.delay_ms == 15000.0

    def test_a_negative_delay_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Feedback(
                query_id="q-1",
                consumer_id="c1_navigation",
                true_label="truck",
                t_feedback=NOW,
                delay_ms=-1.0,
            )

    def test_an_empty_label_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Feedback(
                query_id="q-1",
                consumer_id="c1_navigation",
                true_label="",
                t_feedback=NOW,
                delay_ms=1.0,
            )


class TestADecision:
    def test_it_distinguishes_doing_nothing_from_having_nothing_to_do(self) -> None:
        nothing_to_do = decision_with(accepted=False, reason="no candidate cleared the margin")
        did_nothing = decision_with(accepted=True, reason="kept the ensemble")

        assert nothing_to_do.accepted is False
        assert did_nothing.accepted is True
        assert nothing_to_do.reason != did_nothing.reason

    def test_a_decision_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            decision_with(reason="")

    def test_a_changed_ensemble_must_change_the_version(self) -> None:
        with pytest.raises(ValidationError, match="version did not change"):
            decision_with(accepted=True, action=Action(kind="add", member_in="infer-a"))

    def test_selecting_the_members_already_serving_keeps_the_version(self) -> None:
        deployments = EnsembleDeployments(
            available_inference_services=(member("b6"), member("b7")), max_ensemble_size=2
        )
        served = deployments.register(
            consumer_id="c1_navigation", service_names=["infer-b6-jetson_orin_nano"]
        )
        selected = Action(kind="select", members=("infer-b6-jetson_orin_nano",))
        applied = deployments.apply(consumer_id="c1_navigation", action=selected)

        decision = decision_with(
            accepted=applied.accepted,
            action=selected,
            deployment_before=applied.before.deployment_id,
            deployment_after=applied.after.deployment_id,
        )

        assert decision.deployment_after == served.deployment_id


class TestTimestamps:
    def test_a_zone_that_reports_no_offset_is_refused(self) -> None:
        class NoOffset(tzinfo):
            def utcoffset(self, value: datetime | None) -> timedelta | None:
                return None

            def tzname(self, value: datetime | None) -> str | None:
                return "nowhere"

            def dst(self, value: datetime | None) -> timedelta | None:
                return None

        with pytest.raises(ValidationError, match="time zone"):
            query_at(datetime(2026, 10, 15, 12, 0, tzinfo=NoOffset()))

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="time zone"):
            Query(
                query_id="q-1",
                consumer_id="c1_navigation",
                image_id="img-7",
                reply_topic="eemls/results/c1_navigation",
                correlation_id="corr-1",
                t_arrive=datetime(2026, 10, 15, 12, 0),
            )


class TestEveryErrorHasOneRoot:
    def test_a_configuration_error_is_an_eemls_error(self) -> None:
        from eemls.config.reading import ConfigError

        assert issubclass(ConfigError, EemlsError)

    def test_a_metric_error_is_an_eemls_error(self) -> None:
        from eemls.monitoring.metrics import MetricError

        assert issubclass(MetricError, EemlsError)


def served(**changes: object) -> ServedQuery:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "consumer_id": "c1_navigation",
        "image_id": "img-7",
        "t_arrive": NOW,
        "t_result_ready": NOW.replace(microsecond=174000),
        "wait_ms": 160.0,
        "aggregate_ms": 1.5,
        "context": context(),
        "deployment_id": "v-1",
        "n_members": 3,
        "predicted_label": "car",
        "predicted_confidence": 0.82,
        "cost_per_1k": 10.5,
        "throughput_rps": 18.0,
        "outcome": "accepted",
    }
    return ServedQuery.model_validate(fields | changes)


class TestAServedQuery:
    def test_the_response_time_is_derived_from_the_two_instants(self) -> None:
        assert served().response_time_ms == pytest.approx(174.0)

    def test_a_result_cannot_be_ready_before_the_query_arrived(self) -> None:
        with pytest.raises(ValidationError, match="before"):
            served(t_result_ready=NOW.replace(second=0, microsecond=0, minute=59, hour=11))

    def test_the_parts_cannot_exceed_the_whole(self) -> None:
        with pytest.raises(ValidationError, match="longer than"):
            served(wait_ms=900.0)

    def test_an_accepted_query_carries_its_answer(self) -> None:
        query = served()

        assert query.predicted_label == "car"
        assert query.outcome == "accepted"

    def test_an_accepted_query_without_an_answer_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="accepted but carries no"):
            served(predicted_label=None, predicted_confidence=None)

    @pytest.mark.parametrize("outcome", ["refused", "failed"])
    def test_a_query_that_was_not_served_carries_no_answer(self, outcome: str) -> None:
        with pytest.raises(ValidationError, match="was not served"):
            served(outcome=outcome)

    def test_a_refused_query_needs_no_result_at_all(self) -> None:
        query = served(
            outcome="refused",
            predicted_label=None,
            predicted_confidence=None,
            t_result_ready=None,
            deployment_id=None,
            n_members=None,
            wait_ms=None,
            aggregate_ms=None,
            cost_per_1k=None,
            throughput_rps=None,
        )

        assert query.response_time_ms is None
        assert query.outcome == "refused"

    def test_it_carries_the_context_that_served_it(self) -> None:
        assert served().context.condition == "night"
        assert served().context.period_index == 3

    def test_the_broker_hop_is_reported_on_its_own(self) -> None:
        query = served(t_publish=NOW.replace(microsecond=180000))

        assert query.publish_ms == pytest.approx(6.0)
        assert query.response_time_ms == pytest.approx(174.0)


class TestAnInstance:
    def instance_with(self, **changes: object) -> Instance:
        fields: dict[str, object] = {
            "instance_id": "i-1",
            "service_name": "infer-b6-orin",
            "model": "b6",
            "device": "jetson_orin_nano",
            "node": "orin-1",
            "replica": 2,
            "provider": "TensorrtExecutionProvider",
            "number_format": "float32",
            "runtime_version": "1.20.1",
            "image_tag": "eemls/inference:jetson-abc123",
            "config_hash": "a" * 64,
            "started_at": NOW,
        }
        return Instance.model_validate(fields | changes)

    def test_it_names_what_actually_ran_the_model(self) -> None:
        instance = self.instance_with()

        assert instance.replica == 2
        assert instance.service_name == "infer-b6-orin"
        assert instance.stopped_at is None

    def test_it_cannot_stop_before_it_started(self) -> None:
        with pytest.raises(ValidationError, match="before"):
            self.instance_with(stopped_at=NOW.replace(hour=11))

    def test_a_configuration_hash_that_is_not_a_digest_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self.instance_with(config_hash="zz")


class TestAModelProfile:
    def test_it_carries_the_accuracy_per_class_a_specialist_is_visible_in(self) -> None:
        assert profile_with().accuracy_by_class["car"] == pytest.approx(0.9)

    def test_the_accuracy_per_class_cannot_be_changed(self) -> None:
        assert isinstance(profile_with().accuracy_by_class, MappingProxyType)

    def test_percentiles_out_of_order_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="rising order"):
            profile_with(p50_ms=70.0, mean_ms=70.0)

    def test_a_window_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no length"):
            profile_with(window_end=NOW.replace(hour=11))

    def test_a_carried_over_profile_says_so(self) -> None:
        assert profile_with(source="carried_over").source == "carried_over"


class TestWhatADecisionSaw:
    def test_it_keeps_the_cdm_values_the_policy_saw(self) -> None:
        assert decision_with().cdm_values["accuracy_car"] == pytest.approx(0.86)

    def test_the_time_split_cannot_exceed_the_whole_decision(self) -> None:
        with pytest.raises(ValidationError, match="longer than"):
            decision_with(cdm_ms=30.0, contribution_ms=20.0, select_ms=10.0)

    def test_the_cdm_values_cannot_be_changed(self) -> None:
        assert isinstance(decision_with().cdm_values, MappingProxyType)


class TestTheRemainingRefusals:
    def test_a_version_built_from_mappings_is_ordered_the_same_way(self) -> None:
        first = EnsembleMember(service_name="infer-a", model="a", device="d")
        second = EnsembleMember(service_name="infer-b", model="b", device="d")
        version = EnsembleDeployment.model_validate(
            {"deployment_id": "v-1", "members": [second.model_dump(), first.model_dump()]}
        )

        assert [member.service_name for member in version.members] == ["infer-a", "infer-b"]

    def test_a_version_given_something_that_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleDeployment.model_validate({"deployment_id": "v-1", "members": "infer-a"})

    def test_a_version_given_no_members_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="members"):
            EnsembleDeployment.model_validate({"deployment_id": "v-1"})

    def test_a_result_cannot_be_published_when_none_was_ready(self) -> None:
        with pytest.raises(ValidationError, match="although none was ready"):
            served(
                outcome="refused",
                predicted_label=None,
                predicted_confidence=None,
                t_result_ready=None,
                deployment_id=None,
                n_members=None,
                wait_ms=None,
                aggregate_ms=None,
                cost_per_1k=None,
                throughput_rps=None,
                t_publish=NOW,
            )

    def test_a_result_cannot_be_published_before_it_was_ready(self) -> None:
        with pytest.raises(ValidationError, match="before it was ready"):
            served(t_publish=NOW)


class TestWhatAFrozenValueStillAllows:
    def test_a_copy_with_changes_is_validated_like_a_new_value(self) -> None:
        decision = Decision.model_validate(
            {
                "decision_id": "d-1",
                "consumer_id": "c1_navigation",
                "t": NOW,
                "policy": "cap",
                "action": {"kind": "keep"},
                "context": context().model_dump(),
                "accepted": False,
                "reason": "no candidate cleared the margin",
                "deployment_before": "v-1",
                "deployment_after": "v-1",
                "score": 0.81,
                "cdm_values": {},
                "candidates": 5,
                "scored": None,
                "decide_ms": 38.0,
                "cdm_ms": 20.0,
                "contribution_ms": 10.0,
                "select_ms": 5.0,
            }
        )

        with pytest.raises(ValidationError, match="version did not change"):
            decision.model_copy(
                update={"accepted": True, "action": Action(kind="add", member_in="infer-a")}
            )

    def test_a_copy_without_changes_is_the_same_value(self) -> None:
        version = EnsembleDeployment(deployment_id="v-1", members=(member(),))

        assert version.model_copy() == version

    def test_a_copy_cannot_smuggle_in_a_repeated_member(self) -> None:
        version = EnsembleDeployment(deployment_id="v-1", members=(member("a"),))

        with pytest.raises(ValidationError, match="more than once"):
            version.model_copy(update={"members": (member("a"), member("a"))})


class TestWhatIsWrittenOut:
    def test_the_condition_is_written_out(self) -> None:
        assert context().model_dump()["condition"] == "night"

    def test_the_response_time_is_written_out(self) -> None:
        assert served().model_dump()["response_time_ms"] == pytest.approx(174.0)

    def test_the_number_of_members_is_written_out(self) -> None:
        version = EnsembleDeployment(deployment_id="v-1", members=(member("a"), member("b")))

        assert version.model_dump()["n_members"] == 2

    def test_a_value_holding_a_nested_mapping_can_be_written_out(self) -> None:
        profile = InferenceServiceProfile.model_validate(
            {
                "model": "b6",
                "device": "jetson_orin_nano",
                "node": "orin-1",
                "instance_id": "i-1",
                "service_name": "infer-b6-orin",
                "window_start": NOW,
                "window_end": NOW.replace(minute=5),
                "mean_ms": 42.0,
                "p50_ms": 40.0,
                "p95_ms": 61.0,
                "max_ms": 88.0,
                "throughput_rps": 23.5,
                "n_queries": 500,
                "energy_wh_per_1k": 1.8,
                "cost_per_1k": 3.5,
                "general_accuracy": 0.81,
                "accuracy_by_class": {"car": 0.9},
                "confidence_by_class": {"car": 0.88},
                "source": "measured",
            }
        )

        held = json.loads(profile.model_dump_json())

        assert held["accuracy_by_class"] == {"car": 0.9}
        assert held["confidence_by_class"] == {"car": 0.88}


class TestAnIdentifierIsNotBlank:
    @pytest.mark.parametrize("blank", [" ", "\t", "  "])
    def test_a_blank_label_is_refused(self, blank: str) -> None:
        with pytest.raises(ValidationError):
            Feedback(
                query_id="q-1",
                consumer_id="c1_navigation",
                true_label=blank,
                t_feedback=NOW,
                delay_ms=1.0,
            )

    def test_a_blank_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            decision_with(reason=" ")

    @pytest.mark.parametrize("blank", ["", " "])
    def test_a_member_that_answered_with_a_blank_label_is_refused(self, blank: str) -> None:
        with pytest.raises(ValidationError):
            prediction_with(predicted_label=blank)


class TestAnActionNamesExactlyWhatItChanges:
    @pytest.mark.parametrize("kind", ["add", "replace"])
    def test_a_blank_member_in_is_refused(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            Action(kind=kind, member_in="", member_out="infer-b")

    @pytest.mark.parametrize("named", ["", "infer-a"])
    def test_keeping_the_ensemble_while_naming_a_member_is_refused(self, named: str) -> None:
        with pytest.raises(ValidationError):
            Action(kind="keep", member_in=named)

    def test_adding_may_not_also_name_a_member_to_remove(self) -> None:
        with pytest.raises(ValidationError, match="names no member_out"):
            Action(kind="add", member_in="infer-a", member_out="infer-b")

    def test_removing_may_not_also_name_a_member_to_add(self) -> None:
        with pytest.raises(ValidationError, match="names no member_in"):
            Action(kind="remove", member_in="infer-a", member_out="infer-b")

    def test_replacing_a_member_with_itself_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="with itself"):
            Action(kind="replace", member_in="infer-a", member_out="infer-a")


class TestADecisionCannotContradictItself:
    def test_doing_nothing_cannot_change_the_version(self) -> None:
        with pytest.raises(ValidationError, match="nothing was done"):
            decision_with(accepted=True, deployment_after="v-2")

    def test_a_refused_action_cannot_change_the_version(self) -> None:
        with pytest.raises(ValidationError, match="nothing was done"):
            decision_with(
                accepted=False,
                action=Action(kind="add", member_in="infer-a"),
                deployment_after="v-2",
            )

    def test_the_time_split_must_be_measured_or_declared_absent(self) -> None:
        decision = decision_with(cdm_ms=None, contribution_ms=None, select_ms=None)

        assert decision.cdm_ms is None


class TestAMemberRowCannotBeImpossible:
    def test_the_parts_cannot_exceed_the_whole(self) -> None:
        with pytest.raises(ValidationError, match="longer than"):
            prediction_with(fetch_ms=10.0, inference_ms=10.0, total_ms=11.0)

    @pytest.mark.parametrize(
        ("fetch", "infer", "total"), [(0.1, 0.2, 0.3), (1.0, 2.0, 3.0), (0.0, 0.3, 0.3)]
    )
    def test_parts_that_sum_exactly_to_the_whole_are_accepted(
        self, fetch: float, infer: float, total: float
    ) -> None:
        held = prediction_with(fetch_ms=fetch, inference_ms=infer, total_ms=total)

        assert held.total_ms == pytest.approx(total)

    def test_a_member_that_answered_took_some_time(self) -> None:
        with pytest.raises(ValidationError, match="no time"):
            prediction_with(fetch_ms=0.0, inference_ms=0.0, total_ms=0.0)

    @pytest.mark.parametrize("status", ["timeout", "error", "refused"])
    def test_a_member_that_did_not_answer_carries_no_answer(self, status: str) -> None:
        with pytest.raises(ValidationError, match="did not answer"):
            prediction_with(status=status)

    @pytest.mark.parametrize("status", ["timeout", "error", "refused"])
    def test_a_member_that_did_not_answer_names_no_copy_of_itself(self, status: str) -> None:
        row = prediction_with(
            status=status,
            predicted_label=None,
            confidence=None,
            instance_id=None,
            node=None,
            replica=None,
            provider=None,
            number_format=None,
            cache_hit=None,
            inference_ms=0.0,
        )

        assert (row.instance_id, row.node, row.replica) == (None, None, None)
        assert (row.provider, row.number_format, row.cache_hit) == (None, None, None)

    @pytest.mark.parametrize(
        "unreported", ["instance_id", "node", "replica", "provider", "number_format", "cache_hit"]
    )
    def test_an_answer_that_names_no_copy_is_refused(self, unreported: str) -> None:
        with pytest.raises(ValidationError, match=unreported):
            prediction_with(**{unreported: None})


class TestAServedQueryExplainsItself:
    @pytest.mark.parametrize(
        "missing", ["wait_ms", "aggregate_ms", "cost_per_1k", "throughput_rps"]
    )
    def test_an_accepted_query_carries_every_part(self, missing: str) -> None:
        with pytest.raises(ValidationError, match="accepted but carries no"):
            served(**{missing: None})

    def test_the_broker_hop_is_derived_from_the_two_instants(self) -> None:
        query = served(t_publish=NOW.replace(microsecond=180000))

        assert query.publish_ms == pytest.approx(6.0)


class TestAProfileIsPossible:
    def test_a_mean_outside_the_range_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="mean"):
            profile_with(mean_ms=1000.0)

    def test_a_mean_below_the_median_is_accepted(self) -> None:
        assert profile_with(mean_ms=31.0).mean_ms == pytest.approx(31.0)

    def test_a_window_of_no_length_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no length"):
            profile_with(window_end=NOW)

    def test_throughput_without_a_query_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no query"):
            profile_with(n_queries=0)


class TestEveryValueSurvivesARoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            query(),
            served(),
            member(),
            EnsembleDeployment(deployment_id="v-1", members=(member("a"), member("b"))),
            Action(kind="replace", member_in="infer-a", member_out="infer-b"),
            prediction_with(),
            AggregatedResult(
                query_id="q-1",
                predicted_label="car",
                confidence=0.8,
                deployment_id="v-1",
                n_members=3,
                n_answered=3,
            ),
            Feedback(
                query_id="q-1", consumer_id="c1", true_label="truck", t_feedback=NOW, delay_ms=1.0
            ),
            decision_with(),
            profile_with(),
            context(),
        ],
        ids=lambda value: type(value).__name__,
    )
    def test_a_value_reads_back_from_what_it_wrote(self, value: FrozenModel) -> None:
        again = type(value).model_validate(json.loads(value.model_dump_json()))

        assert again == value


class TestTheFrozenHelpersAtEveryDepth:
    def test_a_nested_sequence_inside_a_mapping_is_frozen_and_reads_back(self) -> None:
        profile = profile_with(accuracy_by_class={"car": 0.9})
        written = json.loads(profile.model_dump_json())

        assert written["accuracy_by_class"] == {"car": 0.9}
        assert isinstance(profile.accuracy_by_class, MappingProxyType)

    def test_replacing_a_member_with_itself_is_refused_in_a_decision(self) -> None:
        with pytest.raises(ValidationError, match="with itself"):
            decision_with(
                accepted=True,
                action=Action(kind="replace", member_in="infer-a", member_out="infer-a"),
                deployment_after="v-2",
            )


class TestTheFreezeHelpersThemselves:
    def test_a_nested_sequence_in_a_mapping_freezes_to_a_tuple(self) -> None:
        held = MetricWeights.model_validate(
            {"by_class": {"vru": ["pedestrian", "cyclist"]}, "by_consumer": {}}
        )

        assert held.by_class["vru"] == ("pedestrian", "cyclist")
        assert type(held.by_class["vru"]) is tuple

    def test_a_nested_sequence_reads_back_as_a_list(self) -> None:
        held = MetricWeights.model_validate(
            {"by_class": {"vru": ["pedestrian", "cyclist"]}, "by_consumer": {}}
        )

        assert json.loads(held.model_dump_json())["by_class"] == {"vru": ["pedestrian", "cyclist"]}

    def test_a_mapping_inside_a_mapping_freezes_and_reads_back(self) -> None:
        held = MetricWeights.model_validate(
            {"by_class": {}, "by_consumer": {"c1_navigation": {"car": 4.0}}}
        )

        assert isinstance(held.by_consumer["c1_navigation"], MappingProxyType)
        assert json.loads(held.model_dump_json())["by_consumer"] == {"c1_navigation": {"car": 4.0}}

    def test_a_blank_identifier_is_refused_wherever_it_appears(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleMember(service_name=" ", model="a", device="d")


class TestTheQueryParts:
    @pytest.mark.parametrize(("wait", "aggregate"), [(0.2, 0.0), (160.0, 12.0)])
    def test_parts_that_sum_inside_the_whole_are_accepted(
        self, wait: float, aggregate: float
    ) -> None:
        held = served(wait_ms=wait, aggregate_ms=aggregate)

        assert held.response_time_ms == pytest.approx(174.0)

    def test_parts_that_sum_exactly_to_the_whole_are_accepted(self) -> None:
        held = served(wait_ms=173.9, aggregate_ms=0.1)

        assert held.response_time_ms == pytest.approx(174.0)

    def test_a_decision_whose_parts_sum_exactly_to_the_whole_is_accepted(self) -> None:
        held = decision_with(decide_ms=0.3, cdm_ms=0.1, contribution_ms=0.2, select_ms=0.0)

        assert held.decide_ms == pytest.approx(0.3)
