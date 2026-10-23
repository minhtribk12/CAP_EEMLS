from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

from eemls.domain.types import Action, EnsembleMember
from eemls.orchestration.ensemble_deployments import (
    ActionOutcome,
    DeploymentError,
    EnsembleDeployments,
    NoSuchConsumerError,
)
from eemls.orchestration.testing import AVAILABLE


def serving(*names: str, max_ensemble_size: int = 3) -> EnsembleDeployments:
    deployments = EnsembleDeployments(
        available_inference_services=AVAILABLE, max_ensemble_size=max_ensemble_size
    )
    deployments.register(consumer_id="c2_safety", service_names=names)
    return deployments


def names_of(deployments: EnsembleDeployments) -> tuple[str, ...]:
    return tuple(member.service_name for member in deployments.current("c2_safety").members)


def applying(deployments: EnsembleDeployments, action: Action) -> ActionOutcome:
    return deployments.apply(consumer_id="c2_safety", action=action)


class TestWhatAConsumerIsRegisteredWith:
    def test_it_is_served_by_the_members_it_was_registered_with(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")

        assert names_of(deployments) == ("b5-cpu", "b6-gpu")

    def test_what_registering_hands_back_is_what_it_stored(self) -> None:
        deployments = EnsembleDeployments(
            available_inference_services=AVAILABLE, max_ensemble_size=3
        )

        registered = deployments.register(consumer_id="c3_surveillance", service_names=("b5-cpu",))

        assert registered == deployments.current("c3_surveillance")

    def test_the_deployment_names_the_consumer_it_serves(self) -> None:
        deployments = serving("b5-cpu")

        assert deployments.current("c2_safety").deployment_id.startswith("c2_safety-")

    def test_the_available_services_are_offered_in_one_order(self) -> None:
        deployments = EnsembleDeployments(
            available_inference_services=AVAILABLE, max_ensemble_size=3
        )

        assert [member.service_name for member in deployments.available_inference_services] == [
            "b3-cpu",
            "b5-cpu",
            "b6-gpu",
            "b7-gpu",
        ]

    def test_it_lists_the_consumers_it_serves(self) -> None:
        deployments = serving("b5-cpu")
        deployments.register(consumer_id="c1_navigation", service_names=("b7-gpu",))

        assert deployments.consumers() == ("c1_navigation", "c2_safety")


class TestASetUpThatCannotServe:
    def test_an_ensemble_size_of_none_is_refused(self) -> None:
        with pytest.raises(DeploymentError, match="serves nobody"):
            EnsembleDeployments(available_inference_services=AVAILABLE, max_ensemble_size=0)

    def test_an_empty_pool_is_refused(self) -> None:
        with pytest.raises(DeploymentError, match="no inference service"):
            EnsembleDeployments(available_inference_services=(), max_ensemble_size=3)

    def test_a_pool_naming_one_service_twice_is_refused(self) -> None:
        with pytest.raises(DeploymentError, match="b5-cpu"):
            EnsembleDeployments(
                available_inference_services=(*AVAILABLE, AVAILABLE[0]), max_ensemble_size=3
            )

    def test_an_ensemble_of_no_members_is_refused(self) -> None:
        with pytest.raises(DeploymentError, match="no member"):
            serving()

    def test_a_member_that_is_not_available_is_refused(self) -> None:
        with pytest.raises(DeploymentError, match="unknown-gpu"):
            serving("unknown-gpu")

    def test_one_service_registered_twice_is_refused_here_and_not_by_the_deployment(
        self,
    ) -> None:
        with pytest.raises(DeploymentError, match="b5-cpu"):
            serving("b5-cpu", "b5-cpu")

    @pytest.mark.parametrize("consumer_id", ["", " ", " c2_safety", "c2_safety "])
    def test_a_consumer_id_that_no_row_could_carry_is_refused(self, consumer_id: str) -> None:
        deployments = EnsembleDeployments(
            available_inference_services=AVAILABLE, max_ensemble_size=3
        )

        with pytest.raises(DeploymentError, match="is not a consumer id"):
            deployments.register(consumer_id=consumer_id, service_names=("b5-cpu",))

    def test_an_ensemble_over_the_limit_is_refused(self) -> None:
        with pytest.raises(DeploymentError, match="at most 2"):
            serving("b5-cpu", "b6-gpu", "b7-gpu", max_ensemble_size=2)

    def test_registering_one_consumer_twice_is_refused(self) -> None:
        deployments = serving("b5-cpu")

        with pytest.raises(DeploymentError, match="c2_safety"):
            deployments.register(consumer_id="c2_safety", service_names=("b7-gpu",))

    def test_a_consumer_it_does_not_serve_has_no_ensemble(self) -> None:
        deployments = serving("b5-cpu")

        with pytest.raises(NoSuchConsumerError, match="c9_nowhere"):
            deployments.current("c9_nowhere")

    def test_a_consumer_it_does_not_serve_takes_no_action(self) -> None:
        deployments = serving("b5-cpu")

        with pytest.raises(NoSuchConsumerError, match="c9_nowhere"):
            deployments.apply(consumer_id="c9_nowhere", action=Action(kind="keep"))

    def test_one_service_name_instead_of_a_list_of_them_is_refused(self) -> None:
        deployments = EnsembleDeployments(
            available_inference_services=AVAILABLE, max_ensemble_size=3
        )

        with pytest.raises(DeploymentError, match="not a list"):
            deployments.register(consumer_id="c2_safety", service_names="b5-cpu")


class TestKeepingTheEnsemble:
    def test_no_change_is_accepted_and_moves_nothing(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        before = deployments.current("c2_safety")

        applied = applying(deployments, Action(kind="keep"))

        assert applied.accepted
        assert applied.before == applied.after == before

    def test_no_change_still_says_what_it_did(self) -> None:
        deployments = serving("b5-cpu")

        assert applying(deployments, Action(kind="keep")).reason


class TestAddingAMember:
    def test_the_member_joins_and_the_version_moves(self) -> None:
        deployments = serving("b5-cpu")
        before = deployments.current("c2_safety")

        applied = applying(deployments, Action(kind="add", member_in="b7-gpu"))

        assert applied.accepted
        assert names_of(deployments) == ("b5-cpu", "b7-gpu")
        assert applied.before == before
        assert applied.after.deployment_id != before.deployment_id

    def test_the_member_that_joined_is_named_in_the_reason(self) -> None:
        deployments = serving("b5-cpu")

        assert "b7-gpu" in applying(deployments, Action(kind="add", member_in="b7-gpu")).reason

    def test_a_member_already_serving_is_refused(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(deployments, Action(kind="add", member_in="b5-cpu"))

        assert not applied.accepted
        assert "already serves" in applied.reason

    def test_a_member_outside_the_pool_is_refused(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(deployments, Action(kind="add", member_in="unknown-gpu"))

        assert not applied.accepted
        assert "unknown-gpu" in applied.reason

    def test_an_ensemble_at_its_limit_takes_no_more(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu", max_ensemble_size=2)

        applied = applying(deployments, Action(kind="add", member_in="b7-gpu"))

        assert not applied.accepted
        assert "at most 2" in applied.reason
        assert names_of(deployments) == ("b5-cpu", "b6-gpu")


class TestRemovingAMember:
    def test_the_member_leaves_and_the_version_moves(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        before = deployments.current("c2_safety")

        applied = applying(deployments, Action(kind="remove", member_out="b5-cpu"))

        assert applied.accepted
        assert names_of(deployments) == ("b6-gpu",)
        assert applied.before == before
        assert applied.after.deployment_id != before.deployment_id

    def test_a_member_that_does_not_serve_cannot_leave(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")

        applied = applying(deployments, Action(kind="remove", member_out="b7-gpu"))

        assert not applied.accepted
        assert "does not serve" in applied.reason

    def test_the_only_member_serving_cannot_leave(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(deployments, Action(kind="remove", member_out="b5-cpu"))

        assert not applied.accepted
        assert "the only member" in applied.reason
        assert names_of(deployments) == ("b5-cpu",)


class TestReplacingAMember:
    def test_the_swap_keeps_the_ensemble_the_same_size(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        before = deployments.current("c2_safety")

        applied = applying(
            deployments, Action(kind="replace", member_in="b7-gpu", member_out="b5-cpu")
        )

        assert applied.accepted
        assert names_of(deployments) == ("b6-gpu", "b7-gpu")
        assert applied.before == before
        assert applied.after.deployment_id != before.deployment_id

    def test_a_member_that_does_not_serve_cannot_be_replaced(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(
            deployments, Action(kind="replace", member_in="b7-gpu", member_out="b6-gpu")
        )

        assert not applied.accepted
        assert "does not serve" in applied.reason

    def test_a_replacement_outside_the_pool_is_refused(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(
            deployments, Action(kind="replace", member_in="unknown-gpu", member_out="b5-cpu")
        )

        assert not applied.accepted
        assert "unknown-gpu" in applied.reason

    def test_a_replacement_already_serving_is_refused(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")

        applied = applying(
            deployments, Action(kind="replace", member_in="b6-gpu", member_out="b5-cpu")
        )

        assert not applied.accepted
        assert "already serves" in applied.reason

    def test_a_replacement_at_the_size_limit_is_still_a_swap(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu", max_ensemble_size=2)

        applied = applying(
            deployments, Action(kind="replace", member_in="b7-gpu", member_out="b5-cpu")
        )

        assert applied.accepted
        assert names_of(deployments) == ("b6-gpu", "b7-gpu")


REFUSED: Sequence[Action] = (
    Action(kind="add", member_in="b5-cpu"),
    Action(kind="add", member_in="unknown-gpu"),
    Action(kind="remove", member_out="b7-gpu"),
    Action(kind="replace", member_in="unknown-gpu", member_out="b5-cpu"),
    Action(kind="replace", member_in="b6-gpu", member_out="b7-gpu"),
)


class TestAnEnsembleOfOne:
    def test_nothing_may_join_it_and_nothing_may_leave_it(self) -> None:
        deployments = serving("b5-cpu", max_ensemble_size=1)

        joining = applying(deployments, Action(kind="add", member_in="b7-gpu"))
        leaving = applying(deployments, Action(kind="remove", member_out="b5-cpu"))

        assert not joining.accepted
        assert not leaving.accepted
        assert names_of(deployments) == ("b5-cpu",)

    def test_its_one_member_may_still_be_swapped(self) -> None:
        deployments = serving("b5-cpu", max_ensemble_size=1)

        applied = applying(
            deployments, Action(kind="replace", member_in="b7-gpu", member_out="b5-cpu")
        )

        assert applied.accepted
        assert names_of(deployments) == ("b7-gpu",)


class TestTheVersion:
    @pytest.mark.parametrize("action", REFUSED)
    def test_a_refused_action_leaves_the_ensemble_where_it_was(self, action: Action) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        before = deployments.current("c2_safety")

        applied = applying(deployments, action)

        assert not applied.accepted
        assert applied.before == applied.after == before
        assert deployments.current("c2_safety") == before

    @pytest.mark.parametrize("action", REFUSED)
    def test_a_refused_action_says_why(self, action: Action) -> None:
        deployments = serving("b5-cpu", "b6-gpu")

        assert applying(deployments, action).reason

    def test_an_ensemble_that_comes_back_serves_a_second_stretch_of_the_run(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        first = deployments.current("c2_safety")

        applying(deployments, Action(kind="replace", member_in="b7-gpu", member_out="b6-gpu"))
        applying(deployments, Action(kind="replace", member_in="b6-gpu", member_out="b7-gpu"))
        again = deployments.current("c2_safety")

        assert again.deployment_id != first.deployment_id
        assert set(again.members) == set(first.members)

    def test_every_stretch_of_one_run_is_named_once(self) -> None:
        deployments = serving("b5-cpu", max_ensemble_size=2)
        swap = Action(kind="replace", member_in="b7-gpu", member_out="b5-cpu")
        back = Action(kind="replace", member_in="b5-cpu", member_out="b7-gpu")
        served = [deployments.current("c2_safety").deployment_id]

        for turn in range(6):
            served.append(applying(deployments, back if turn % 2 else swap).after.deployment_id)

        assert len(set(served)) == len(served)

    def test_the_id_a_version_carries_is_the_same_in_every_run(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")

        assert deployments.current("c2_safety").deployment_id == "c2_safety-0001"
        assert (
            applying(deployments, Action(kind="add", member_in="b7-gpu")).after.deployment_id
            == "c2_safety-0002"
        )

    def test_a_keep_serves_the_stretch_it_is_already_in(self) -> None:
        deployments = serving("b5-cpu")
        first = deployments.current("c2_safety").deployment_id

        applying(deployments, Action(kind="keep"))

        assert deployments.current("c2_safety").deployment_id == first

    def test_one_member_set_is_one_ensemble_however_it_was_listed(self) -> None:
        listed = serving("b5-cpu", "b6-gpu")
        reversed_listing = serving("b6-gpu", "b5-cpu")

        assert listed.current("c2_safety") == reversed_listing.current("c2_safety")

    def test_what_one_consumer_does_leaves_the_other_consumers_alone(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        deployments.register(consumer_id="c1_navigation", service_names=("b7-gpu",))
        untouched = deployments.current("c1_navigation")

        applying(deployments, Action(kind="add", member_in="b3-cpu"))

        assert deployments.current("c1_navigation") == untouched

    def test_the_ensemble_a_caller_read_is_a_value_and_not_a_view(self) -> None:
        deployments = serving("b5-cpu")
        held = deployments.current("c2_safety")

        applying(deployments, Action(kind="add", member_in="b7-gpu"))

        assert held.members == (EnsembleMember(service_name="b5-cpu", model="b5", device="cpu"),)
        assert held.deployment_id != deployments.current("c2_safety").deployment_id

    def test_two_consumers_holding_the_same_members_hold_two_versions(self) -> None:
        deployments = serving("b5-cpu")
        deployments.register(consumer_id="c1_navigation", service_names=("b5-cpu",))

        c2 = deployments.current("c2_safety").deployment_id
        c1 = deployments.current("c1_navigation").deployment_id

        assert c1 != c2


class TestAnActionOfAKindItDoesNotKnow:
    def test_it_is_never_recorded_as_a_no_change(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        before = deployments.current("c2_safety")

        unknown = Action.model_construct(kind="split")  # type: ignore[arg-type]

        with pytest.raises(AssertionError):
            applying(deployments, unknown)

        assert deployments.current("c2_safety") == before


class TestSeveralCallersAtOnce:
    def test_every_accepted_action_is_in_the_ensemble_that_serves(self) -> None:
        joining = [Action(kind="add", member_in=name) for name in ("b6-gpu", "b3-cpu", "b7-gpu")]

        for _ in range(40):
            deployments = serving("b5-cpu", max_ensemble_size=4)
            with ThreadPoolExecutor(max_workers=len(joining)) as callers:
                turns = [callers.submit(applying, deployments, action) for action in joining]

            accepted = [turn.result() for turn in turns if turn.result().accepted]
            serving_now = set(deployments.current("c2_safety").members)
            assert all(set(one.after.members) <= serving_now for one in accepted)
            assert len(serving_now) == 1 + len(accepted)


class TestSelectingAWholeEnsemble:
    def test_the_members_it_names_are_the_ones_serving(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(deployments, Action(kind="select", members=("b6-gpu", "b7-gpu")))

        assert applied.accepted
        assert names_of(deployments) == ("b6-gpu", "b7-gpu")

    def test_it_can_change_every_member_at_once(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")

        applying(deployments, Action(kind="select", members=("b3-cpu", "b7-gpu")))

        assert names_of(deployments) == ("b3-cpu", "b7-gpu")

    def test_the_reason_says_which_ensemble_was_selected(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(deployments, Action(kind="select", members=("b7-gpu",)))

        assert "b7-gpu" in applied.reason

    def test_an_ensemble_above_the_size_limit_is_refused(self) -> None:
        deployments = serving("b5-cpu", max_ensemble_size=2)

        applied = applying(
            deployments,
            Action(kind="select", members=("b5-cpu", "b6-gpu", "b7-gpu")),
        )

        assert not applied.accepted
        assert "at most 2" in applied.reason
        assert names_of(deployments) == ("b5-cpu",)

    def test_a_member_the_pool_does_not_hold_is_refused(self) -> None:
        deployments = serving("b5-cpu")

        applied = applying(deployments, Action(kind="select", members=("infer-nobody",)))

        assert not applied.accepted
        assert "infer-nobody" in applied.reason

    def test_selecting_the_ensemble_already_serving_changes_no_version(self) -> None:
        deployments = serving("b5-cpu", "b6-gpu")
        before = deployments.current("c2_safety").deployment_id

        applied = applying(deployments, Action(kind="select", members=("b6-gpu", "b5-cpu")))

        assert applied.accepted
        assert deployments.current("c2_safety").deployment_id == before

    def test_the_order_it_names_them_in_does_not_decide_the_deployment(self) -> None:
        one = serving("b5-cpu")
        other = serving("b5-cpu")

        applying(one, Action(kind="select", members=("b6-gpu", "b7-gpu")))
        applying(other, Action(kind="select", members=("b7-gpu", "b6-gpu")))

        assert names_of(one) == names_of(other)
