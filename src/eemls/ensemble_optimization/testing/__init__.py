from __future__ import annotations

from collections.abc import Callable, Sequence

from eemls.domain.types import Action
from eemls.ensemble_optimization.approach import Observation, Policy


def decisions(
    build: Callable[[], Policy], observations: Sequence[Observation]
) -> tuple[Action, ...]:
    policy = build()
    return tuple(policy.decide(one).action for one in observations)


def members_after(action: Action, observed: Observation) -> tuple[str, ...]:
    if action.kind == "select":
        return action.members
    held = tuple(
        one.service_name
        for one in observed.deployment.members
        if one.service_name != action.member_out
    )
    return held if action.member_in is None else (*held, action.member_in)


def check_valid_actions(build: Callable[[], Policy], observations: Sequence[Observation]) -> None:
    for observed, action in zip(observations, decisions(build, observations), strict=True):
        available = {one.service_name for one in observed.available}
        serving = set(observed.deployment.service_names)
        named = {*action.members, *([action.member_in] if action.member_in is not None else [])}
        unknown = sorted(named - available)
        if unknown:
            raise AssertionError(f"{action.kind} names {unknown}, which is not available")
        if action.member_out is not None and action.member_out not in serving:
            raise AssertionError(
                f"{action.kind} takes out {action.member_out}, which does not serve"
            )
        if action.member_in is not None and action.member_in in serving:
            raise AssertionError(
                f"{action.kind} brings in {action.member_in}, which already serves"
            )
        if not members_after(action, observed):
            raise AssertionError(f"{action.kind} leaves no member serving")


def check_ensemble_size(
    build: Callable[[], Policy], observations: Sequence[Observation], *, max_ensemble_size: int
) -> None:
    for observed, action in zip(observations, decisions(build, observations), strict=True):
        size = len(members_after(action, observed))
        if size > max_ensemble_size:
            raise AssertionError(
                f"{action.kind} serves {size} members, above the limit of {max_ensemble_size}"
            )


def check_same_seed_same_decisions(
    build: Callable[[], Policy], observations: Sequence[Observation]
) -> None:
    if decisions(build, observations) != decisions(build, observations):
        raise AssertionError("two builds on the same seed decide differently")


def check_conformance(
    build: Callable[[], Policy], observations: Sequence[Observation], *, max_ensemble_size: int
) -> None:
    check_valid_actions(build, observations)
    check_ensemble_size(build, observations, max_ensemble_size=max_ensemble_size)
    check_same_seed_same_decisions(build, observations)
