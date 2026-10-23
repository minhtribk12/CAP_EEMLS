from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import assert_never

from eemls.domain.errors import EemlsError
from eemls.domain.types import Action, EnsembleDeployment, EnsembleMember


class DeploymentError(EemlsError): ...


class NoSuchConsumerError(DeploymentError): ...


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    accepted: bool
    reason: str
    before: EnsembleDeployment
    after: EnsembleDeployment


def _deployment_of(
    consumer_id: str, version: int, members: Sequence[EnsembleMember]
) -> EnsembleDeployment:
    return EnsembleDeployment(deployment_id=f"{consumer_id}-{version:04d}", members=tuple(members))


def _same(members: Sequence[EnsembleMember], others: Sequence[EnsembleMember]) -> bool:
    return {one.service_name for one in members} == {one.service_name for one in others}


def _did(action: Action) -> str:
    if action.kind == "add":
        return f"{action.member_in} joined the ensemble"
    if action.kind == "remove":
        return f"{action.member_out} left the ensemble"
    if action.kind == "replace":
        return f"{action.member_in} took the place of {action.member_out}"
    if action.kind == "keep":
        return "the ensemble was kept as it was"
    if action.kind == "select":
        return f"{list(action.members)} were selected to serve"
    assert_never(action.kind)


class EnsembleDeployments:
    def __init__(
        self, *, available_inference_services: Sequence[EnsembleMember], max_ensemble_size: int
    ) -> None:
        if max_ensemble_size < 1:
            raise DeploymentError(
                f"an ensemble of at most {max_ensemble_size} members serves nobody"
            )
        if not available_inference_services:
            raise DeploymentError("no inference service is available to build an ensemble from")
        named = [member.service_name for member in available_inference_services]
        repeated = {name for name in named if named.count(name) > 1}
        if repeated:
            raise DeploymentError(
                f"the available inference services name {sorted(repeated)} more than once"
            )
        self._available: Mapping[str, EnsembleMember] = MappingProxyType(
            {member.service_name: member for member in available_inference_services}
        )
        self._max_ensemble_size = max_ensemble_size
        self._served: dict[str, EnsembleDeployment] = {}
        self._versions: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def available_inference_services(self) -> tuple[EnsembleMember, ...]:
        return tuple(self._available[name] for name in sorted(self._available))

    def consumers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._served))

    def register(self, *, consumer_id: str, service_names: Sequence[str]) -> EnsembleDeployment:
        if not consumer_id or consumer_id != consumer_id.strip():
            raise DeploymentError(f"{consumer_id!r} is not a consumer id")
        members = self._members_of(consumer_id, service_names)
        with self._lock:
            held = self._served.get(consumer_id)
            if held is not None:
                raise DeploymentError(f"{consumer_id} is already served by {held.deployment_id}")
            return self._serve(consumer_id, 1, members)

    def current(self, consumer_id: str) -> EnsembleDeployment:
        with self._lock:
            return self._require(consumer_id)

    def apply(self, *, consumer_id: str, action: Action) -> ActionOutcome:
        with self._lock:
            before = self._require(consumer_id)
            serving = {member.service_name: member for member in before.members}
            refusal = self._refusal(action, serving)
            if refusal is not None:
                return ActionOutcome(accepted=False, reason=refusal, before=before, after=before)
            did = _did(action)
            after = self._after(action, serving)
            if action.kind == "keep" or _same(after, before.members):
                return ActionOutcome(accepted=True, reason=did, before=before, after=before)
            version = self._versions[consumer_id] + 1
            served = self._serve(consumer_id, version, after)
            return ActionOutcome(accepted=True, reason=did, before=before, after=served)

    def _serve(
        self, consumer_id: str, version: int, members: Sequence[EnsembleMember]
    ) -> EnsembleDeployment:
        """Put one ensemble version in service. Lock held."""
        deployment = _deployment_of(consumer_id, version, members)
        self._served[consumer_id] = deployment
        self._versions[consumer_id] = version
        return deployment

    def _require(self, consumer_id: str) -> EnsembleDeployment:
        """The ensemble of a served consumer. Lock held."""
        held = self._served.get(consumer_id)
        if held is None:
            raise NoSuchConsumerError(f"{consumer_id} is not served by this control plane")
        return held

    def _members_of(self, consumer_id: str, service_names: Sequence[str]) -> list[EnsembleMember]:
        if isinstance(service_names, str):
            raise DeploymentError(
                f"{consumer_id} is registered with the name {service_names!r}, not a list of them"
            )
        if not service_names:
            raise DeploymentError(f"{consumer_id} is registered with no member to serve it")
        refusal = self._selected(service_names)
        if refusal is not None:
            raise DeploymentError(f"{consumer_id} is registered, and {refusal}")
        return [self._available[name] for name in service_names]

    def _refusal(self, action: Action, serving: Mapping[str, EnsembleMember]) -> str | None:
        """Why this action is refused, or None. Lock held."""
        if action.kind == "select":
            return self._selected(action.members)
        going, coming = action.member_out, action.member_in
        if going is not None and going not in serving:
            return f"{going} does not serve this consumer"
        if coming is not None and coming not in self._available:
            return f"{coming} is not an available inference service"
        if coming is not None and coming in serving:
            return f"{coming} already serves this consumer"
        if action.kind == "remove" and len(serving) == 1:
            return f"{going} is the only member serving, and an ensemble holds one"
        if action.kind == "add" and len(serving) >= self._max_ensemble_size:
            return (
                f"{len(serving)} members already serve, and at most {self._max_ensemble_size} may"
            )
        return None

    def _selected(self, members: Sequence[str]) -> str | None:
        """Why this selection is refused, or None."""
        unknown = [name for name in members if name not in self._available]
        if unknown:
            return f"{unknown} is not an available inference service"
        repeated = sorted({name for name in members if list(members).count(name) > 1})
        if repeated:
            return f"{repeated} are named more than once"
        if len(members) > self._max_ensemble_size:
            return (
                f"{len(members)} members were selected, and at most "
                f"{self._max_ensemble_size} may serve"
            )
        return None

    def _after(self, action: Action, serving: Mapping[str, EnsembleMember]) -> list[EnsembleMember]:
        """The members after this action. Lock held."""
        if action.kind == "select":
            return [self._available[name] for name in action.members]
        held = [member for name, member in serving.items() if name != action.member_out]
        if action.member_in is not None:
            held.append(self._available[action.member_in])
        return held
