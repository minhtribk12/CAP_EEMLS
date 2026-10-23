from __future__ import annotations

from dataclasses import dataclass

from eemls.domain.errors import EemlsError
from eemls.domain.types import RuntimeContext


class TableError(EemlsError): ...


@dataclass(slots=True)
class _Average:
    mean: float = 0.0
    taken: int = 0

    def take(self, contribution: float) -> None:
        self.taken += 1
        self.mean += (contribution - self.mean) / self.taken


class IdentityTable:
    def __init__(self, context_key: str) -> None:
        if context_key not in RuntimeContext.model_fields:
            raise TableError(f"the runtime context carries no {context_key} to key a table on")
        self.context_key = context_key
        self._average: dict[tuple[str, str], _Average] = {}

    def key_of(self, context: RuntimeContext) -> str:
        return str(getattr(context, self.context_key))

    def contribution(self, context: RuntimeContext, service_name: str) -> float:
        held = self._average.get((self.key_of(context), service_name))
        return 0.0 if held is None else held.mean

    def learn(self, context: RuntimeContext, service_name: str, contribution: float) -> None:
        self._average.setdefault((self.key_of(context), service_name), _Average()).take(
            contribution
        )


class IdentityTables:
    def __init__(self, context_keys: tuple[str, ...]) -> None:
        self.by_key = {key: IdentityTable(key) for key in context_keys}

    def looked_up(self, context: RuntimeContext, service_name: str) -> tuple[float, ...]:
        return tuple(one.contribution(context, service_name) for one in self.by_key.values())

    def learn(self, context: RuntimeContext, service_name: str, contribution: float) -> None:
        for one in self.by_key.values():
            one.learn(context, service_name, contribution)
