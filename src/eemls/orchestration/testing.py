from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any

import httpx

from eemls.domain.types import EnsembleMember

AVAILABLE = (
    EnsembleMember(service_name="b5-cpu", model="b5", device="cpu"),
    EnsembleMember(service_name="b6-gpu", model="b6", device="gpu"),
    EnsembleMember(service_name="b3-cpu", model="b3_v3", device="cpu"),
    EnsembleMember(service_name="b7-gpu", model="b7_b_16", device="gpu"),
)

PORT = 8082


def member(name: str, model: str = "b5", device: str = "raspberry_pi5") -> EnsembleMember:
    return EnsembleMember(service_name=name, model=model, device=device)


class Answers:
    def __init__(
        self,
        *,
        label: str = "car",
        confidence: float = 0.9,
        status_code: int = 200,
        timeout: bool = False,
        unreachable: bool = False,
        unreadable: bool = False,
        fetch_ms: float = 2.0,
        inference_ms: float = 18.0,
        total_ms: float = 21.0,
        instance_id: str | None = None,
        node: str = "pi-1",
        replica: int = 1,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.status_code = status_code
        self.timeout = timeout
        self.unreachable = unreachable
        self.unreadable = unreadable
        self.fetch_ms = fetch_ms
        self.inference_ms = inference_ms
        self.total_ms = total_ms
        self.instance_id = instance_id
        self.node = node
        self.replica = replica


class Members(httpx.AsyncBaseTransport):
    """Given `together`, no service answers until that many calls have arrived."""

    def __init__(self, answering: Mapping[str, Answers], *, together: int | None = None) -> None:
        self._answering = dict(answering)
        self._together = together
        self._arrived = 0
        self._all_here = asyncio.Event()
        self.asked: list[str] = []
        self.sent: list[object] = []
        self._held: dict[str, asyncio.Event] = {}

    def hold(self, name: str) -> None:
        self._held[name] = asyncio.Event()

    def release(self, name: str) -> None:
        self._held[name].set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        name = request.url.host
        self.asked.append(name)
        asked = json.loads(request.content)
        self.sent.append(asked)
        await self._with_the_others()
        if name in self._held:
            await self._held[name].wait()
        answers = self._answering.get(name)
        if answers is None:
            return httpx.Response(404, json={"detail": f"no service {name}"})
        if answers.timeout:
            raise httpx.ReadTimeout("the member did not answer in time", request=request)
        if answers.unreachable:
            raise httpx.ConnectError(f"no copy of {name} is there", request=request)
        if answers.unreadable:
            return httpx.Response(200, content=b"not an answer")
        if answers.status_code != 200:
            return httpx.Response(answers.status_code, json={"detail": f"{name} says no"})
        return httpx.Response(200, json=self._body(name, answers, asked))

    async def _with_the_others(self) -> None:
        if self._together is None:
            return
        self._arrived += 1
        if self._arrived >= self._together:
            self._all_here.set()
        await self._all_here.wait()

    def _body(self, name: str, answers: Answers, asked: Mapping[str, object]) -> dict[str, object]:
        return {
            "query_id": asked.get("query_id"),
            "instance_id": answers.instance_id or f"{name}-1",
            "service_name": name,
            "model": "b5",
            "device": "raspberry_pi5",
            "node": answers.node,
            "replica": answers.replica,
            "predicted_label": answers.label,
            "confidence": answers.confidence,
            "fetch_ms": answers.fetch_ms,
            "inference_ms": answers.inference_ms,
            "total_ms": answers.total_ms,
            "execution_provider": "CPUExecutionProvider",
            "number_format": "float32",
            "cache_hit": True,
            "status": "ok",
        }


Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

HOST = "127.0.0.1"


CHOSEN = random.Random(5)  # noqa: S311


async def sometimes_slow(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] != "http":
        return
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    asked = json.loads(body or b"{}")
    await asyncio.sleep(CHOSEN.choice((0.01, 0.01, 0.5, 2.0)))
    answer = {
        "query_id": asked.get("query_id"),
        "instance_id": f"{HOST}-1",
        "service_name": HOST,
        "node": "pi-1",
        "replica": 1,
        "predicted_label": "car",
        "confidence": 0.9,
        "fetch_ms": 1.0,
        "inference_ms": 5.0,
        "total_ms": 7.0,
        "execution_provider": "CPUExecutionProvider",
        "number_format": "float32",
        "cache_hit": True,
    }
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": json.dumps(answer).encode()})
