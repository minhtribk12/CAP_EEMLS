from __future__ import annotations

import itertools
import json
import random
from collections.abc import Callable
from pathlib import Path

import httpx

from eemls.image_catalogue import Catalogue
from eemls.monitoring.metrics import Monitoring
from eemls_consumer.config import load_consumer_config
from eemls_consumer.feedback import LabelsToReturn
from eemls_consumer.query_schedule import ImageQualityEvent, Period, Schedule
from eemls_consumer.query_sender import QuerySender
from eemls_consumer.results import Pending

ROOT = Path(__file__).resolve().parents[4]
CONSUMER = "c2_safety"
TOPIC = "eemls/results/c2_safety/client-1"

PERIODS = (
    Period(
        name="morning",
        starts_at_s=0.0,
        class_mix={"pedestrian": 1.0},
        condition="day",
        weather="clear",
    ),
    Period(
        name="night", starts_at_s=10.0, class_mix={"car": 1.0}, condition="night", weather="clear"
    ),
)
DAY_LENGTH_S = 20.0
EVENT = ImageQualityEvent(starts_at_s=5.0, ends_at_s=7.0, image_quality="blurred")
# The morning period asks about pedestrians by day, the night period about cars at night.
IDS = {
    ("pedestrian", "day", "clear"): ("ped-clear-0",),
    ("pedestrian", "day", "blurred"): ("ped-blurred-0",),
    ("car", "night", "clear"): ("car-clear-0",),
    ("car", "night", "blurred"): ("car-blurred-0",),
}


class Moment:
    def __init__(self, at: float = 0.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def advancing(
    moment: Moment, by: float, waits: list[float] | None = None
) -> Callable[[float], None]:
    def sleeper(interval_s: float) -> None:
        if waits is not None:
            waits.append(interval_s)
        moment.at += by

    return sleeper


def label_count(returned: dict[str, object]) -> int:
    labels = returned["labels"]
    assert isinstance(labels, list)
    return len(labels)


class Stub:
    def __init__(
        self,
        *,
        status: int = 202,
        reachable: bool = True,
        body: dict[str, object] | None = None,
        feedback_status: int = 202,
        takes_feedback: bool = True,
        while_accepting: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.status = status
        self.reachable = reachable
        self.body = body
        self.feedback_status = feedback_status
        self.takes_feedback = takes_feedback
        self.while_accepting = while_accepting
        self.queries: list[dict[str, object]] = []
        self.returned: list[dict[str, object]] = []
        self.returned_after: list[int] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.answer)

    def answer(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("nothing is listening for the control plane")
        body = json.loads(request.content)
        if request.url.path == "/feedback":
            if not self.takes_feedback:
                raise httpx.ConnectError("the control plane dropped the connection")
            if self.feedback_status // 100 != 2:
                return httpx.Response(self.feedback_status, json={"detail": "not these labels"})
            self.returned.append(body)
            self.returned_after.append(len(self.queries))
            return httpx.Response(self.feedback_status, json={"kept": len(body["labels"])})
        self.queries.append(body)
        if self.status != 202:
            return httpx.Response(self.status, json={"detail": "over the cap"})
        if self.while_accepting is not None:
            self.while_accepting(body)
        if self.body is not None:
            return httpx.Response(self.status, json=self.body)
        return httpx.Response(202, json={"query_id": f"q-{len(self.queries)}"})


def built(
    stub: Stub,
    *,
    send_window: int = 64,
    fraction: float = 0.0,
    label_delay_queries: int = 200,
    decision_interval_queries: int = 5,
    rate: float = 20.0,
    draw: float | Callable[[], float] = 0.0,
    moment: Moment | None = None,
) -> tuple[QuerySender, Monitoring, Moment]:
    monitoring = Monitoring.from_config(load_consumer_config(ROOT / "config").services.monitoring)
    clock = moment if moment is not None else Moment()
    names = itertools.count(1)
    client = QuerySender(
        consumer_id=CONSUMER,
        reply_topic=TOPIC,
        schedule=Schedule(
            request_rate_per_s=rate, periods=PERIODS, day_length_s=DAY_LENGTH_S, events=(EVENT,)
        ),
        catalogue=Catalogue(IDS),
        pending=Pending(timeout_ms=2000.0),
        labels_to_return=LabelsToReturn(
            fraction=fraction,
            label_delay_queries=label_delay_queries,
            draw=random.Random(11).random,  # noqa: S311
        ),
        control_plane=httpx.Client(
            base_url="http://control-plane:8080", transport=stub.transport()
        ),
        send_window=send_window,
        feedback_batch=32,
        decision_interval_queries=decision_interval_queries,
        draw=draw if callable(draw) else lambda: draw,
        identifiers=lambda: f"corr-{next(names)}",
        monitoring=monitoring,
        clock=clock,
    )
    return client, monitoring, clock
