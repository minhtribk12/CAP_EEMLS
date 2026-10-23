from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import httpx

from eemls.domain.statistics import percentile
from eemls.domain.waiting import all_finished

Post = Callable[[int], Awaitable[int]]


@dataclass(frozen=True)
class Step:
    rate_per_s: float
    duration_s: float


@dataclass(frozen=True)
class StepOutcome:
    rate_per_s: float
    since: datetime
    until: datetime
    accepted: int
    refused: int
    failed: int
    not_sent: int


@dataclass(frozen=True)
class Report:
    rate_per_s: float
    accepted: int
    refused: int
    failed: int
    not_sent: int
    answered: int
    unanswered: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    fetch_ms: float | None
    inference_ms: float | None
    wait_ms: float | None
    aggregate_ms: float | None


def steps_of(rates: Sequence[float], *, step_s: float) -> tuple[Step, ...]:
    if not rates or step_s <= 0.0 or any(rate <= 0.0 for rate in rates):
        raise ValueError(
            f"a step needs a positive rate and length, not {list(rates)} for {step_s} s"
        )
    return tuple(Step(rate_per_s=rate, duration_s=step_s) for rate in rates)


def send_times(step: Step) -> list[float]:
    count = math.floor(step.rate_per_s * step.duration_s + 1e-9)
    return [index / step.rate_per_s for index in range(count)]


async def run_load_test(
    steps: Sequence[Step],
    *,
    post: Post,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    wall: Callable[[], datetime],
    max_in_flight: int,
) -> list[StepOutcome]:
    counts = [Counter[str]() for _ in steps]
    in_flight = 0
    tasks: set[asyncio.Task[None]] = set()
    stretches: list[tuple[datetime, datetime]] = []
    sent = 0

    async def one(index: int, counted: Counter[str]) -> None:
        nonlocal in_flight
        try:
            status = await post(index)
        except httpx.HTTPError:
            counted["failed"] += 1
        else:
            counted[_kind(status)] += 1
        finally:
            in_flight -= 1

    started = clock()
    for step, counted in zip(steps, counts, strict=True):
        since = wall()
        for at in send_times(step):
            await sleep(started + at - clock())
            if in_flight >= max_in_flight:
                counted["not_sent"] += 1
                continue
            in_flight += 1
            sent += 1
            task = asyncio.create_task(one(sent, counted))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        started += step.duration_s
        await sleep(started - clock())
        stretches.append((since, wall()))
    await all_finished(tasks)
    return [
        StepOutcome(
            rate_per_s=step.rate_per_s,
            since=since,
            until=until,
            accepted=counted["accepted"],
            refused=counted["refused"],
            failed=counted["failed"],
            not_sent=counted["not_sent"],
        )
        for step, counted, (since, until) in zip(steps, counts, stretches, strict=True)
    ]


def _kind(status: int) -> str:
    if status == httpx.codes.ACCEPTED:
        return "accepted"
    if status == httpx.codes.SERVICE_UNAVAILABLE:
        return "refused"
    return "failed"


def reported(
    sent: StepOutcome,
    served: Sequence[Mapping[str, object]],
    members: Sequence[Mapping[str, object]],
) -> Report:
    answered = [row for row in served if row["outcome"] != "failed"]
    times = [_number(row["response_time_ms"]) for row in answered]
    ids = {row["query_id"] for row in answered}
    used = [
        row
        for row in members
        if row["query_id"] in ids and row["status"] == "ok" and row["in_ensemble"]
    ]
    return Report(
        rate_per_s=sent.rate_per_s,
        accepted=sent.accepted,
        refused=sent.refused,
        failed=sent.failed,
        not_sent=sent.not_sent,
        answered=len(answered),
        unanswered=len(served) - len(answered),
        p50_ms=percentile(times, 50) if times else None,
        p95_ms=percentile(times, 95) if times else None,
        p99_ms=percentile(times, 99) if times else None,
        fetch_ms=_mean(row["fetch_ms"] for row in used),
        inference_ms=_mean(row["inference_ms"] for row in used),
        wait_ms=_mean(row["wait_ms"] for row in answered),
        aggregate_ms=_mean(row["aggregate_ms"] for row in answered),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{value!r} is not a number")
    return float(value)


def _mean(values: Iterable[object]) -> float | None:
    held = [_number(value) for value in values if value is not None]
    return sum(held) / len(held) if held else None


COLUMNS = (
    ("rate/s", "rate_per_s"),
    ("accepted", "accepted"),
    ("refused", "refused"),
    ("not sent", "not_sent"),
    ("unanswered", "unanswered"),
    ("p50 ms", "p50_ms"),
    ("p95 ms", "p95_ms"),
    ("p99 ms", "p99_ms"),
    ("fetch ms", "fetch_ms"),
    ("model ms", "inference_ms"),
    ("wait ms", "wait_ms"),
    ("vote ms", "aggregate_ms"),
)


def table(reports: Sequence[Report]) -> str:
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = [
        "| " + " | ".join(_cell(getattr(report, field)) for _, field in COLUMNS) + " |"
        for report in reports
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
