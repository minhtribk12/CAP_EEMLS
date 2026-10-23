from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from eemls.domain.measuring import milliseconds_since
from eemls.domain.random_seeds import consumer_seed
from eemls.domain.types import Action, PositiveCount, PositiveNumber, Ratio, RuntimeContext
from eemls.ensemble_optimization.approach import (
    API_VERSION,
    Approach,
    ApproachSettings,
    Observation,
    PolicyDecision,
    rows_of_the_same_queries,
)
from eemls.ensemble_optimization.approaches.cap.action_evaluation import contributions
from eemls.ensemble_optimization.approaches.cap.action_selection import action_toward
from eemls.ensemble_optimization.approaches.cap.context_signals import (
    SIGNAL_NAMES,
    consumer_signals,
    service_signals,
    strong_classes,
    vector,
)
from eemls.ensemble_optimization.approaches.cap.identity_tables import IdentityTables
from eemls.ensemble_optimization.approaches.cap.learned_weights import Weights, standard_normal
from eemls.ensemble_optimization.approaches.cap.night_detection import NightDetector
from eemls.ensemble_optimization.approaches.cap.switching_margin import SwitchingMargin
from eemls.evaluation_engine.window import Window

CONTEXT_KEYS = ("period_name", "condition")


@dataclass(frozen=True, slots=True)
class PendingAction:
    signals: Mapping[str, tuple[float, ...]]
    context: RuntimeContext


class FeedbackDelayQueue:
    def __init__(self, *, delay_intervals: int) -> None:
        self._delay = delay_intervals
        self._held: deque[PendingAction] = deque()

    def push(self, one: PendingAction) -> PendingAction | None:
        self._held.append(one)
        if len(self._held) <= self._delay + 1:
            return None
        return self._held.popleft()


@dataclass(slots=True)
class LearnedValues:
    weights: Weights
    margin: SwitchingMargin
    tables: IdentityTables
    queue: FeedbackDelayQueue
    condition: NightDetector
    noise_variance: float
    share: tuple[float, ...] | None = None
    strong: dict[str, float] = field(default_factory=dict)

    def learn(
        self,
        signals: Sequence[float],
        context: RuntimeContext,
        service_name: str,
        contribution: float,
    ) -> None:
        measured = self.margin.measured()
        noise = measured**2 if measured is not None and measured > 0.0 else self.noise_variance
        self.weights.learn(signals, contribution, noise_variance=noise)
        self.tables.learn(context, service_name, contribution)
        self.margin.learn(contribution)


class Cap:
    name = "cap"

    def __init__(
        self,
        *,
        classes: Sequence[str],
        max_ensemble_size: int,
        evaluation_window: int,
        initial_switching_margin: float,
        contributions_before_margin: int,
        prior_precision: float,
        noise_variance: float,
        night_confidence_drop: float,
        seed: int,
    ) -> None:
        self._classes = tuple(classes)
        self._limit = max_ensemble_size
        self._window = evaluation_window
        self._initial_margin = initial_switching_margin
        self._before_margin = contributions_before_margin
        self._prior_precision = prior_precision
        self._noise_variance = noise_variance
        self._dimension = len(SIGNAL_NAMES(self._classes))
        self._night_drop = night_confidence_drop
        self._seed = seed
        self._held: dict[str, LearnedValues] = {}

    def decide(self, observed: Observation) -> PolicyDecision:
        selecting = time.perf_counter()
        held = self._for(observed)
        window = observed.window.last(self._window)
        members = rows_of_the_same_queries(observed.member_window, window)
        serving = observed.deployment.service_names
        context = observed.context.model_copy(update={"condition": held.condition.read(window)})
        signals = self._signals(held, observed, context, window, members, serving)
        margin = held.margin.value()
        action = self._acted(held, signals, serving, margin)
        select_ms = milliseconds_since(selecting)
        contributing = time.perf_counter()
        due = held.queue.push(
            PendingAction(signals=MappingProxyType(dict(signals)), context=context)
        )
        if due is not None:
            self._learn(held, due, observed, members, serving)
        return PolicyDecision(
            action=action,
            candidates=len(signals),
            contribution_ms=milliseconds_since(contributing),
            select_ms=select_ms,
        )

    def _signals(
        self,
        held: LearnedValues,
        observed: Observation,
        context: RuntimeContext,
        window: Window,
        members: Window,
        serving: Sequence[str],
    ) -> Mapping[str, tuple[float, ...]]:
        shared = consumer_signals(
            window=window,
            classes=self._classes,
            previous=held.share,
            n_members=len(serving),
        )
        signals: dict[str, tuple[float, ...]] = {}
        for one in observed.available:
            name = one.service_name
            profile = observed.profiles.get(name)
            period, condition = held.tables.looked_up(context, name)
            mine = service_signals(
                strong=frozenset() if profile is None else strong_classes(profile),
                window=window,
                member_window=members,
                service_name=name,
                previous_strong_share=held.strong.get(name),
                period_contribution=period,
                condition_contribution=condition,
            )
            signals[name] = vector(shared, mine)
            held.strong[name] = mine.strong_class_share
        held.share = shared.class_share
        return signals

    def _acted(
        self,
        held: LearnedValues,
        signals: Mapping[str, tuple[float, ...]],
        serving: Sequence[str],
        margin: float,
    ) -> Action:
        sampled = held.weights.sampled()
        inside = set(serving)
        target = tuple(
            name
            for name, mine in signals.items()
            if is_wanted(sampled.scores(mine).difference, is_member=name in inside, margin=margin)
        )
        expected = {name: held.weights.expected(mine) for name, mine in signals.items()}
        return action_toward(
            target,
            serving=serving,
            available=tuple(signals),
            ranked=lambda name: expected[name],
            margin=margin,
            limit=self._limit,
        )

    def _learn(
        self,
        held: LearnedValues,
        due: PendingAction,
        observed: Observation,
        members: Window,
        serving: Sequence[str],
    ) -> None:
        measured = contributions(
            observed.consumer,
            member_window=members,
            ensemble=serving,
            available=tuple(due.signals),
            profiles=observed.profiles,
            evaluation_window=self._window,
            window=observed.window,
        )
        for name, delta in measured.items():
            held.learn(due.signals[name], due.context, name, delta)

    def _for(self, observed: Observation) -> LearnedValues:
        consumer_id = observed.consumer.contract.consumer_id
        held = self._held.get(consumer_id)
        if held is not None:
            return held
        learned = LearnedValues(
            weights=Weights(
                dimension=self._dimension,
                prior_precision=self._prior_precision,
                draw=standard_normal(consumer_seed(self._seed, consumer_id)),
            ),
            margin=SwitchingMargin(
                initial=self._initial_margin,
                contributions_before_margin=self._before_margin,
            ),
            tables=IdentityTables(CONTEXT_KEYS),
            queue=FeedbackDelayQueue(
                delay_intervals=observed.consumer.contract.feedback.delay_intervals
            ),
            condition=NightDetector(night_confidence_drop=self._night_drop),
            noise_variance=self._noise_variance,
        )
        self._held[consumer_id] = learned
        return learned


def is_wanted(difference: float, *, is_member: bool, margin: float) -> bool:
    return difference > -margin if is_member else difference > margin


class CapSettings(ApproachSettings):
    evaluation_window: PositiveCount
    initial_switching_margin: PositiveNumber
    contributions_before_margin: PositiveCount
    prior_precision: PositiveNumber
    noise_variance: PositiveNumber
    night_confidence_drop: Ratio


def build(settings: CapSettings, *, classes: Sequence[str], seed: int) -> Cap:
    return Cap(
        classes=classes,
        max_ensemble_size=settings.max_ensemble_size,
        evaluation_window=settings.evaluation_window,
        initial_switching_margin=settings.initial_switching_margin,
        contributions_before_margin=settings.contributions_before_margin,
        prior_precision=settings.prior_precision,
        noise_variance=settings.noise_variance,
        night_confidence_drop=settings.night_confidence_drop,
        seed=seed,
    )


approach = Approach(name=Cap.name, api_version=API_VERSION, Settings=CapSettings, build=build)
