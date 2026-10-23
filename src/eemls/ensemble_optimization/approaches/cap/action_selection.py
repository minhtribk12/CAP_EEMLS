from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

from eemls.domain.types import Action

Ranking = Callable[[str], float]


def action_toward(
    target: Collection[str],
    *,
    serving: Sequence[str],
    available: Sequence[str],
    ranked: Ranking,
    margin: float,
    limit: int,
) -> Action:
    held = set(target)
    inside = sorted(serving)
    outside = sorted(name for name in available if name not in set(serving))
    wanted = [name for name in outside if name in held]
    loose = [name for name in inside if name not in held]
    if wanted:
        coming = _best(wanted, ranked)
        if len(inside) < limit:
            return Action(kind="add", member_in=coming)
        if loose:
            return Action(kind="replace", member_in=coming, member_out=_worst(loose, ranked))
    elif loose and len(inside) > 1:
        return Action(kind="remove", member_out=_worst(loose, ranked))
    if not loose and outside and inside:
        going = _worst(inside, ranked)
        coming = _best(outside, ranked)
        if ranked(coming) - ranked(going) > margin:
            return Action(kind="replace", member_in=coming, member_out=going)
    return Action(kind="keep")


def _best(names: Sequence[str], ranked: Ranking) -> str:
    return max(names, key=ranked)


def _worst(names: Sequence[str], ranked: Ranking) -> str:
    return min(names, key=ranked)
