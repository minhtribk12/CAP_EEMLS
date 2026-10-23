from __future__ import annotations

from eemls.domain.errors import EemlsError

FORBIDDEN = ("+", "#", "/", "\x00")


class TopicError(EemlsError): ...


def result_topic(*, prefix: str, consumer_id: str, client_id: str) -> str:
    return (
        f"{_prefix(prefix)}/{_segment('consumer_id', consumer_id)}"
        f"/{_segment('client_id', client_id)}"
    )


def instance_topic(*, prefix: str, run_id: str, instance_id: str) -> str:
    return f"{_prefix(prefix)}/{_segment('run_id', run_id)}/{_segment('instance_id', instance_id)}"


def instances_of_run(*, prefix: str, run_id: str) -> str:
    return f"{_prefix(prefix)}/{_segment('run_id', run_id)}/+"


def parts_of(topic: str, *, prefix: str) -> tuple[str, str]:
    head = _prefix(prefix)
    if not topic.startswith(f"{head}/"):
        raise TopicError(f"{topic!r} is not under {head!r}")
    named = topic[len(head) + 1 :].split("/")
    if len(named) != 2:
        plural = "part" if len(named) == 1 else "parts"
        raise TopicError(
            f"{topic!r} names {len(named)} {plural} under {head!r}, and a result names two"
        )
    return _segment("consumer_id", named[0]), _segment("client_id", named[1])


def _segment(name: str, value: str) -> str:
    if not value:
        raise TopicError(f"{name} is empty, and a topic names something")
    held = [one for one in FORBIDDEN if one in value]
    if held:
        raise TopicError(f"{name} {value!r} holds {held}, which no topic carries")
    if value.strip() != value:
        raise TopicError(f"{name} {value!r} is padded with whitespace")
    return value


def _prefix(prefix: str) -> str:
    if not prefix:
        raise TopicError("the topic prefix is empty")
    if prefix.startswith("/") or prefix.endswith("/"):
        raise TopicError(f"the topic prefix {prefix!r} starts or ends with a separator")
    for part in prefix.split("/"):
        _segment("a prefix name", part)
    return prefix
