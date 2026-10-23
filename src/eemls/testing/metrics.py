from __future__ import annotations

import re


def series(body: str, name: str, **labels: str) -> float | None:
    wanted = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    selector = f"{{{wanted}}}" if labels else ""
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        if sample == f"{name}{selector}":
            return float(value)
    return None


def published(body: str, name: str) -> bool:
    return bool(re.search(rf"^# TYPE {re.escape(name)} ", body, re.MULTILINE))


def histogram_count(body: str, name: str, **labels: str) -> float | None:
    return series(body, f"{name}_count", **labels)


def histogram_sum(body: str, name: str, **labels: str) -> float | None:
    return series(body, f"{name}_sum", **labels)


def bucket_edges(body: str, name: str, **labels: str) -> list[float]:
    edges: list[float] = []
    for line in body.splitlines():
        if line.startswith("#") or f"{name}_bucket{{" not in line:
            continue
        inside = line[line.index("{") + 1 : line.rindex("}")]
        pairs: dict[str, str] = {}
        for part in inside.split(","):
            key, _, value = part.partition("=")
            pairs[key] = value.strip('"')
        if all(pairs.get(key) == value for key, value in labels.items()):
            edges.append(float(pairs["le"]))
    return edges
