from __future__ import annotations

import hashlib


def consumer_seed(seed: int, consumer_id: str) -> int:
    held = hashlib.sha256(f"{seed}:{consumer_id}".encode()).digest()
    return int.from_bytes(held[:8], "big")
