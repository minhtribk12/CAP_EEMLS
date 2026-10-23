from __future__ import annotations

import os

from eemls.domain.errors import EemlsError


class MissingPasswordError(EemlsError): ...


def password_from_environment() -> str:
    password = os.environ.get("EEMLS_DATABASE_PASSWORD")
    if not password:
        raise MissingPasswordError(
            "EEMLS_DATABASE_PASSWORD is not set, and a password is never in the tree"
        )
    return password
