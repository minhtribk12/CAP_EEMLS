from __future__ import annotations

import itertools
import shutil
from pathlib import Path
from typing import Any

from support.repository import ROOT

FILE_TARGETS = ("swarm", "kube")


def of_kind(documents: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    return {one["metadata"]["name"]: one for one in documents if one["kind"] == kind}


def deploy_options(args: list[str]) -> list[dict[str, str]]:
    starts = [at for at, one in enumerate(args) if one == "run"] + [len(args)]
    calls = [args[a:b] for a, b in itertools.pairwise(starts)]
    return [
        dict(zip(call[3::2], call[4::2], strict=False))
        for call in calls
        if call[1:2] == ["eemls-deploy"] and call[2:3] and call[2] in FILE_TARGETS
    ]


def copied_scripts(root: Path, folder: str) -> None:
    shutil.copytree(ROOT / "deploy" / folder, root / "deploy" / folder, dirs_exist_ok=True)
    shutil.copy(ROOT / "deploy" / "common.sh", root / "deploy" / "common.sh")
