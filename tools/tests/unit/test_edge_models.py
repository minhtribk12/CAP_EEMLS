from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from eemls.testing.trees import committed_tree
from support.repository import ROOT

# poetry runs this interpreter; ssh and rsync write their arguments to calls.
POETRY = f"""#!/bin/sh
shift
case "$1" in
  python3) shift; exec {sys.executable} "$@" ;;
  eemls-config) shift; exec {sys.executable} -c \\
    "import sys; from eemls_tools.config_command import main; sys.exit(main(sys.argv[1:]))" "$@" ;;
esac
"""
RECORD = '#!/bin/sh\nprintf "%s\\n" "$@" >> "$(dirname "$0")/calls"\n'


def copied(root: Path, **env: str) -> list[str]:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "poetry").write_text(POETRY, encoding="utf-8")
    for tool in ("ssh", "rsync"):
        (bin_dir / tool).write_text(RECORD, encoding="utf-8")
    for tool in ("poetry", "ssh", "rsync"):
        (bin_dir / tool).chmod(0o755)
    shutil.copytree(ROOT / "deploy" / "edge", root / "deploy" / "edge")
    committed_tree(root / "config")
    (root / "config" / "local").mkdir()
    shutil.copy(ROOT / "examples" / "local" / "models.yaml", root / "config" / "local")
    for model in ("b1_general", "b2_general", "b3_motorcycle", "b4_general"):
        (root / "models" / model).mkdir(parents=True)
    done = subprocess.run(
        ["sh", "deploy/edge/models.sh", "pi@device", "raspberry_pi4"],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PYTHONPATH": os.pathsep.join(sys.path),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return (bin_dir / "calls").read_text(encoding="utf-8").splitlines()


def test_it_copies_the_models_of_the_committed_pool(tmp_path: Path) -> None:
    calls = copied(tmp_path)

    assert "models/b1_general/" in calls
    assert "models/b4_general/" not in calls


def test_it_copies_the_models_of_the_pool_the_overrides_give(tmp_path: Path) -> None:
    pool = yaml.safe_load((ROOT / "examples" / "overrides" / "small_pool.yaml").read_text())
    pool["models/inference_services.yaml"]["services"]["infer-b1-general-pi4"]["model"] = (
        "b4_general"
    )
    (tmp_path / "pool.yaml").write_text(yaml.safe_dump(pool), encoding="utf-8")

    calls = copied(tmp_path, OVERRIDES="pool.yaml")

    assert "models/b4_general/" in calls
    assert "models/b1_general/" not in calls
