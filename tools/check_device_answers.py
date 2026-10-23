"""Check one inference service answers the same on a device as here.

Usage:
    python3 tools/check_device_answers.py --host <username>@<host> --service infer-b1-general-pi4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from eemls_tools.config_tree import load_config
from eemls_tools.datasets import MANIFEST, manifest_rows
from eemls_tools.deployment_plan import Images

ROOT = Path(__file__).resolve().parents[1]

TOLERANCE = 0.01
HUB = ROOT / "data" / "hub"
STORE = "/srv/eemls/smoke"
CONFIG = "/srv/eemls/smoke_config"
MODELS = "/srv/eemls/models"

# Start the data hub and the service; print one answer per image.
SERVE = r"""
serve() {
set -eu
IMAGE="$1" HUB_IMAGE="$2" SERVICE="$3" STORE="$4" MODELS="$5" SOURCES="$6" CONFIG="$7"
shift 7
GPU=""
if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
  case "$(uname -m)" in aarch64) GPU="--runtime nvidia" ;; *) GPU="--gpus all" ;; esac
fi
stop_services() {
  docker rm -f eemls-smoke-hub eemls-smoke-inference >/dev/null 2>&1 || true
  docker network rm eemls-smoke >/dev/null 2>&1 || true
}
stop_services
trap stop_services EXIT
docker network create eemls-smoke >/dev/null
docker pull -q "$HUB_IMAGE" >&2
docker pull -q "$IMAGE" >&2
docker run -d --name eemls-smoke-hub --network eemls-smoke --network-alias data-hub \
  -p 127.0.0.1:18081:8081 -v "$STORE:/var/lib/eemls/images:ro" \
  -v "$CONFIG:/opt/eemls/config:ro" "$HUB_IMAGE" >/dev/null
docker run -d --name eemls-smoke-inference --network eemls-smoke $GPU -p 127.0.0.1:18082:8082 \
  -v "$MODELS:/opt/eemls/models:ro" \
  -v "$CONFIG:/opt/eemls/config:ro" \
  -v "$SOURCES:/opt/eemls/model_sources.yaml:ro" \
  -e EEMLS_MODEL_SOURCES=/opt/eemls/model_sources.yaml \
  -e EEMLS_SERVICE_NAME="$SERVICE" -e EEMLS_INSTANCE_ID=smoke -e EEMLS_NODE="$(hostname)" \
  "$IMAGE" >/dev/null
waited=0
until curl -sf http://127.0.0.1:18081/readyz >/dev/null \
  && curl -sf http://127.0.0.1:18082/readyz >/dev/null; do
  waited=$((waited + 2))
  if [ "$waited" -gt 600 ]; then docker logs --tail 20 eemls-smoke-inference >&2; exit 1; fi
  sleep 2
done
for id in "$@"; do
  curl -sf -X POST http://127.0.0.1:18082/predict -H 'content-type: application/json' \
    -d "{\"image_id\": \"$id\"}"
  echo
done
}
serve "$@" </dev/null
"""


class ServeError(Exception): ...


@dataclass(frozen=True)
class Agreement:
    label_here: str
    label_there: str
    confidence_here: float
    confidence_there: float
    provider: str

    @property
    def same(self) -> bool:
        return (
            self.label_here == self.label_there
            and abs(self.confidence_here - self.confidence_there) <= TOLERANCE
        )


def spread_by_label(rows: Sequence[Mapping[str, str]], count: int) -> list[Mapping[str, str]]:
    by_label: dict[str, list[Mapping[str, str]]] = {}
    for one in sorted(rows, key=lambda one: one["image_id"]):
        by_label.setdefault(one["label"], []).append(one)
    turns = zip_longest(*(by_label[label] for label in sorted(by_label)))
    return [one for turn in turns for one in turn if one is not None][:count]


def store_of(rows: Sequence[Mapping[str, str]], source: Path, out: Path) -> Path:
    for one in rows:
        target = out / one["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / one["path"], target)
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(
        "".join(json.dumps(dict(one), sort_keys=True) + "\n" for one in rows), encoding="utf-8"
    )
    return out


def compare_answers(here: Sequence[str], there: Sequence[str]) -> list[Agreement]:
    if len(there) != len(here):
        raise ValueError(f"the device answered {len(there)} of {len(here)} images")
    held = []
    for mine, theirs in zip(here, there, strict=True):
        one, other = json.loads(mine), json.loads(theirs)
        held.append(
            Agreement(
                label_here=one["predicted_label"],
                label_there=other["predicted_label"],
                confidence_here=one["confidence"],
                confidence_there=other["confidence"],
                provider=other["execution_provider"],
            )
        )
    return held


def served(command: list[str], arguments: list[str]) -> list[str]:
    done = subprocess.run(
        [*command, "sh", "-s", "--", *arguments],
        input=SERVE,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise ServeError(f"{' '.join(command) or 'here'}: {done.stderr.strip()[-2000:]}")
    return [line for line in done.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_device_answers")
    parser.add_argument("--host", required=True, help="the device, as user@host")
    parser.add_argument("--service", required=True)
    parser.add_argument("--tag", help="the image tag the device runs, else its device class's")
    parser.add_argument("--images", type=int, default=20)
    args = parser.parse_args(argv)
    config = load_config(ROOT / "config")
    named = config.images
    if named is None:
        print(
            "refused: no config/local/images.yaml; run tools/new_local.py and fill it in",
            file=sys.stderr,
        )
        return 1
    images = Images(registry=named.registry, version=named.version)
    service = config.models.inference_services.services[args.service]
    provider = config.cost.devices[service.device_type].execution_provider
    tags = config.services.deployment.image_tags
    tag = args.tag or tags[provider]
    rows = manifest_rows(HUB)
    chosen = spread_by_label(rows, args.images)
    ids = [one["image_id"] for one in chosen]
    with tempfile.TemporaryDirectory() as scratch:
        store = store_of(chosen, HUB, Path(scratch) / "store")
        store.chmod(0o755)
        here = served(
            [],
            [
                images.of("inference-service", tags["x86_cpu"]),
                images.of("data-hub"),
                args.service,
                str(store),
                str(ROOT / "models"),
                str(ROOT / "config" / "local" / "models.yaml"),
                str(ROOT / "config"),
                *ids,
            ],
        )
        sent = [
            "rsync",
            "-a",
            "--delete",
            "--chmod=Da+rx,Fa+r",
            f"{store}/",
            f"{args.host}:{STORE}/",
        ]
        subprocess.run(sent, check=True)
    synced = ["rsync", "-a", "--delete", "--exclude=/local/", "--chmod=Da+rx,Fa+r"]
    subprocess.run([*synced, f"{ROOT / 'config'}/", f"{args.host}:{CONFIG}/"], check=True)
    there = served(
        ["ssh", args.host],
        [
            images.of("inference-service", tag),
            images.of("data-hub"),
            args.service,
            STORE,
            MODELS,
            f"{MODELS}/models.yaml",
            CONFIG,
            *ids,
        ],
    )
    held = compare_answers(here, there)
    same = sum(one.same for one in held)
    print(
        f"{args.host} {args.service} {tag}: {same} of {len(held)} agree, on "
        f"{', '.join(sorted({one.provider for one in held}))}"
    )
    for image_id, one in zip(ids, held, strict=True):
        if not one.same:
            print(
                f"  {image_id}: here {one.label_here} {one.confidence_here:.4f}, "
                f"there {one.label_there} {one.confidence_there:.4f}"
            )
    return 0 if same == len(held) else 1


if __name__ == "__main__":
    sys.exit(main())
