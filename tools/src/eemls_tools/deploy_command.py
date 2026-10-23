"""`eemls-deploy`: write the Swarm stack or the Kubernetes folder.

    eemls-deploy swarm --config config --profiles models/profiles.jsonl \\
        --images /srv/eemls/images --run-id <run id> --seed 0 --out stack.yaml
    eemls-deploy kube --config config --profiles models/profiles.jsonl \\
        --images /srv/eemls/images --run-id <run id> --seed 0 --out runs/<run>/kube
    eemls-deploy labels --config config [--device-type <device type>] [--core]
    eemls-deploy name --config config metrics_database
    eemls-deploy pod --config config --name <pod> --image consumer --out pod.yaml -- <command>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from eemls.domain.records import InferenceServiceProfile
from eemls.observation_service.profile_updates import profiles_in
from eemls_tools.config_tree import ConfigError, ConfigTree, load_config
from eemls_tools.deployment_plan import (
    LATEST_VERSION,
    DeployError,
    Images,
    Mounts,
    RunSettings,
    node_labels,
    plan_deployment,
    service_names,
)
from eemls_tools.kube_files import KubeFiles, kube_files, one_off_pod
from eemls_tools.swarm_stack import swarm_stack

GROUPS = ("core", "inference", "consumers")
DEFAULT_REGISTRY = "localhost:5000"


def measured_throughput(path: Path) -> dict[str, float]:
    newest: dict[str, InferenceServiceProfile] = {}
    for held in profiles_in(path):
        known = newest.get(held.service_name)
        if known is None or held.window_end > known.window_end:
            newest[held.service_name] = held
    return {name: held.throughput_rps for name, held in newest.items()}


def _write_files(folder: Path, files: KubeFiles) -> None:
    for group in GROUPS:
        for stale in (folder / group).glob("*.yaml"):
            stale.unlink()
    for path, held in files.items():
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump_all(held, sort_keys=False), encoding="utf-8")


def _images(config: ConfigTree, args: argparse.Namespace) -> Images:
    local_images = config.images
    return Images(
        registry=args.registry or (local_images.registry if local_images else DEFAULT_REGISTRY),
        version=args.version or (local_images.version if local_images else LATEST_VERSION),
    )


def _with_images(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", help=f"local/images.yaml of the tree, or {DEFAULT_REGISTRY}")
    parser.add_argument("--version", help=f"local/images.yaml of the tree, or {LATEST_VERSION}")


def _files_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--images", required=True, help="host folder of the prepared images")
    parser.add_argument("--out", type=Path, required=True, help="a file, or kube's folder")
    _with_images(parser)
    parser.add_argument("--mount-config", help="host folder mounted as config; default --config")
    parser.add_argument(
        "--models",
        default="/srv/eemls/models",
        help="folder on each device holding its models and their models.yaml",
    )
    parser.add_argument("--run-id", required=True, help="the run every row is stored under")
    parser.add_argument("--scenario", help="add one consumer service per consumer, following it")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("experiments/scenarios"),
        help="host folder of the scenario files, mounted into the consumers",
    )
    parser.add_argument("--duration-s", type=float)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="the seed the control plane and every consumer draw from",
    )


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eemls-deploy")
    targets = parser.add_subparsers(dest="target", required=True)
    for target in ("swarm", "kube"):
        files = targets.add_parser(target)
        files.add_argument("--config", type=Path, required=True)
        _files_arguments(files)
    labels = targets.add_parser("labels", help="print the node labels, as <label>=true")
    labels.add_argument("--config", type=Path, required=True)
    labels.add_argument("--device-type", help="only this device type; default every one")
    labels.add_argument("--core", action="store_true", help="add the label of the core services")
    name = targets.add_parser("name", help="print the host name of one service")
    name.add_argument("--config", type=Path, required=True)
    name.add_argument("service")
    pod = targets.add_parser("pod", help="write a one-off pod on a core node")
    pod.add_argument("--config", type=Path, required=True)
    pod.add_argument("--name", required=True)
    pod.add_argument("--image", required=True, help="the eemls image, such as consumer")
    pod.add_argument("--out", type=Path, required=True)
    _with_images(pod)
    pod.add_argument("command", nargs="+")
    args = parser.parse_args(argv)
    if args.target in ("swarm", "kube") and (args.scenario is None) != (args.duration_s is None):
        parser.error("consumers need --scenario and --duration-s together")
    return args


def _labels(config: ConfigTree, args: argparse.Namespace) -> int:
    devices = (args.device_type,) if args.device_type else tuple(sorted(config.cost.devices))
    print(" ".join(f"{label}=true" for label in node_labels(config, devices, core=args.core)))
    return 0


def _pod(config: ConfigTree, args: argparse.Namespace) -> int:
    manifest = one_off_pod(
        str(args.config),
        name=args.name,
        image=_images(config, args).of(args.image),
        command=args.command,
    )
    args.out.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return 0


def _files(config: ConfigTree, args: argparse.Namespace) -> int:
    run = RunSettings(args.run_id, args.scenario, args.duration_s, seed=args.seed)
    mounts = Mounts(
        images=args.images,
        config=args.mount_config or str(args.config.resolve()),
        models=args.models,
        scenarios=str(args.scenarios.resolve()),
    )
    deployment = plan_deployment(config, measured_throughput(args.profiles))
    images = _images(config, args)
    if args.target == "swarm":
        text = yaml.safe_dump(
            swarm_stack(deployment, images=images, mounts=mounts, run=run), sort_keys=False
        )
        args.out.write_text(text, encoding="utf-8")
    else:
        _write_files(args.out, kube_files(deployment, images=images, mounts=mounts, run=run))
    print(f"{args.target} files for {len(deployment.pool)} inference services in {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        config = load_config(args.config)
        if args.target == "labels":
            return _labels(config, args)
        if args.target == "name":
            names = service_names(config)
            if args.service not in names:
                raise DeployError(f"{args.service} is not one of {', '.join(names)}")
            print(names[args.service])
            return 0
        if args.target == "pod":
            return _pod(config, args)
        return _files(config, args)
    except (ConfigError, DeployError, OSError, ValueError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
