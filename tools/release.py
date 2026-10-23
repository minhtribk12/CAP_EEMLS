from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pin import ROOT, framework_version, tag_name


def git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release")
    parser.add_argument("version")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    version = framework_version(args.root)
    if args.version != version:
        print(f"refused: {args.version} is not the root version {version}", file=sys.stderr)
        return 1
    changed = git(args.root, "status", "--porcelain", "--", "src", "pyproject.toml")
    if changed.returncode != 0 or changed.stdout.strip():
        print("refused: commit the changes under src/ and pyproject.toml first", file=sys.stderr)
        return 1
    tagged = git(args.root, "tag", tag_name(version))
    if tagged.returncode != 0:
        print(f"refused: {tagged.stderr.strip()}", file=sys.stderr)
        return 1
    print(tag_name(version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
