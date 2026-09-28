#!/usr/bin/env python3
"""Publish a committed first-party package only after its worker-image preflight passes."""

from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("anima", "minimax-h3", "minimax-h3-tools", "sdxl"))
    parser.add_argument("--tensorhub", type=Path, required=True, help="Tensorhub operator checkout")
    parser.add_argument(
        "--profile", action="append", required=True, help="supported worker profile"
    )
    parser.add_argument("--dotenv", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--secrets-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="check without publishing")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", args.package], cwd=repository, text=True
    )
    if dirty:
        parser.error(
            "commit the package changes before publishing; the source snapshot must be fixed"
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "archive", revision, "--", args.package], cwd=repository
    )
    command = ["go", "run", "./cmd/check-package-runtime"]
    for profile in args.profile:
        command.extend(("--profile", profile))
    for name in ("dotenv", "config", "secrets-dir"):
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            command.extend(("--" + name, str(value.resolve())))
    with tempfile.TemporaryDirectory(prefix="cozy-first-party-release-") as temporary:
        frozen = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(frozen, filter="data")
        project = frozen / args.package
        wheels = frozen / "dist"
        subprocess.run(
            ["uv", "build", "--project", str(project), "--wheel", "--out-dir", str(wheels)],
            check=True,
        )
        (wheel,) = wheels.glob("*.whl")
        print(f"Checking {args.package} from committed source {revision}", flush=True)
        subprocess.run([*command, str(wheel)], cwd=args.tensorhub.resolve(), check=True)
        if args.check_only:
            return
        # Both commands use the same archive: concurrent edits in the developer's
        # checkout cannot change what is published after the preflight.
        subprocess.run(["cozy", "package", "publish", "--json", "--full"], cwd=project, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
