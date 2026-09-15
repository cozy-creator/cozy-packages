#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Publish a committed first-party package only after its Runtime image preflight."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def main() -> None:
    policy = json.loads((ROOT / "release-profiles.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=policy["packages"])
    parser.add_argument("--tensorhub-dir", type=Path, default=ROOT.parent / "tensorhub")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dotenv", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if run("git", "status", "--porcelain", "--untracked-files=no", "--", args.package):
        raise SystemExit("Commit the package's source, metadata and lock before releasing it.")
    revision = run("git", "rev-parse", "HEAD")
    releases = ROOT / ".local" / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.package + "-", dir=releases))
    # Publish this same committed snapshot after checking it; edits in the working
    # checkout cannot change the package between the gate and Creator capture.
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", revision, args.package], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(output, filter="data")
    project = output / args.package
    wheels = output / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels), str(project)], check=True
    )
    built = list(wheels.glob("*.whl"))
    if len(built) != 1:
        raise SystemExit("Expected one wheel for the committed package.")
    hub = args.tensorhub_dir.resolve()
    command = [
        "go", "run", "./cmd/check-package-runtime",
        "--config", str((args.config or hub / "config.local.yaml").resolve()),
        "--dotenv", str((args.dotenv or hub / ".env").resolve()),
        "--purpose", "both",
    ]
    for profile in policy["runtime_profiles"]:
        command += ["--profile", profile]
    command.append(str(built[0]))
    proof = run(*command, cwd=hub)
    measured = json.loads(proof)
    (output / "runtime-images.json").write_text(
        json.dumps({"source_commit": revision, **measured}, indent=2) + "\n"
    )
    print(f"Runtime image preflight passed; evidence: {output / 'runtime-images.json'}", flush=True)
    if not args.preflight_only:
        subprocess.run(["cozy", "package", "publish"], cwd=project, check=True)


if __name__ == "__main__":
    main()
