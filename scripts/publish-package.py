#!/usr/bin/env python3
"""Publish a committed first-party package from a frozen snapshot of its source."""

from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    packages = sorted(path.parent.name for path in repository.glob("*/package.toml"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=packages)
    args = parser.parse_args()
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
    with tempfile.TemporaryDirectory(prefix="cozy-first-party-release-") as temporary:
        frozen = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(frozen, filter="data")
        print(f"Publishing {args.package} from committed source {revision}", flush=True)
        # Edits in the developer's checkout cannot change what is published.
        subprocess.run(
            ["cozy", "package", "publish", "--json", "--full"], cwd=frozen / args.package, check=True
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
