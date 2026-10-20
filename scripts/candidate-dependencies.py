#!/usr/bin/env python3
"""Lock a runner checkout against unpublished Runtime and TensorFS release wheels.

The normal static checks and `task conform` then use these candidate locks. This changes
only the runner checkout; its locks are candidate evidence, not publication locks.
"""

from __future__ import annotations

import argparse
import email
import json
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


def requirement(wheel: Path, name: str, floor: tuple[int, int, int]) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(f"{wheel.name}: expected one wheel METADATA")
        message = email.message_from_bytes(archive.read(metadata[0]))
    version = str(message["Version"])
    if str(message["Name"]).replace("_", "-").lower() != name:
        raise ValueError(f"{wheel.name}: expected {name}, got {message['Name']}")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or tuple(map(int, version.split("."))) < floor:
        raise ValueError(f"{wheel.name}: {name} {version} is below the candidate release floor")
    print(f"candidate: {name} {version} from {wheel}", flush=True)
    return f"{name} @ {wheel.resolve().as_uri()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--tensorfs-wheel", type=Path, required=True)
    parser.add_argument("--account-index", default="https://tensorhub.com/v1/index/fidika/simple/")
    parser.add_argument("projects", nargs="+")
    args = parser.parse_args()
    requirements = [
        requirement(args.runtime_wheel, "cozy-runtime", (0, 30, 0)),
        requirement(args.tensorfs_wheel, "tensorfs", (0, 13, 0)),
    ]
    evidence = Path(".candidate-dependencies")
    evidence.mkdir(exist_ok=True)
    (evidence / "requirements.txt").write_text("\n".join(requirements) + "\n")
    environment = f"sys_platform == '{sys.platform}' and platform_machine == '{platform.machine()}'"
    for project in args.projects:
        path = Path(project) / "pyproject.toml"
        source = path.read_text()
        uv = tomllib.loads(source)["tool"]["uv"]
        if "override-dependencies" in uv:
            raise ValueError(f"{project}: candidate overrides already exist")
        source = source.replace(
            "[tool.uv]\n",
            "[tool.uv]\noverride-dependencies = " + json.dumps(requirements) + "\n",
            1,
        )
        source, changed = re.subn(
            r"(?m)^environments = .*?$", "environments = " + json.dumps([environment]), source
        )
        if changed != 1:
            raise ValueError(f"{project}: expected one uv environment declaration")
        path.write_text(source)
        subprocess.run(
            [
                "uv", "lock", "--python", "3.12", "--project", project,
                "--index", f"tensorhub={args.account_index}",
            ],
            check=True,
        )
        destination = evidence / project
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path.parent / "uv.lock", destination / "uv.lock")


if __name__ == "__main__":
    main()
