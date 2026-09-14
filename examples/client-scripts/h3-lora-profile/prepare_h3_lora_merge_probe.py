"""Stage the reviewed merge arithmetic alongside unchanged H3 serving source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("arithmetic", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if digest(args.wheel) != "3a7f8fbf97348ea53ff08016e9de6a59d4929e87a25d52185f78d42dbe40a627":
        raise ValueError("expected published H3 1.14.3 source")
    if (
        digest(args.arithmetic)
        != "cb2d602d1e10d7dba3e28f93016d8649cace40e9e4983082316f947a8886e238"
    ):
        raise ValueError("expected the reviewed scale-policy merge arithmetic")
    here = Path(__file__).resolve().parent
    destination = args.destination.resolve()
    subprocess.run(
        [
            sys.executable,
            str(here.parent / "h3-first-step" / "prepare_h3_first_step.py"),
            str(args.wheel.resolve()),
            str(destination),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    project = destination / "pyproject.toml"
    text = project.read_text().replace('name = "h3-first-step"', 'name = "h3-lora-merge-probe"')
    text = text.replace('version = "0.1.2"', 'version = "0.1.0"').replace(
        "h3_first_step", "h3_lora_merge_probe"
    )
    text = text.replace(
        '    "h3_lora_merge_probe.py",',
        '    "h3_lora_merge_probe.py",\n    "merge_arithmetic.py",\n    "merge_sources.json",',
    )
    project.write_text(text)
    (destination / "package.toml").write_text('[application]\nobject = "h3_lora_merge_probe:app"\n')
    (destination / "h3_first_step.py").unlink()
    source = here / "h3_lora_merge_probe.py"
    shutil.copyfile(source, destination / source.name)
    shutil.copyfile(args.arithmetic, destination / "merge_arithmetic.py")
    facts = {
        "h3_wheel_sha256": digest(args.wheel),
        "driver_sha256": digest(source),
        "arithmetic_sha256": digest(args.arithmetic),
        "preparer_sha256": digest(Path(__file__)),
        "scope": (
            "one actual Q projection after one denoise step; separate candidate buffers; "
            "no cache or complete video"
        ),
    }
    for path in (
        destination / "merge_sources.json",
        destination.with_name(destination.name + ".provenance.json"),
    ):
        path.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
