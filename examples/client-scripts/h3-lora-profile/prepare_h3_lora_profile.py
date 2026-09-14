"""Prepare a private H3 profile project without changing its serving source or running it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    if digest != "3a7f8fbf97348ea53ff08016e9de6a59d4929e87a25d52185f78d42dbe40a627":
        raise ValueError("expected the reviewed published H3 1.14.3 wheel")
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
    text = project.read_text().replace('name = "h3-first-step"', 'name = "h3-lora-profile"')
    text = text.replace('version = "0.1.2"', 'version = "0.1.0"')
    text = text.replace("h3_first_step", "h3_lora_profile")
    text = text.replace(
        '    "h3_lora_profile.py",', '    "h3_lora_profile.py",\n    "profile_sources.json",'
    )
    project.write_text(text)
    (destination / "package.toml").write_text('[application]\nobject = "h3_lora_profile:app"\n')
    (destination / "h3_first_step.py").unlink()
    source = here / "h3_lora_profile.py"
    shutil.copyfile(source, destination / source.name)
    provenance = {
        "h3_wheel_sha256": digest,
        "driver_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "one profiled DiT step inside ordinary standard30 FL2VA; no weight mutation",
    }
    (destination / "profile_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    destination.with_name(destination.name + ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
