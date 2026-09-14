"""Prepare an unpublished compiler oracle from an exact MiniMax H3 serving wheel."""

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
    source = Path(__file__).resolve().parent
    destination = args.destination.resolve()
    # Reuse the source-complete extraction and dependency preservation of the
    # first-step diagnostic. That preparer refuses a nonempty destination.
    subprocess.run(
        [
            sys.executable,
            str(source.parent / "h3-first-step" / "prepare_h3_first_step.py"),
            str(args.wheel.resolve()),
            str(destination),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    project = destination / "pyproject.toml"
    document = project.read_text()
    for previous, replacement in (
        ('name = "h3-first-step"', 'name = "h3-compile-oracle"'),
        ('version = "0.1.2"', 'version = "0.1.0"'),
        ("h3_first_step", "h3_compile_oracle"),
    ):
        if previous not in document:
            raise ValueError(f"first-step preparer changed: missing {previous!r}")
        document = document.replace(previous, replacement)
    project.write_text(document)
    (destination / "package.toml").write_text('[application]\nobject = "h3_compile_oracle:app"\n')
    (destination / "h3_first_step.py").unlink()
    helper = source / "h3_compile_oracle.py"
    shutil.copyfile(helper, destination / helper.name)
    proof = destination.with_name(destination.name + ".provenance.json")
    provenance = json.loads(proof.read_text())
    provenance["diagnostic_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    provenance["application"] = "h3_compile_oracle:app"
    provenance["preparer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    proof.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
