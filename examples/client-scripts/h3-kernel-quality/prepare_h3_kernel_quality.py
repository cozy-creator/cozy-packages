"""Stage unchanged H3 serving source plus reviewed private attention helpers; run no models."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

H3_WHEEL = "3a7f8fbf97348ea53ff08016e9de6a59d4929e87a25d52185f78d42dbe40a627"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("backend_project", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if digest(args.wheel) != H3_WHEEL:
        raise ValueError("expected the reviewed published MiniMax H3 1.14.3 wheel")
    here = Path(__file__).resolve().parent
    destination = args.destination.resolve()
    backend = args.backend_project.resolve()
    source_config = tomllib.loads((backend / "pyproject.toml").read_text())
    selected = "cozy-h3-fa3-fp8-tile128"
    kernel = Path(source_config["tool"]["uv"]["sources"][selected]["path"])
    if not kernel.is_absolute():
        kernel = backend / kernel
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
    pyproject = destination / "pyproject.toml"
    text = pyproject.read_text()
    for old, new in (
        ('name = "h3-first-step"', 'name = "h3-kernel-quality"'),
        ('version = "0.1.2"', 'version = "0.1.0"'),
        ("h3_first_step", "h3_kernel_quality"),
    ):
        if old not in text:
            raise ValueError(f"source preparer changed: missing {old!r}")
        text = text.replace(old, new)
    text = text.replace("dependencies = [\n", f'dependencies = [\n    "{selected}",\n', 1)
    text = text.replace(
        '    "h3_kernel_quality.py",\n',
        '    "h3_kernel_quality.py",\n    "attention_backends.py",\n'
        '    "attention_quantized.py",\n    "quality_sources.json",\n',
        1,
    )
    kernel_target = destination / "wheels" / kernel.name
    kernel_target.parent.mkdir()
    shutil.copyfile(kernel, kernel_target)
    text += f'\n[tool.uv.sources]\n{selected} = {{ path = "wheels/{kernel.name}" }}\n'
    pyproject.write_text(text)
    (destination / "package.toml").write_text('[application]\nobject = "h3_kernel_quality:app"\n')
    (destination / "h3_first_step.py").unlink()
    sources = [
        here / "h3_kernel_quality.py",
        backend / "attention_backends.py",
        backend / "attention_quantized.py",
    ]
    for source in sources:
        shutil.copyfile(source, destination / source.name)
    provenance = {
        "h3_version": "1.14.3",
        "source_wheel_sha256": H3_WHEEL,
        "source_files": {p.name: digest(p) for p in sources},
        "candidate_wheel": {"filename": kernel.name, "sha256": digest(kernel)},
        "preparer_sha256": digest(Path(__file__)),
        "application": "h3_kernel_quality:app",
        "scope": "ordinary standard30 FL2VA, one GPU, only 50 main attention blocks",
    }
    (destination / "quality_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    destination.with_name(destination.name + ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
