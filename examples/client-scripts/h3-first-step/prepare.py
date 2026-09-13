"""Create an unpublished probe project from an exact H3 wheel; makes no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("destination must be absent or empty")
    with zipfile.ZipFile(args.wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("expected one wheel distribution")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if metadata["Name"].replace("_", "-").lower() != "minimax-h3":
            raise ValueError("expected the exact MiniMax H3 serving wheel")
        members = []
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("unsafe wheel member")
            if any(part.endswith(".dist-info") for part in path.parts):
                continue
            if not name.endswith("/"):
                members.append(name)
        if "h3.py" not in members or "turbo.py" not in members:
            raise ValueError("wheel does not contain the H3 serving source")
        destination.mkdir(parents=True, exist_ok=True)
        for name in members:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    helper = Path(__file__).with_name("h3_first_step.py")
    shutil.copyfile(helper, destination / helper.name)
    quote = json.dumps
    dependencies = metadata.get_all("Requires-Dist", [])
    roots = sorted({PurePosixPath(name).parts[0] for name in members} | {helper.name})
    project = (
        '[project]\nname = "h3-first-step"\nversion = "0.1.2"\n'
        f"requires-python = {quote(metadata['Requires-Python'])}\n"
        "dependencies = [\n" + "".join(f"    {quote(value)},\n" for value in dependencies) + "]\n\n"
        '[project.entry-points."cozy.application"]\ndefault = "h3_first_step:app"\n\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
        "[tool.hatch.build.targets.wheel]\nonly-include = [\n"
        + "".join(f"    {quote(value)},\n" for value in roots)
        + "]\n\n"
        "[tool.uv]\ndefault-groups = []\n"
    )
    (destination / "pyproject.toml").write_text(project)
    (destination / "package.toml").write_text('[application]\nobject = "h3_first_step:app"\n')
    provenance = {
        "source_wheel": str(args.wheel.resolve()),
        "source_wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "h3_version": metadata["Version"],
        "diagnostic_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "project": str(destination),
    }
    # Keep observations outside the captured source inventory.
    proof_path = destination.with_name(destination.name + ".provenance.json")
    proof_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
