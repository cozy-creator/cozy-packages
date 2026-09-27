"""Extract an exact serving wheel and add the private matched-source comparison job."""

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
        metadata_paths = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("expected one wheel distribution")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if metadata["Name"].replace("_", "-").lower() != "minimax-h3":
            raise ValueError("expected the MiniMax H3 serving wheel")
        members = []
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("unsafe wheel member")
            if any(p.endswith(".dist-info") for p in path.parts) or name.endswith("/"):
                continue
            members.append(name)
        if not {"h3.py", "long_form_state.py", "story.py"}.issubset(members):
            raise ValueError("wheel must contain the complete serving source")
        destination.mkdir(parents=True, exist_ok=True)
        for name in members:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    helper = Path(__file__).with_name("h3_motion_context.py")
    shutil.copyfile(helper, destination / helper.name)
    roots = sorted({PurePosixPath(n).parts[0] for n in members} | {helper.name})
    quote = json.dumps
    project = (
        '[project]\nname = "minimax-h3"\n'
        f"version = {quote(metadata['Version'])}\n"
        f"requires-python = {quote(metadata['Requires-Python'])}\n"
        "dependencies = [\n"
        + "".join(f"    {quote(v)},\n" for v in metadata.get_all("Requires-Dist", []))
        + "]\n\n"
        '[project.entry-points."cozy.application"]\ndefault = "h3_motion_context:app"\n\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
        "[tool.hatch.build.targets.wheel]\nonly-include = [\n"
        + "".join(f"    {quote(v)},\n" for v in roots)
        + "]\n\n"
        '[tool.uv.sources]\nqwen-image-2 = { index = "tensorhub" }\n'
    )
    (destination / "pyproject.toml").write_text(project)
    (destination / "package.toml").write_text('[application]\nobject = "h3_motion_context:app"\n')
    (destination.with_name(destination.name + ".provenance.json")).write_text(
        json.dumps(
            {
                "source_wheel": str(args.wheel.resolve()),
                "source_wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                "diagnostic_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "h3_version": metadata["Version"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
