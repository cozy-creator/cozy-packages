"""Stage the exact old/new Runtime methods for a bounded actual-input comparison."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

OLD = "381d8077e3523790a51257568ea6dce347b02111"
NEW = "1f07bba3cadc9f9b91fd8316739e88ea9f3abf34"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("arithmetic", type=Path)
    parser.add_argument("runtime_repo", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    destination = args.destination.resolve()
    subprocess.run(
        [
            sys.executable,
            str(here / "prepare_h3_lora_merge_probe.py"),
            str(args.wheel.resolve()),
            str(args.arithmetic.resolve()),
            str(destination),
        ],
        check=True,
    )
    facts = {}
    for label, commit in (("old", OLD), ("candidate", NEW)):
        full = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.runtime_repo),
                "show",
                commit + ":src/cozy_runtime/internal/lora.py",
            ],
            text=True,
        )
        tree = ast.parse(full)
        cls = next(
            x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "LinearAdapters"
        )
        function = next(
            x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_apply"
        )
        body = textwrap.dedent(
            "\n".join(full.splitlines()[function.lineno - 1 : function.end_lineno]) + "\n"
        )
        (destination / (label + "_apply.py")).write_text(body)
        facts[label + "_apply_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        facts[label + "_commit"] = commit
    source = here / "h3_lora_scratch_probe.py"
    shutil.copyfile(source, destination / source.name)
    facts["driver_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    facts["h3_wheel_sha256"] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    facts["preparer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (destination / "scratch_sources.json").write_text(json.dumps(facts, indent=2) + "\n")
    project = destination / "pyproject.toml"
    text = project.read_text().replace(
        'name = "h3-lora-merge-probe"', 'name = "h3-lora-scratch-probe"'
    )
    text = text.replace(
        'default = "h3_lora_merge_probe:app"', 'default = "h3_lora_scratch_probe:app"'
    )
    text = text.replace(
        '    "h3_lora_merge_probe.py",',
        '    "h3_lora_merge_probe.py",\n    "h3_lora_scratch_probe.py",\n'
        '    "candidate_apply.py",\n    "scratch_sources.json",',
    )
    project.write_text(text)
    (destination / "package.toml").write_text(
        '[application]\nobject = "h3_lora_scratch_probe:app"\n'
    )
    subprocess.run(["uv", "lock", "--offline"], cwd=destination, check=True)
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
