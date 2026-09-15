"""Stage a source-guarded private profiling package and exact capture commands; no run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/client-scripts/h3-scaling-profile"
NATIVE = "06dd8828d620334912b97bc5bf1ef6ba363d4586ccab882b79b4e4a0d7d7e812"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--backend", choices=("flash-attn3", "sol-attn"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.runtime_manifest.read_text())
    wheel = Path(manifest["wheel"])
    if (
        manifest["source_commit"] != args.expected_source
        or manifest["sha256"] != args.expected_wheel_sha256
        or sha(wheel) != args.expected_wheel_sha256
        or manifest["native_extension_sha256"] != NATIVE
    ):
        raise SystemExit(
            "Runtime manifest differs from the explicitly reviewed source/wheel/native"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    package = args.out / "package"
    package.mkdir()
    for name in ("h3_scaling_profile.py", "profile_trace.py", "package.toml"):
        shutil.copyfile(SOURCE / name, package / name)
    code = (package / "h3_scaling_profile.py").read_text()
    code = code.replace(
        'RUNTIME_SOURCE = "d140609bf5fd07f8c67eaf7e87de433f5177f94f"',
        "RUNTIME_SOURCE = " + json.dumps(args.expected_source),
    ).replace(
        'ATTENTION_BACKEND: Literal["sol-attn", "flash-attn3"] = "flash-attn3"',
        'ATTENTION_BACKEND: Literal["sol-attn", "flash-attn3"] = ' + json.dumps(args.backend),
    )
    (package / "h3_scaling_profile.py").write_text(code)
    name = "h3-" + ("fa3" if args.backend == "flash-attn3" else "sol") + "-scaling-profile"
    project = (
        (SOURCE / "pyproject.toml")
        .read_text()
        .replace('name = "h3-scaling-profile"', f'name = "{name}"')
    )
    project = project.replace(
        'minimax-h3 = {path = "../../../minimax-h3"}',
        "minimax-h3 = {path = "
        + json.dumps(str(args.workflow.resolve(strict=True)))
        + "}\n"
        + "cozy-runtime = {path = "
        + json.dumps(str(wheel.resolve(strict=True)))
        + "}\n"
        + 'torch = {index = "pytorch-cu130"}\ntorchvision = {index = "pytorch-cu130"}',
    ).replace(
        "default-groups = []",
        'default-groups = []\nconstraint-dependencies = ["tensorfs==0.3.42", '
        '"torch==2.13.0+cu130", "torchvision==0.28.0+cu130"]',
    )
    project += (
        '\n[[tool.uv.index]]\nname = "pytorch-cu130"\n'
        'url = "https://download.pytorch.org/whl/cu130"\nexplicit = true\n'
    )
    (package / "pyproject.toml").write_text(project)
    commands = [
        ["uv", "sync", "--no-dev", "--no-default-groups", "--project", str(package)],
        [
            str(package / ".venv/bin/cozy-runtime"),
            "--json",
            "--dir",
            str(package),
            "--conformance",
            "describe",
        ],
        ["cozy", "package", "install", str(package), "--editable", "--no-model-download", "--json"],
    ]
    plan = {
        "state": "prepared_only_no_capture_install_or_gpu_run",
        "package": name,
        "runtime_source": args.expected_source,
        "runtime_wheel_sha256": args.expected_wheel_sha256,
        "runtime_manifest": str(args.runtime_manifest),
        "workflow": str(args.workflow),
        "backend": args.backend,
        "commands": commands,
        "source_sha256": {p.name: sha(p) for p in package.iterdir()},
        "instructions": (
            "Keep logs outside package; no GPU run until root reviews. "
            "A second SDK needs reviewed source/hash arguments and a fresh output directory."
        ),
    }
    (args.out / "capture-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "package": str(package),
                "runtime_source": args.expected_source,
                "backend": args.backend,
            }
        )
    )


if __name__ == "__main__":
    main()
