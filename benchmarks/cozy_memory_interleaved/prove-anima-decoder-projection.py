"""Read-only exact private-source projection; no app/model import or artifact build."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/fidika/.cozy/outputs/comfy-cozy-memory-20260929")
OUT = BASE / "analysis-anima-blend-private-prep-20260930"
TARGET = (
    ROOT
    / "benchmarks/cozy_memory_interleaved/cohorts/r20-anima-blend-decoder-probe-20260930/packages/anima"
)
PROJECTION = (
    ROOT
    / "benchmarks/cozy_memory_interleaved/cohorts/r20-anima-blend-private-projection-20260930/packages/anima"
)
REVIEWED = "50bb350abe9cae65fccc0aef60f2303ffe365aa1"
AUTHOR = "f2b66865eacb40a45d37cf73dbb2f1064f3414e3b52d6688ff7b0871e8ea0178"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    manifest = json.loads(
        (BASE / "analysis-startup-store-read-qualification/control-input-manifest.json").read_text()
    )
    frozen = manifest["packages"]["anima"]
    source = Path(frozen["path"])
    original_files = frozen["frozen_files_sha256"]
    expected_files = set(original_files) | {"anima/diagnostic_decode.py"}
    actual_files = {str(path.relative_to(TARGET)) for path in TARGET.rglob("*") if path.is_file()}
    assert actual_files == expected_files
    for relative, digest in original_files.items():
        assert sha((source / relative).read_bytes()) == digest, relative
    allowed = {
        "anima/__init__.py",
        "anima/diagnostic_decode.py",
        "package.toml",
        "pyproject.toml",
        "uv.lock",
        "metadata/package-interface.json",
    }
    changed = {
        relative
        for relative in original_files
        if (TARGET / relative).read_bytes() != (source / relative).read_bytes()
    } | {"anima/diagnostic_decode.py"}
    assert changed == allowed
    author = (TARGET / "anima/__init__.py").read_text()
    assert author == (PROJECTION / "anima/__init__.py").read_text()
    reviewed = subprocess.check_output(
        ["git", "show", f"{REVIEWED}:anima/anima/__init__.py"], cwd=ROOT, text=True
    )
    helper = next(
        node
        for node in ast.parse(reviewed).body
        if isinstance(node, ast.ClassDef) and node.name == "_AnimaVae"
    )
    helper_text = ast.get_source_segment(reviewed, helper)
    assert helper_text is not None
    insertion = helper_text + "\n\n\n"
    assert author.count(insertion) == 1
    inverse = (
        author.replace(insertion, "", 1)
        .replace(
            'vae = _AnimaVae.from_config(mapping["vae"]).to(torch.bfloat16)',
            'vae = AutoencoderKLQwenImage.from_config(mapping["vae"]).to(torch.bfloat16)',
            1,
        )
        .encode()
    )
    assert sha(inverse) == AUTHOR and inverse == (source / "anima/__init__.py").read_bytes()
    new_project = tomllib.loads((TARGET / "pyproject.toml").read_text())
    old_project = tomllib.loads((source / "pyproject.toml").read_text())
    expected_project = copy.deepcopy(old_project)
    name = new_project["project"]["name"]
    expected_project["project"]["name"] = name
    expected_project["project"]["entry-points"]["cozy.application"]["default"] = (
        "anima.diagnostic_decode:app"
    )
    assert new_project == expected_project
    assert tomllib.loads((TARGET / "package.toml").read_text()) == {
        "application": {"object": "anima.diagnostic_decode:app"}
    }
    old_lock = tomllib.loads((source / "uv.lock").read_text())
    new_lock = tomllib.loads((TARGET / "uv.lock").read_text())
    expected_lock = copy.deepcopy(old_lock)
    for record in expected_lock["package"]:
        if record["name"] == old_project["project"]["name"]:
            record["name"] = name
    assert new_lock == expected_lock
    wheel_rows = []
    for record in new_lock["package"]:
        path = record.get("source", {}).get("path", "")
        if path.endswith(".whl"):
            wheel = Path(path)
            (row,) = record["wheels"]
            assert row["filename"] == wheel.name and row["hash"] == "sha256:" + sha(
                wheel.read_bytes()
            )
            if "size" in row:
                assert row["size"] == wheel.stat().st_size
            wheel_rows.append({"name": record["name"], "source": record["source"], "wheel": row})
    actual_interface = (OUT / "static-diagnostic-interface.json").read_bytes()
    assert (
        TARGET / "metadata/package-interface.json"
    ).read_bytes().strip() == actual_interface.strip()
    interface = json.loads(actual_interface)
    (entry,) = interface["entrypoints"]
    assert not interface["jobs"] and entry["name"] == "generate"
    bounds = {
        field["name"]: field["asset_bound"]["max_bytes"]
        for field in entry["result"]["fields"]
        if "asset_bound" in field
    }
    assert len(bounds) == 8 and sum(bounds.values()) == 234 << 20
    result = {
        "status": "exact private source projection and pinned AST interface proven; no artifact or activation",
        "owner": "/root/decoder_diagnostic_completion",
        "worktree_base_sha": "5001dcb42ca5d6c7731e6bbd86b70fc87144e67f",
        "source_package": str(source),
        "diagnostic_package": str(TARGET),
        "copied_qualified_content_sha256": original_files,
        "reviewed_helper_commit": REVIEWED,
        "inverse_author_sha256": sha(inverse),
        "exact_six_changed_paths": sorted(changed),
        "all_other_qualified_bytes_exact": True,
        "lock_objects_exact_except_private_root_name": True,
        "project_objects_exact_except_private_identity_and_app": True,
        "direct_locked_wheel_rows": wheel_rows,
        "runtime_sha256": sha(Path(manifest["runtime_wheel"]).read_bytes()),
        "agent_sha256": sha(Path(manifest["agent"]).read_bytes()),
        "tensorfs_sha256": sha(Path(manifest["tensorfs_wheel"]).read_bytes()),
        "diagnostic_files_sha256": {
            relative: sha((TARGET / relative).read_bytes()) for relative in sorted(expected_files)
        },
        "interface_sha256": sha(actual_interface),
        "entrypoint_count": 1,
        "generated_descriptor_bound_bytes": bounds,
        "generated_descriptor_total_bytes": sum(bounds.values()),
        "base_encoded_leaves_existing_and_inherited": entry["models"][0]["encoded_leaves"]
        == "accept",
        "identity_fields_are_reproduction_facts_not_peer_admission_gates": True,
        "model_imported": False,
        "conformance_oracle_run": False,
        "artifact_built": False,
        "registered": False,
        "installed": False,
        "gpu_executed": False,
    }
    assert (
        result["runtime_sha256"]
        == "6660f87c105bf5de504f9e783a0019164d1f35f12d7fa162c0a3f9081e23e073"
    )
    (OUT / "DIAGNOSTIC-SOURCE-FACTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "six_changed_paths": sorted(changed),
                "total_MiB": sum(bounds.values()) >> 20,
                "status": result["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
