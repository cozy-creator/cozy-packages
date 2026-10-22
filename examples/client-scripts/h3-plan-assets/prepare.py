# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30,<0.31", "tensorfs>=0.13,<0.14", "cbor2>=5.6"]
# [tool.cozy]
# accelerator = false
# ///
"""Prepare a metadata-only H3 migration; never upload or move a release.

Run with cozy on the local machine. source_revision must be a pinned tree digest.
plan_dir contains the component directories emitted by scripts/h3_plans.py.
The receipt is outside the proposed upload folder, which retains every source repo file.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import cbor2
import tensorfs
from cozy_runtime.author import FileAsset

PLAN = "timestep-plan.json"
EXPECTED = {
    "transformer": ("fl2va", "9a48803d17d7bb5499ca8c018f60c86c9890eb91496249ac7cb199bc9e45201e"),
    "transformer_ref": ("ref2va", "565a164cbf0cefa58d4976cb4c84887cf9807cc7c4be62263d5e0e0c5d9bc50e"),
    "fl2va_turbo": ("fl2va", "d2eb1605c1c1e01a4c5fdaaf1912ab43f33d9ce4772febf1c75b9bbfc8f7ba9d"),
    "ref2va_turbo": ("ref2va", "896a10881805e3047f6df514dd4e70fcea078837469eeb22ce67f0bd3f679cb7"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): _sha(p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).parts[0] != "chunks"
    }


def _header(raw: bytes) -> list[Any]:
    stream = io.BytesIO(raw)
    doc = cbor2.load(stream)
    if stream.tell() != len(raw) or not isinstance(doc, list) or len(doc) < 6:
        raise ValueError("expected one complete CozyTensors header")
    if doc[0] != "cozytensors/1":
        raise ValueError("migration supports only cozytensors/1")
    if cbor2.dumps(doc, canonical=True) != raw:
        raise ValueError("source header must roundtrip byte-for-byte through canonical CBOR")
    return doc


def _prepare(
    *, source_repo: str, source_revision: str, endpoint: str,
    plan_dir: str, work_dir: str, turbo: bool,
) -> str:
    """Return the receipt path after native metadata validation; do not publish anything."""
    if (len(source_revision) != 71 or not source_revision.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in source_revision[7:])):
        raise ValueError("source_revision must pin the complete source tree by SHA256")
    names = ("fl2va_turbo", "ref2va_turbo") if turbo else ("transformer", "transformer_ref")
    plans = {}
    for name in names:
        raw = (Path(plan_dir) / name / PLAN).read_bytes()
        task, digest = EXPECTED[name]
        document = json.loads(raw)
        if _sha(raw) != digest or document["task"] != task:
            raise ValueError(f"{name}: plan differs from the historical approved release")
        plans[name] = (raw, document)

    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=False)
    source, proposed = work / "source", work / "proposed"
    # Empty selection downloads all variant headers and repository files, no tensor or
    # component-asset objects. No cache/home override or private credentials are needed.
    tensorfs.materialize(source_repo, revision=source_revision, endpoint=endpoint,
                         local_dir=source, components=[])
    if any((source / "chunks").iterdir()):
        raise ValueError("metadata-only snapshot unexpectedly contains component payload")
    original_files = _files(source)
    headers = sorted(p for p in original_files if p.endswith(".cozytensors"))
    if not headers:
        raise ValueError("source tree contains no CozyTensors variants")
    shutil.copytree(source, proposed)
    receipts = []
    for path in headers:
        before = _header((source / path).read_bytes())
        after = copy.deepcopy(before)
        components = {c[0]: c for c in after[4]}
        if not set(names) <= set(components):
            raise ValueError(f"{path}: missing target components")
        added = []
        for name in names:
            raw, plan = plans[name]
            component = components[name]
            stamp = component[2].get("cozy_h3", {})
            if "generating_projection_digest" in stamp:
                raise ValueError(f"{path}/{name}: legacy projection binding needs its own migration")
            if stamp.get("modulation") == "adaln-pruned" and stamp.get("table_keys") != plan["table_keys"]:
                raise ValueError(f"{path}/{name}: plan rows differ from existing table metadata")
            reference = [bytes.fromhex(_sha(raw)), len(raw)]
            existing = component[4].get(PLAN)
            if existing is not None and existing != reference:
                raise ValueError(f"{path}/{name}: refusing to replace another plan asset")
            if existing is None:
                component[4][PLAN] = reference
                added.append(name)
            (proposed / "chunks" / _sha(raw)).write_bytes(raw)
        restored = copy.deepcopy(after)
        for component in restored[4]:
            if component[0] in added:
                del component[4][PLAN]
        if restored != before:
            raise ValueError("migration altered data beyond the new plan asset references")
        encoded = cbor2.dumps(after, canonical=True)
        target = proposed / path
        target.write_bytes(encoded)
        with tensorfs.open(target, local_files_only=True, lazy=True,
                           endpoint="https://no-payload.invalid") as checkpoint:
            if checkpoint.id.removeprefix("sha256:") != _sha(encoded):
                raise ValueError("native header identity differs")
            expected_chunks = [(digest.hex(), length) for digest, length in before[1]]
            actual_chunks = [(digest.removeprefix("sha256:"), length)
                             for digest, length in checkpoint.chunks]
            if actual_chunks != expected_chunks:
                raise ValueError("native reader changed tensor chunk references")
            for name in names:
                if checkpoint.components[name].asset(PLAN) != plans[name][0]:
                    raise ValueError("native plan-asset readback differs")
            with checkpoint.open_file() as stream:
                if stream.read(len(encoded)) != encoded:
                    raise ValueError("native header roundtrip differs")
        receipts.append({"path": path, "old_header": original_files[path],
                         "new_header": _sha(encoded), "added_to": added,
                         "tensor_chunks_unchanged": len(before[1]),
                         "tensor_rows_unchanged": sum(len(c[3]) for c in before[4])})
    files = _files(proposed)
    if set(files) != set(original_files):
        raise ValueError("repository file paths changed")
    if any(files[p] != digest for p, digest in original_files.items() if p not in headers):
        raise ValueError("repository files outside the headers changed")
    receipt = {
        "source_repo": source_repo, "source_revision": source_revision, "endpoint": endpoint,
        "tensorfs": tensorfs.__version__, "proposed_folder": str(proposed),
        "source_files": original_files, "proposed_files": files, "headers": receipts,
        "plan_assets": {name: _sha(raw) for name, (raw, _) in plans.items()},
        "payload_scope": "only new plan assets; no source component payload requested",
        "published": False,
    }
    destination = work / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    return str(destination)


def main(*, request: FileAsset) -> str:
    """Read the explicit migration request from a granted JSON file."""
    document = json.loads(request.read_bytes())
    expected = {"source_repo", "source_revision", "endpoint", "plan_dir", "work_dir", "turbo"}
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError(f"request must contain exactly {sorted(expected)}")
    if type(document["turbo"]) is not bool or any(
        not isinstance(value, str) for key, value in document.items() if key != "turbo"
    ):
        raise ValueError("request paths and references must be strings, turbo must be a boolean")
    return _prepare(**document)
