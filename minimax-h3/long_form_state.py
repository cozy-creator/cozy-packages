"""Exact code, model, and software provenance shared by the shots in one rendering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from pathlib import Path, PurePosixPath

import msgspec
from cozy_runtime.author import InvalidRequest

SOFTWARE = (
    "cozy-runtime",
    "tensorfs",
    "torch",
    "diffusers",
    "transformers",
    "numpy",
    "pillow",
    "av",
)


class SoftwareVersion(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    version: str


class RenderProvenance(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    model_manifest: str
    code_digest: str
    software: list[SoftwareVersion]
    turbo_lora_manifest: str = ""


def code_digest() -> str:
    """Hash installed workflow and builtin inference payloads, excluding bookkeeping."""
    distribution = importlib.metadata.distribution("minimax-h3")
    digest = hashlib.sha256()
    files = distribution.files
    if files is None:
        raise InvalidRequest("H3 package has no installed source inventory", code="render_code")
    names = {str(member) for member in files}
    if not {"h3.py", "long_form_state.py"}.issubset(names):
        raise InvalidRequest(
            "render provenance needs the captured H3 wheel inventory", code="render_code"
        )
    count = 0
    total = 0
    for member in sorted(files, key=str):
        path = PurePosixPath(str(member))
        if any(part.endswith(".dist-info") for part in path.parts) or "__pycache__" in path.parts:
            continue
        if path.is_absolute() or ".." in path.parts:
            raise InvalidRequest("H3 package source inventory is not relative", code="render_code")
        source = Path(__file__).resolve().parent.joinpath(path)
        if not source.is_file():
            raise InvalidRequest("H3 package source file is unavailable", code="render_code")
        size = source.stat().st_size
        total += size
        if total > 128 << 20:
            raise InvalidRequest(
                "H3 package source inventory exceeds its bound", code="render_code"
            )
        name = str(path).encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with source.open("rb") as stream:
            while block := stream.read(1 << 20):
                digest.update(block)
        count += 1
    if count == 0:
        raise InvalidRequest("H3 package source inventory is empty", code="render_code")
    # The Runtime version can stay constant during content-addressed development.
    # Record the builtin's actual installed code/assets as well as its version.
    runtime = importlib.metadata.distribution("cozy-runtime")
    builtin_prefix = "cozy_runtime/models/minimax_h3/"
    builtin_files = [
        member
        for member in runtime.files or ()
        if str(member).startswith(builtin_prefix)
        and "__pycache__" not in PurePosixPath(str(member)).parts
    ]
    if not any(str(member) == builtin_prefix + "model.py" for member in builtin_files):
        raise InvalidRequest("H3 builtin source inventory is unavailable", code="render_code")
    for member in sorted(builtin_files, key=str):
        path = PurePosixPath(str(member))
        if path.is_absolute() or ".." in path.parts:
            raise InvalidRequest("H3 builtin source inventory is not relative", code="render_code")
        source = Path(str(runtime.locate_file(member)))
        if not source.is_file():
            raise InvalidRequest("H3 builtin source file is unavailable", code="render_code")
        size = source.stat().st_size
        total += size
        if total > 128 << 20:
            raise InvalidRequest("H3 source inventory exceeds its bound", code="render_code")
        name = ("cozy-runtime/" + str(path)).encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with source.open("rb") as stream:
            while block := stream.read(1 << 20):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def provenance(model_manifest: str, turbo_lora_manifest: str = "") -> RenderProvenance:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", model_manifest):
        raise InvalidRequest("H3 rendering needs its exact model manifest", code="render_model")
    if turbo_lora_manifest and not re.fullmatch(r"sha256:[0-9a-f]{64}", turbo_lora_manifest):
        raise InvalidRequest("H3 turbo needs its exact adapter manifest", code="render_model")
    return RenderProvenance(
        model_manifest,
        code_digest(),
        [SoftwareVersion(name, importlib.metadata.version(name)) for name in SOFTWARE],
        turbo_lora_manifest,
    )


def compatible(actual: RenderProvenance, expected: RenderProvenance | None) -> None:
    if expected is not None and actual != expected:
        raise InvalidRequest(
            "shots in one rendering must use the same base and adapter bytes, code, and software",
            code="render_provenance",
        )
