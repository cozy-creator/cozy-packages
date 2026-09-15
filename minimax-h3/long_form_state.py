"""Native long-form prefix bytes and the provenance of the shots they actually contain."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

import msgspec
from cozy_runtime.author import (
    AssetBound,
    Context,
    FileAsset,
    ImageAsset,
    InvalidRequest,
    Outputs,
    Tree,
    VideoAsset,
)

from assembly import MAX_SHOTS

MAX_PREFIX_BYTES = 256 << 20
MAX_MANIFEST_BYTES = 256 << 10
VIDEO_BOUND = AssetBound(
    max_bytes=MAX_PREFIX_BYTES, max_decoded_bytes=32 << 20, media_types=("video/mp4",)
)
FRAME_BOUND = AssetBound(max_bytes=64 << 20, max_decoded_bytes=64 << 20, media_types=("image/png",))
# JSON has no byte signature; the bounded FileAsset is validated by PrefixManifest decoding.
MANIFEST_BOUND = AssetBound(max_bytes=MAX_MANIFEST_BYTES)
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


class ShotIntent(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    prompt: str
    seed: int
    duration_s: int
    steps: int
    first_frame_digest: str


class StoredShot(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    intent: ShotIntent
    provenance: RenderProvenance
    child_request_id: str
    frames: int
    video_digest: str
    video_bytes: int
    continuation_frame_digest: str
    continuation_frame_bytes: int


class PrefixManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    format: Literal["minimax-h3.long-form-prefix/1"]
    producer_request_id: str
    shots: list[StoredShot]


def code_digest() -> str:
    """Hash installed workflow and builtin inference payloads, excluding bookkeeping."""
    distribution = importlib.metadata.distribution("minimax-h3")
    digest = hashlib.sha256()
    files = distribution.files
    if files is None:
        raise InvalidRequest("H3 package has no installed source inventory", code="prefix_code")
    names = {str(member) for member in files}
    if not {"h3.py", "long_form_state.py"}.issubset(names):
        raise InvalidRequest(
            "prefix provenance needs the captured H3 wheel inventory", code="prefix_code"
        )
    count = 0
    total = 0
    for member in sorted(files, key=str):
        path = PurePosixPath(str(member))
        if any(part.endswith(".dist-info") for part in path.parts) or "__pycache__" in path.parts:
            continue
        if path.is_absolute() or ".." in path.parts:
            raise InvalidRequest("H3 package source inventory is not relative", code="prefix_code")
        source = Path(__file__).resolve().parent.joinpath(path)
        if not source.is_file():
            raise InvalidRequest("H3 package source file is unavailable", code="prefix_code")
        size = source.stat().st_size
        total += size
        if total > 128 << 20:
            raise InvalidRequest(
                "H3 package source inventory exceeds its bound", code="prefix_code"
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
        raise InvalidRequest("H3 package source inventory is empty", code="prefix_code")
    # The Runtime version can stay constant during content-addressed development.
    # Record the builtin's actual installed code/assets as well as its version.
    runtime = importlib.metadata.distribution("cozy-runtime")
    builtin_prefix = "cozy_runtime/models/minimax_h3/"
    builtin_files = [
        member for member in runtime.files or ()
        if str(member).startswith(builtin_prefix)
        and "__pycache__" not in PurePosixPath(str(member)).parts
    ]
    if not any(str(member) == builtin_prefix + "model.py" for member in builtin_files):
        raise InvalidRequest("H3 builtin source inventory is unavailable", code="prefix_code")
    for member in sorted(builtin_files, key=str):
        path = PurePosixPath(str(member))
        if path.is_absolute() or ".." in path.parts:
            raise InvalidRequest("H3 builtin source inventory is not relative", code="prefix_code")
        source = Path(str(runtime.locate_file(member)))
        if not source.is_file():
            raise InvalidRequest("H3 builtin source file is unavailable", code="prefix_code")
        size = source.stat().st_size
        total += size
        if total > 128 << 20:
            raise InvalidRequest("H3 source inventory exceeds its bound", code="prefix_code")
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
        raise InvalidRequest("H3 rendering needs its exact model manifest", code="prefix_model")
    if turbo_lora_manifest and not re.fullmatch(r"sha256:[0-9a-f]{64}", turbo_lora_manifest):
        raise InvalidRequest("H3 turbo needs its exact adapter manifest", code="prefix_model")
    return RenderProvenance(
        model_manifest,
        code_digest(),
        [SoftwareVersion(name, importlib.metadata.version(name)) for name in SOFTWARE],
        turbo_lora_manifest,
    )


def compatible(actual: RenderProvenance, expected: RenderProvenance | None) -> None:
    if expected is not None and actual != expected:
        raise InvalidRequest(
            "retained shots use different base or adapter bytes, code or rendering software; "
            "reuse their recorded source cohort to extend this prefix",
            code="prefix_provenance",
            fields=["resume_from"],
        )


def read_prefix(
    tree: Tree, *, requested: int
) -> tuple[PrefixManifest, list[VideoAsset], list[ImageAsset]]:
    member = tree.member("manifest.json", FileAsset, bound=MANIFEST_BOUND)
    try:
        manifest = msgspec.json.decode(member.read_bytes(), type=PrefixManifest)
    except (msgspec.ValidationError, msgspec.DecodeError) as exc:
        raise InvalidRequest("retained prefix manifest is invalid", code="prefix_manifest") from exc
    if not 1 <= len(manifest.shots) <= requested <= MAX_SHOTS:
        raise InvalidRequest(
            "retained prefix exceeds the requested shot list", code="prefix_length"
        )
    videos: list[VideoAsset] = []
    frames: list[ImageAsset] = []
    previous = ""
    expected = manifest.shots[0].provenance
    for index, shot in enumerate(manifest.shots):
        compatible(shot.provenance, expected)
        if index and shot.intent.first_frame_digest != previous:
            raise InvalidRequest("retained prefix handoff is inconsistent", code="prefix_handoff")
        video = tree.member(f"shots/{index:06d}/video.mp4", VideoAsset, bound=VIDEO_BOUND)
        frame = tree.member(f"shots/{index:06d}/continuation.png", ImageAsset, bound=FRAME_BOUND)
        if (video.digest, video.size_bytes, frame.digest, frame.size_bytes) != (
            shot.video_digest,
            shot.video_bytes,
            shot.continuation_frame_digest,
            shot.continuation_frame_bytes,
        ):
            raise InvalidRequest(
                "retained prefix bytes differ from its manifest", code="prefix_bytes"
            )
        if shot.frames < 1 or not shot.child_request_id:
            raise InvalidRequest("retained prefix omitted shot provenance", code="prefix_manifest")
        videos.append(video)
        frames.append(frame)
        previous = frame.digest
    return manifest, videos, frames


def check_intent(record: StoredShot, expected: ShotIntent, frames: int) -> None:
    if record.intent != expected or record.frames != frames:
        raise InvalidRequest(
            "a retained shot's prompt, seed, duration, steps or opening frame changed; "
            "only the remaining shot prompts may change",
            code="prefix_intent",
            fields=["shots", "resume_from"],
        )


def save_prefix(
    ctx: Context,
    out: Outputs,
    records: list[StoredShot],
    videos: Sequence[VideoAsset],
    frames: Sequence[ImageAsset],
) -> Tree:
    """Copy exact encoded members into one declared native output; never re-encode a shot."""
    raw = msgspec.json.encode(
        PrefixManifest("minimax-h3.long-form-prefix/1", ctx.request_id, records)
    )
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InvalidRequest("retained prefix metadata exceeds its bound", code="prefix_manifest")
    if len(raw) + sum(item.size_bytes for item in (*videos, *frames)) > MAX_PREFIX_BYTES:
        raise InvalidRequest("retained prefix exceeds its declared byte bound", code="prefix_bytes")
    directory = out.temporary_file(".prefix")
    directory.mkdir()
    (directory / "manifest.json").write_bytes(raw)
    for index, (video, frame) in enumerate(zip(videos, frames, strict=True)):
        ctx.raise_if_cancelled()
        shot = directory / "shots" / f"{index:06d}"
        shot.mkdir(parents=True)
        (shot / "video.mp4").write_bytes(video.read_bytes())
        (shot / "continuation.png").write_bytes(frame.read_bytes())
    return out.save_tree(directory)
