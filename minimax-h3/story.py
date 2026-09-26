"""Shared reference assets and verbatim H3 segment prompt assembly."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import AssetBound, AudioAsset, ImageAsset, InvalidRequest

MAX_IMAGES = 9
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")


class StorySegment(msgspec.Struct, forbid_unknown_fields=True):
    """One invocation with ordinary H3 text, including any camera-shot markers."""

    summary: Annotated[str, msgspec.Meta(min_length=1, max_length=1024)]
    detailed_description: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    overall_soundscape: Annotated[str, msgspec.Meta(max_length=1024)]
    non_diegetic_music: Annotated[str, msgspec.Meta(max_length=1024)]
    seed: int | None = None
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)] = 10


class StoryReference(msgspec.Struct, forbid_unknown_fields=True):
    """A supplied image or an image generated from its shared visual description."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    kind: Literal["character", "scene", "audio"]
    description: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    image: Annotated[
        ImageAsset | None,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ] = None
    audio: Annotated[AudioAsset | None, AssetBound(max_bytes=256 << 20)] = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None


def reference_seed(reference: StoryReference, request_id: str) -> int:
    if reference.seed is not None:
        return reference.seed
    identity = f"{request_id}/h3/reference/{reference.name.casefold()}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


async def resolve_reference_images(
    references: Sequence[StoryReference],
    generate: Callable[[StoryReference], Awaitable[ImageAsset]],
) -> dict[str, ImageAsset | AudioAsset]:
    """Reuse granted handles and generate only missing images, draining failed siblings."""
    images: dict[str, ImageAsset | AudioAsset] = {
        ref.name: (ref.audio if ref.kind == "audio" else ref.image)
        for ref in references
        if ref.audio is not None or ref.image is not None
    }
    pending = [ref for ref in references if ref.kind != "audio" and ref.image is None]

    async def one(reference: StoryReference) -> ImageAsset:
        return await generate(reference)

    tasks = [asyncio.create_task(one(reference)) for reference in pending]
    try:
        generated = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    images.update((ref.name, image) for ref, image in zip(pending, generated, strict=True))
    return images


def image_prompt(reference: StoryReference) -> str:
    if reference.kind == "character":
        return (
            f"{reference.description.strip()}\n"
            "A full-body subject reference with clear identifying features, a natural "
            "neutral pose, and the entire subject visible against a plain white studio "
            "background. One subject, one view, no panels or labels."
        )
    return (
        f"{reference.description.strip()}\n"
        "An environment reference image showing the architecture, materials and defining "
        "features of the location, without people, labels or panels."
    )


def validate_references(references: Sequence[StoryReference]) -> dict[str, StoryReference]:
    if not 1 <= len(references) <= MAX_IMAGES:
        raise InvalidRequest("declare between one and nine references", fields=["references"])
    by_name: dict[str, StoryReference] = {}
    for reference in references:
        if not _NAME.fullmatch(reference.name):
            raise InvalidRequest(
                "reference names must start with a letter and use letters, digits, _ or -",
                fields=["references"],
            )
        key = reference.name.casefold()
        if key in by_name:
            raise InvalidRequest(
                f"duplicate reference name: {reference.name}", fields=["references"]
            )
        if reference.kind == "audio" and (reference.audio is None or reference.image is not None):
            raise InvalidRequest(
                f"audio reference {reference.name} requires audio and forbids image",
                fields=["references"],
            )
        if reference.kind != "audio" and reference.audio is not None:
            raise InvalidRequest(
                f"{reference.kind} reference {reference.name} cannot carry audio",
                fields=["references"],
            )
        if reference.kind != "audio" and reference.image is None and not reference.description.strip():
            raise InvalidRequest(
                f"reference {reference.name} needs an image or a nonempty description",
                fields=["references"],
            )
        by_name[key] = reference
    return by_name


def segment_prompt(
    style: str,
    segment: StorySegment,
    *,
    subject_definitions: str,
    retention_analysis: str,
    index: int,
) -> str:
    """Add section headings and shared style; never interpret or rewrite authored text."""
    for field, value in (
        ("subject_definitions", subject_definitions),
        ("retention_analysis", retention_analysis),
        ("summary", segment.summary),
        ("detailed_description", segment.detailed_description),
    ):
        if not value.strip():
            raise InvalidRequest(
                f"segment {index + 1}: {field} must not be blank", fields=[field]
            )
    description = style + "\n" + segment.detailed_description if style else segment.detailed_description
    prompt = "\n\n".join(
        (
            "subject_definitions:\n" + subject_definitions,
            "summary:\n" + segment.summary,
            "retention_analysis:\n" + retention_analysis,
            "detailed_description:\n" + description,
            "overall_soundscape:\n" + segment.overall_soundscape,
            "non_diegetic_music:\n" + segment.non_diegetic_music,
        )
    )
    if len(prompt) > 4096:
        raise InvalidRequest(
            f"segment {index + 1}'s complete prompt exceeds 4096 characters",
            fields=["segments", "style", "subject_definitions", "retention_analysis"],
        )
    return prompt


def compile_segments(
    style: str,
    segments: Sequence[StorySegment],
    references: Sequence[StoryReference],
    *,
    subject_definitions: str,
    retention_analysis: str,
) -> list[str]:
    """Validate before generation, with fixed assets and shared authored definitions."""
    validate_references(references)
    return [
        segment_prompt(
            style, segment, subject_definitions=subject_definitions,
            retention_analysis=retention_analysis, index=index,
        )
        for index, segment in enumerate(segments)
    ]
