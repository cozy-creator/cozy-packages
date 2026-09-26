"""Shared reference assets and verbatim H3 segment prompt assembly."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

import msgspec
from msgspec.structs import replace
from cozy_runtime.author import AssetBound, ImageAsset, InvalidRequest

MAX_IMAGES = 9
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")


class StorySegment(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """One invocation with ordinary H3 text, including any camera-shot markers."""

    summary: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    detailed_description: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    overall_soundscape: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    non_diegetic_music: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    seed: int | None = None
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)]


class StoryReference(msgspec.Struct, forbid_unknown_fields=True):
    """A supplied image or an image generated from its shared visual description."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    kind: Literal["character", "scene"]
    description: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    retention_analysis: Annotated[str | None, msgspec.Meta(max_length=1024)] = msgspec.field(
        default=None, name="retention-analysis"
    )
    image: Annotated[
        ImageAsset | None,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ] = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None


def reference_seed(reference: StoryReference, request_id: str) -> int:
    if reference.seed is not None:
        return reference.seed
    identity = f"{request_id}/h3/reference/{reference.name.casefold()}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


async def resolve_reference_images(
    references: Sequence[StoryReference],
    generate: Callable[[StoryReference], Awaitable[ImageAsset]],
) -> dict[str, ImageAsset]:
    """Reuse granted handles and generate only missing images, draining failed siblings."""
    images = {ref.name: ref.image for ref in references if ref.image is not None}
    pending = [ref for ref in references if ref.image is None]

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
        if reference.image is None and not reference.description.strip():
            raise InvalidRequest(
                f"reference {reference.name} needs an image or a nonempty description",
                fields=["references"],
            )
        if reference.retention_analysis is not None and not reference.retention_analysis.strip():
            raise InvalidRequest(
                f"reference {reference.name} has blank retention-analysis; omit it for the default",
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
    overall_soundscape: str = "",
    non_diegetic_music: str = "",
) -> list[str]:
    """Generate shared reference sections once; preserve all authored segment text."""
    validate_references(references)
    subject_definitions, retention_analysis = reference_sections(references)
    return [
        segment_prompt(
            style,
            replace(
                segment,
                overall_soundscape="\n".join(
                    value for value in (overall_soundscape, segment.overall_soundscape) if value
                ),
                non_diegetic_music="\n".join(
                    value for value in (non_diegetic_music, segment.non_diegetic_music) if value
                ),
            ),
            subject_definitions=subject_definitions,
            retention_analysis=retention_analysis, index=index,
        )
        for index, segment in enumerate(segments)
    ]


def reference_sections(references: Sequence[StoryReference]) -> tuple[str, str]:
    """Use the same ordered image slots and literal names as the shared assets."""
    definitions, retention = [], []
    for slot, reference in enumerate(references, 1):
        label = f"<{reference.name}>"
        definitions.append(
            f"{label} is the {reference.kind} shown in <Picture {slot}>."
            + (f" {reference.description}" if reference.description else "")
        )
        preserved = (
            "the identity and defining visual features of the referenced character"
            if reference.kind == "character"
            else "the architecture, materials and defining features of the referenced environment"
        )
        analysis = (
            f"fully_preserved - {preserved}."
            if reference.retention_analysis is None else reference.retention_analysis
        )
        retention.append(f"{label}: {analysis}")
    return "\n".join(definitions), "\n".join(retention)
