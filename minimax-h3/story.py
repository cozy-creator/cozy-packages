"""Shared reference assets and verbatim H3 segment prompt assembly."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

import msgspec
from msgspec.structs import replace
from cozy_runtime.author import AssetBound, AudioAsset, ImageAsset, InvalidRequest
from cozy_runtime.models.minimax_h3.official import validate_reference_policy

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
# H3 fills audio time no section describes with invented speech; both sections are required.
_AUDIO_HINTS = {
    "overall_soundscape": "describe the concrete non-voice sounds (room tone, footsteps, impacts, "
    "breathing); write N/A only for total silence",
    "non_diegetic_music": "describe the score, or write N/A for none",
}


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
    kind: Literal["character", "scene", "audio"]
    description: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    retention_analysis: Annotated[str, msgspec.Meta(max_length=1024)] | None = msgspec.field(
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
    audio: Annotated[
        AudioAsset | None,
        AssetBound(max_bytes=256 << 20, max_decoded_bytes=2 << 30),
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
) -> dict[str, ImageAsset | AudioAsset]:
    """Reuse granted handles and generate only missing images, draining failed siblings."""
    images: dict[str, ImageAsset | AudioAsset] = {}
    for ref in references:
        supplied = ref.audio if ref.kind == "audio" else ref.image
        if supplied is not None:
            images[ref.name] = supplied
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
    # The render children enforce this same policy after model preparation; refuse here first.
    try:
        validate_reference_policy(
            ["audio" if ref.kind == "audio" else "image" for ref in references]
        )
    except ValueError as exc:
        raise InvalidRequest(str(exc), code="reference_policy", fields=["references"]) from exc
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
    for field, hint in _AUDIO_HINTS.items():
        if not getattr(segment, field).strip():
            raise InvalidRequest(
                f"segment {index + 1}: {field} is blank; {hint}",
                code="h3_audio_section_required",
                fields=[field],
            )
    description = style + "\n" + segment.detailed_description if style else segment.detailed_description
    sections = [
        "subject_definitions:\n" + subject_definitions,
        "retention_analysis:\n" + retention_analysis,
        "detailed_description:\n" + description,
        "overall_soundscape:\n" + segment.overall_soundscape,
        "non_diegetic_music:\n" + segment.non_diegetic_music,
    ]
    if segment.summary.strip():
        sections.insert(1, "summary:\n" + segment.summary)
    prompt = "\n\n".join(sections)
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
    def combine_audio(*values: str) -> str:
        stated = [value.strip() for value in values if value.strip()]
        described = [value for value in stated if value.upper() != "N/A"]
        return "\n".join(described) if described else "N/A" if stated else ""

    return [
        segment_prompt(
            style,
            replace(
                segment,
                overall_soundscape=combine_audio(overall_soundscape, segment.overall_soundscape),
                non_diegetic_music=combine_audio(non_diegetic_music, segment.non_diegetic_music),
            ),
            subject_definitions=subject_definitions,
            retention_analysis=retention_analysis, index=index,
        )
        for index, segment in enumerate(segments)
    ]


def reference_sections(references: Sequence[StoryReference]) -> tuple[str, str]:
    """Use the same ordered image slots and literal names as the shared assets."""
    definitions, retention = [], []
    pictures = audios = 0
    for reference in references:
        if reference.kind == "audio":
            audios += 1
            label_kind = f"Audio {audios}"
        else:
            pictures += 1
            label_kind = f"Picture {pictures}"
        label = f"<{reference.name}>"
        prompt_label = f"<{label_kind}>"
        if reference.kind == "audio":
            definitions.append(
                f"{prompt_label} is the supplied audio reference for {label}."
                + (f" {reference.description}" if reference.description else "")
            )
        else:
            definitions.append(
                f"{label} is the {reference.kind} shown in {prompt_label}."
                + (f" {reference.description}" if reference.description else "")
            )
        preserved = (
            "the identity and defining visual features of the referenced character"
            if reference.kind == "character"
            else "the referenced audio signal and its supplied timing"
            if reference.kind == "audio"
            else "the architecture, materials and defining features of the referenced environment"
        )
        analysis = (
            f"fully_preserved - {preserved}."
            if reference.retention_analysis is None else reference.retention_analysis
        )
        retention.append(f"{prompt_label if reference.kind == 'audio' else label}: {analysis}")
    return "\n".join(definitions), "\n".join(retention)
