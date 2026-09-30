"""Shared reference assets and H3 segment prompts: authored text plus the global sections."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from itertools import pairwise
from typing import Annotated, Literal, NamedTuple

import msgspec
from cozy_runtime.author import AssetBound, AudioAsset, ImageAsset, InvalidRequest
from cozy_runtime.models.minimax_h3.official import Task, validate_reference_policy

#: MiniMax's full-reference sections, plus `style`, in their canonical order.
SECTIONS = (
    "subject_definitions",
    "summary",
    "style",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
# A header starts a line; any case, and `_`, `-` and space are the same separator.
_HEADER = re.compile(
    r"^[ \t]*(" + "|".join(name.replace("_", "[ _-]+") for name in SECTIONS) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_TOKEN = re.compile(r"<([^<>\n]+)>")
_MEDIA = re.compile(r"\s*(picture|audio|video)\s+(\d+)\s*", re.IGNORECASE)
_SHOT = re.compile(r"\[Shot\s*(\d+)\]", re.IGNORECASE)


class StoryReference(msgspec.Struct):
    """A supplied image or an image generated from its shared visual description."""

    name: str
    kind: Literal["character", "scene", "audio"]
    description: str = ""
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

    def __post_init__(self) -> None:
        # Prompts quote the name as <name>: any text reads as a name once its brackets and
        # extra whitespace go.
        self.name = " ".join(self.name.replace("<", " ").replace(">", " ").split())
        if not self.name:
            raise ValueError("a reference name needs at least one visible character")


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


def image_prompt(reference: StoryReference, style: str = "") -> str:
    parts = [reference.description.strip()]
    if style.strip():
        parts.append(f"style:\n{style.strip()}")
    if reference.kind == "character":
        parts.append(
            "Generate exactly one image: a character design sheet on a plain white background. "
            "Within this single image, place two full-body views of the same character side "
            "by side: the front view on the left and the back view on the right. Show the "
            "entire character in each view, in a natural neutral pose. Keep the character's "
            "appearance, proportions and clothing consistent across both views."
        )
    else:
        parts.append(
            "An environment reference image showing the architecture, materials and defining "
            "features of the location, without people, labels or panels."
        )
    parts.append(
        "Follow the visual style explicitly requested in the description or style section. "
        "If no visual style is specified, use a realistic photographic look. "
        "Use anime styling only when explicitly requested."
    )
    return "\n\n".join(parts)


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
        key = name_key(reference.name)
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


def encoder_prompt(task: Task, prompt: str) -> str:
    """The text H3's encoder receives: the prompt verbatim, plus `N/A` music unless it names it.

    H3 invents score and speech for audio no section describes. The line takes the task's
    upstream layout: fl2va's single-line fields, ref2va's heading-over-value sections.
    """
    if "non_diegetic_music" in prompt.casefold() or "non_diegetic_music" in _sections(prompt):
        return prompt
    line = "non_diegetic_music: N/A" if task.startswith("fl2va") else "non_diegetic_music:\nN/A"
    return "\n\n".join(filter(None, (prompt.rstrip(), line)))


class SegmentCall(NamedTuple):
    prompt: str
    references: list[StoryReference]
    warnings: list[str]


def name_key(name: str) -> str:
    """How a `<Name>` token matches a reference: any case, `_`, `-` and space alike."""
    return " ".join(re.sub(r"[_-]", " ", name).split()).casefold()


def compile_segments(
    prompts: Sequence[str],
    references: Sequence[StoryReference],
    *,
    style: str = "",
    overall_soundscape: str = "",
    non_diegetic_music: str = "N/A",
    segment_music: Sequence[str | None] | None = None,
) -> list[SegmentCall]:
    """Each child prompt uses its music override, inline score, or overall score."""
    validate_references(references)
    shared = {
        "style": style.strip(),
        "overall_soundscape": overall_soundscape.strip(),
    }
    overrides = [None] * len(prompts) if segment_music is None else segment_music
    calls = []
    for index, (text, override) in enumerate(zip(prompts, overrides, strict=True)):
        music = non_diegetic_music if override is None else override
        text = _fill(
            text, {"non_diegetic_music": music.strip() or "N/A"},
            replace_music=override is not None,
        )
        calls.append(_segment_call(text, references, shared, index))
    return calls


def _slot(ref: StoryReference) -> str:
    return "audio" if ref.kind == "audio" else "picture"


def _numbering(references: Sequence[StoryReference]) -> dict[tuple[str, int], StoryReference]:
    """`<Picture N>` and `<Audio N>`: each kind counted in the given order."""
    counts = {"picture": 0, "audio": 0}
    slots: dict[tuple[str, int], StoryReference] = {}
    for ref in references:
        counts[_slot(ref)] += 1
        slots[_slot(ref), counts[_slot(ref)]] = ref
    return slots


def _segment_call(
    text: str,
    references: Sequence[StoryReference],
    shared: dict[str, str],
    index: int,
) -> SegmentCall:
    """Scenes always; characters, and audio, only where the text names them.

    A voice (audio whose description names a character) follows its character. Global
    `<Picture N>`/`<Audio N>` tokens count as mentions and are renumbered for the child.
    """
    by_key = {name_key(ref.name): ref for ref in references}
    by_slot = _numbering(references)

    def lookup(token: re.Match[str]) -> StoryReference | None:
        media = _MEDIA.fullmatch(token.group(1))
        if media is None:
            return by_key.get(name_key(token.group(1)))
        return by_slot.get((media.group(1).lower(), int(media.group(2))))

    def mentioned(span: str) -> list[StoryReference]:
        found = {ref.name: ref for token in _TOKEN.finditer(span) if (ref := lookup(token))}
        return list(found.values())

    unmatched = dict.fromkeys(
        token.group(0)
        for token in _TOKEN.finditer(text)
        if _MEDIA.fullmatch(token.group(1)) and lookup(token) is None
    )
    warnings = (
        [f"Segment {index + 1}: {', '.join(unmatched)} match no reference; left as written."]
        if unmatched else []
    )
    voices: dict[str, list[StoryReference]] = {}
    for ref in references:
        owner = [o for o in mentioned(ref.description) if o.kind != "audio"]
        if ref.kind == "audio" and owner:
            voices.setdefault(owner[0].name, []).append(ref)
    attached = {ref.name: ref for ref in references if ref.kind == "scene"}
    for ref in mentioned(text):
        attached.setdefault(ref.name, ref)
        for voice in voices.get(ref.name, []):
            attached.setdefault(voice.name, voice)
    used = list(attached.values())
    if all(ref.kind == "audio" for ref in used):
        # H3's ref2va needs an image: with no scene and no character named, send them all.
        used = list(references)
        warnings.append(
            f"Segment {index + 1} names no character and there is no scene, so it receives "
            f"all {len(used)} references."
        )
    _, start, end = _sections(text).get("detailed_description", (0, 0, 0))
    markers = list(_SHOT.finditer(text, start, end))
    spans = pairwise([marker.start() for marker in markers] + [end])
    shots = [
        (m.group(1), {ref.name for ref in mentioned(text[a:b])})
        for m, (a, b) in zip(markers, spans, strict=True)
    ]
    child = {ref.name: slot for slot, ref in _numbering(used).items()}

    def renumber(token: re.Match[str]) -> str:
        ref = lookup(token) if _MEDIA.fullmatch(token.group(1)) else None
        if ref is None or ref.name not in child:
            return token.group(0)
        kind, number = child[ref.name]
        return f"<{kind.capitalize()} {number}>"

    text = _TOKEN.sub(renumber, text)
    definitions, retention = [], []
    for ref in used:
        kind, number = child[ref.name]
        slot = f"<{kind.capitalize()} {number}>"
        if ref.kind == "audio":
            definitions.append(f"{slot} is the supplied audio reference for <{ref.name}>.")
            preserved = "the referenced audio signal and its supplied timing"
            label = slot
        else:
            definitions.append(f"<{ref.name}> is the {ref.kind} shown in {slot}.")
            preserved = (
                "keep the face, hair, and outfit of the referenced character. Ignore the white background"
                if ref.kind == "character"
                else "the architecture, materials and defining features of the referenced environment"
            )
            appears = [f"[Shot {n}]" for n, names in shots if ref.name in names]
            label = f"<{ref.name}>" + (f" (appears in {', '.join(appears)})" if appears else "")
        if ref.description:
            definitions[-1] += f" {ref.description}"
        retention.append(f"{label}: {ref.retention_analysis or f'fully_preserved - {preserved}.'}")
    additions = {
        "subject_definitions": "\n".join(definitions),
        "retention_analysis": "\n".join(retention),
        **shared,
    }
    return SegmentCall(_fill(text, additions), used, warnings)


def _sections(text: str) -> dict[str, tuple[int, int, int]]:
    """Each section's first header: (header start, body start, body end).

    Text before the first header is the description when no header names one.
    """
    headers = list(_HEADER.finditer(text))
    found: dict[str, tuple[int, int, int]] = {}
    for i, header in enumerate(headers):
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        name = re.sub(r"[ _-]+", "_", header.group(1).lower())
        found.setdefault(name, (header.start(), header.end(), body_end))
    lead = headers[0].start() if headers else len(text)
    if "detailed_description" not in found and text[:lead].strip():
        found["detailed_description"] = (0, 0, lead)
    return found


def _fill(text: str, additions: dict[str, str], *, replace_music: bool = False) -> str:
    """Append each addition inside its section, or add the section in canonical order.

    Only these insertions change the text. An `N/A` body gives way to described content;
    a segment's own style and music are kept unless a structured music override is given.
    """
    additions = {name: additions[name] for name in SECTIONS if additions.get(name)}
    if not text.strip():
        return "\n\n".join(f"{name}:\n{value}" for name, value in additions.items())
    found = _sections(text)
    edits: list[tuple[int, int, int, str]] = []
    for order, (name, addition) in enumerate(additions.items()):
        if name in found:
            _, start, end = found[name]
            body = text[start:end].strip()
            if body and name == "non_diegetic_music" and replace_music:
                at = text.index(body, start)
                edits.append((at, order, at + len(body), addition))
                continue
            if body and (name in {"style", "non_diegetic_music"} or addition.upper() == "N/A"):
                continue
            if body.upper() == "N/A":
                at = text.index(body, start)
                edits.append((at, order, at + len(body), addition))
            else:
                at = start + len(text[start:end].rstrip())
                edits.append((at, order, at, "\n" + addition))
            continue
        later = [found[n][0] for n in SECTIONS[SECTIONS.index(name) + 1:] if n in found]
        section = f"{name}:\n{addition}"
        if later:
            edits.append((min(later), order, min(later), section + "\n\n"))
        else:
            at = len(text.rstrip())
            edits.append((at, order, at, "\n\n" + section))
    for start, _, end, insert in sorted(edits, reverse=True):
        text = text[:start] + insert + text[end:]
    return text
