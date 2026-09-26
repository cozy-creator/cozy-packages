"""Named, fixed character and scene references for independent camera-cut shots."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import AssetBound, ImageAsset, InvalidRequest

MAX_IMAGES = 9
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_-]*)\}")
_DIALOGUE_MARKER = re.compile(r"\{dialogue:([1-9][0-9]*)\}")
_MANUAL_SPEAKER = re.compile(r"\(S[1-9][0-9]*(?:,S[1-9][0-9]*)*\)")
_DIALOGUE_BODY = re.compile(r"(<d>.*?</d>)", re.DOTALL)
_SCREEN_TEXT_MARKER = re.compile(r"\{screen_text:([1-9][0-9]*)\}")
_STORY_MARKER = re.compile(r"\{(dialogue|screen_text):([1-9][0-9]*)\}")
ScreenText = Annotated[str, msgspec.Meta(min_length=1, max_length=1024)]


def _unquote_narrative(text: str) -> str:
    """Remove quotation delimiters, retaining apostrophes within words."""
    parts = _DIALOGUE_BODY.split(text)
    for part_index, part in enumerate(parts):
        if part.startswith("<d>"):
            continue
        parts[part_index] = "".join(
            char
            for index, char in enumerate(part)
            if char not in "\"'“”‘’«»"
            or (
                char in "'‘’"
                and 0 < index < len(part) - 1
                and part[index - 1].isalnum()
                and part[index + 1].isalnum()
            )
        )
    return "".join(parts)


class DialogueLine(msgspec.Struct, forbid_unknown_fields=True):
    """Exact speech inserted at its one-based {dialogue:N} marker."""

    speaker: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    text: Annotated[str, msgspec.Meta(min_length=1, max_length=1024)]
    language: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    delivery: Annotated[str, msgspec.Meta(max_length=256)] = ""
    voiceover: bool = False


class StorySegment(msgspec.Struct, forbid_unknown_fields=True):
    """One model invocation; camera-shot markers belong inside its description."""

    summary: Annotated[str, msgspec.Meta(min_length=1, max_length=1024)]
    detailed_description: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    overall_soundscape: Annotated[str, msgspec.Meta(max_length=1024)]
    non_diegetic_music: Annotated[str, msgspec.Meta(max_length=1024)]
    seed: int | None = None
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)] = 10
    dialogue: Annotated[list[DialogueLine], msgspec.Meta(max_length=16)] = msgspec.field(default_factory=list)
    screen_text: Annotated[list[ScreenText], msgspec.Meta(max_length=16)] = msgspec.field(default_factory=list)


class StoryReference(msgspec.Struct, forbid_unknown_fields=True):
    """A supplied image or an image generated from its shared visual description."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    kind: Literal["character", "scene"]
    description: Annotated[str, msgspec.Meta(max_length=1024)] = ""
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
        by_name[key] = reference
    return by_name


def select_references(
    names: Sequence[str], by_name: dict[str, StoryReference], *, index: int
) -> list[StoryReference]:
    keys = [name.casefold() for name in names]
    if not 1 <= len(names) <= MAX_IMAGES or len(set(keys)) != len(names):
        raise InvalidRequest(
            f"segment {index + 1} must select one to nine distinct reference names", fields=["segments"]
        )
    missing = [name for name in names if name.casefold() not in by_name]
    if missing:
        raise InvalidRequest(
            f"segment {index + 1} selects unknown references: {', '.join(missing)}", fields=["segments"]
        )
    return [by_name[key] for key in keys]


def dialogue_speakers(
    descriptions: Sequence[str],
    lines_by_shot: Sequence[Sequence[DialogueLine]],
    references_by_shot: Sequence[Sequence[StoryReference]],
    *,
    shared: Sequence[str] = (),
) -> dict[str, str]:
    """Validate before image generation; number speakers by actual marker chronology."""
    structured = any(lines_by_shot)
    speakers: dict[str, str] = {}
    if structured and any(
        "<d>" in text or "</d>" in text or _MANUAL_SPEAKER.search(text) for text in shared
    ):
        raise InvalidRequest(
            "structured dialogue cannot mix with manual H3 dialogue/speaker tags in shared fields",
            fields=["style", "soundscape", "music", "references"],
        )
    for index, (description, lines, references) in enumerate(
        zip(descriptions, lines_by_shot, references_by_shot, strict=True)
    ):
        def refuse(detail: str) -> None:
            raise InvalidRequest(f"segment {index + 1}: {detail}", fields=["segments", "dialogue"])

        if structured and (
            "<d>" in description or "</d>" in description or _MANUAL_SPEAKER.search(description)
        ):
            refuse("use either structured dialogue or manual H3 dialogue/speaker tags across the story")
        marker_text = _DIALOGUE_BODY.sub("", description)
        markers = list(_DIALOGUE_MARKER.finditer(marker_text))
        if "{dialogue:" in _DIALOGUE_MARKER.sub("", marker_text):
            refuse("dialogue markers must be {dialogue:1}, {dialogue:2}, and so on")
        positions = [int(marker.group(1)) for marker in markers]
        if sorted(positions) != list(range(1, len(lines) + 1)):
            refuse("insert every dialogue line exactly once with its one-based {dialogue:N} marker")
        selected = {reference.name.casefold(): reference for reference in references}
        for position in positions:
            line = lines[position - 1]
            key = line.speaker.casefold()
            reference = selected.get(key)
            if reference is None or reference.kind != "character":
                refuse(f"dialogue speaker {line.speaker!r} must be a selected character reference")
            if re.fullmatch(r"[A-Za-z][A-Za-z -]{0,47}", line.language) is None:
                refuse("dialogue language must be a plain label such as English or Mandarin Chinese")
            if not line.text.strip() or any(token in line.text for token in ("<", ">")):
                refuse("dialogue text must contain spoken words only, without H3/XML tags")
            ending = line.text.rstrip().rstrip('\"\'”’»」』')
            if not ending or ending[-1] not in ".!?。！？…":
                refuse("finish dialogue text with punctuation; supplied words and punctuation are never rewritten")
            if any(token in line.delivery for token in ("<", ">", "{", "}")) or (
                _MANUAL_SPEAKER.search(line.delivery)
            ):
                refuse("delivery must be plain direction outside the spoken text")
            if key not in speakers:
                speakers[key] = f"S{len(speakers) + 1}"
    return speakers


def segment_prompt(
    style: str,
    description: str,
    references: Sequence[StoryReference],
    *,
    index: int,
    summary: str,
    soundscape: str = "",
    music: str = "",
    dialogue: Sequence[DialogueLine] = (),
    speakers: dict[str, str] | None = None,
    screen_text: Sequence[str] = (),
) -> str:
    labels = {
        reference.name.casefold(): f"<Subject {slot}>"
        for slot, reference in enumerate(references, 1)
    }

    def substitute(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name.casefold() not in labels:
                raise InvalidRequest(
                    f"segment {index + 1} mentions {{{name}}} without selecting that reference",
                    fields=["segments", "detailed_description"],
                )
            return labels[name.casefold()]

        # Manual dialogue is already authored markup: never rewrite its spoken text.
        parts = _DIALOGUE_BODY.split(text.strip())
        return "".join(
            part if part.startswith("<d>") else _PLACEHOLDER.sub(replace, _unquote_narrative(part))
            for part in parts
        )

    if not description.strip():
        raise InvalidRequest(f"segment {index + 1} has an empty prompt", fields=["segments"])
    marker_text = _DIALOGUE_BODY.sub("", description)
    positions = [int(match.group(1)) for match in _SCREEN_TEXT_MARKER.finditer(marker_text)]
    if "{screen_text:" in _SCREEN_TEXT_MARKER.sub("", marker_text) or (
        sorted(positions) != list(range(1, len(screen_text) + 1))
    ):
        raise InvalidRequest(
            f"segment {index + 1}: insert every screen_text item exactly once with {{screen_text:N}}",
            fields=["segments", "screen_text"],
        )
    if any(not text.strip() or len(text) > 1024 or "<" in text or ">" in text for text in screen_text):
        raise InvalidRequest(
            f"segment {index + 1}: screen_text must contain 1–1024 characters of visible text without H3/XML tags",
            fields=["segments", "screen_text"],
        )
    if any("<d>" in text or "</d>" in text for text in (summary, soundscape, music)):
        raise InvalidRequest(
            "place dialogue in detailed_description, not summary, soundscape or music",
            fields=["segments"],
        )
    subjects: list[str] = []
    retention: list[str] = []
    for slot, reference in enumerate(references, 1):
        label = labels[reference.name.casefold()]
        appearance = _unquote_narrative(reference.description.strip())
        subjects.append(
            f"{label} is {reference.name}, the {reference.kind} shown in <Picture {slot}>."
            + (f" {appearance}" if appearance else "")
        )
        retained = (
            "the identity and defining visual features of the referenced character"
            if reference.kind == "character"
            else "the architecture, materials and defining features of the referenced environment"
        )
        retention.append(f"{label} (appears in [Shot 1]): fully_preserved - {retained}.")
    description = substitute(description)
    if not description.strip():
        raise InvalidRequest(
            f"segment {index + 1} has no prompt after removing quotation delimiters", fields=["segments"]
        )
    if dialogue and speakers is None:
        raise InvalidRequest("structured dialogue needs the story speaker map", fields=["segments"])

    def literal(match: re.Match[str]) -> str:
        position = int(match.group(2)) - 1
        if match.group(1) == "screen_text":
            return f'"{screen_text[position]}"'
        if speakers is not None:
            line = dialogue[position]
            key = line.speaker.casefold()
            voice = "says in an off-screen voiceover" if line.voiceover else "says"
            delivery = _unquote_narrative(line.delivery.strip())
            delivery = f", {delivery}," if delivery else ""
            text = (
                f"{labels[key]} ({speakers[key]}){delivery} {voice}: "
                f"<d>[{line.language}] {line.text}</d>"
            )
            if line.voiceover:
                text += f" {labels[key]}'s lips remain completely closed."
            return text
        raise InvalidRequest("structured dialogue needs the story speaker map", fields=["segments"])

    # One pass: literal speech/display text is never reinterpreted as another marker.
    description = "".join(
        part if part.startswith("<d>") else _STORY_MARKER.sub(literal, part)
        for part in _DIALOGUE_BODY.split(description)
    )
    sections = [
        "subject_definitions:\n" + "\n".join(subjects),
        "summary:\n[reference generation] " + substitute(summary),
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\n"
        + (substitute(style) + "\n" if style.strip() else "")
        + ("" if description.startswith("[Shot 1]") else "[Shot 1]\n") + description,
    ]
    sections.append("overall_soundscape:\n" + substitute(soundscape))
    sections.append("non_diegetic_music:\n" + substitute(music))
    prompt = "\n\n".join(sections)
    if len(prompt) > 4096:
        raise InvalidRequest(
            f"segment {index + 1}'s style, audio, reference descriptions and shot exceed 4096 "
            "characters; shorten them before rendering",
            fields=["segments", "style", "references"],
        )
    return prompt


def compile_segments(
    style: str, segments: Sequence[StorySegment], references: Sequence[StoryReference]
) -> list[str]:
    """Validate and compile against one fixed global reference table before any work."""
    validate_references(references)
    speakers = dialogue_speakers(
        [segment.detailed_description for segment in segments],
        [segment.dialogue for segment in segments],
        [references] * len(segments),
        shared=[
            style,
            *(reference.description for reference in references),
            *(segment.summary for segment in segments),
            *(segment.overall_soundscape for segment in segments),
            *(segment.non_diegetic_music for segment in segments),
        ],
    )
    return [
        segment_prompt(
            style, segment.detailed_description, references, index=index,
            summary=segment.summary, soundscape=segment.overall_soundscape,
            music=segment.non_diegetic_music, dialogue=segment.dialogue,
            speakers=speakers, screen_text=segment.screen_text,
        )
        for index, segment in enumerate(segments)
    ]
