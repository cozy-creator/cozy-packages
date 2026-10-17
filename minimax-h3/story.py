"""Shared reference assets and H3 segment prompts: authored text plus the global sections."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from itertools import pairwise
from typing import Annotated, Literal, NamedTuple

import msgspec
from cozy_runtime.author import AssetBound, AudioAsset, ImageAsset, InvalidRequest, VideoAsset

from minimax_h3.official import MAX_VIDEO_REFERENCES, Task, validate_reference_policy

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
_CUT = re.compile(r"\bcut(?:s|ting)?\b", re.IGNORECASE)
#: Each kind's file field, its H3 reference kind and its `<Picture|Audio|Video N>` label.
FILE = {"character": "image", "background": "image", "audio": "audio", "video": "video"}
_SLOT = {"character": "picture", "background": "picture", "audio": "audio", "video": "video"}
_POLICY = {"character": "image", "background": "image", "audio": "audio", "video": "video"}
_SPEAKER = re.compile(r"\s*\((S\d+)\)")
_PREFIX = re.compile(r"\s*\[([^\]\n]*)\]")
#: A clip's default retention: action guidance, never a subject.
MOTION = (
    "its movement, timing and choreography transfer to the subjects who perform them; "
    "its people, faces, clothing and setting are not kept"
)


class StoryReference(msgspec.Struct):
    """A supplied file (`FILE[kind]`), or a character or background image generated from its
    description."""

    name: str
    kind: Literal["character", "background", "audio", "video"]
    description: str = ""
    retention_analysis: Annotated[str, msgspec.Meta(max_length=1024)] | None = None
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
    video: Annotated[
        VideoAsset | None,
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
    prepare_character: Callable[[StoryReference], Awaitable[ImageAsset]] | None = None,
    show: Callable[[StoryReference, ImageAsset], None] | None = None,
) -> dict[str, ImageAsset | AudioAsset | VideoAsset]:
    """Prepare supplied characters, reuse other supplied files, and generate missing images
    once, concurrently. `show` gets each made image in request order, as soon as it and every
    earlier one exist."""
    images: dict[str, ImageAsset | AudioAsset | VideoAsset] = {}
    for ref in references:
        supplied = getattr(ref, FILE[ref.kind])
        if supplied is not None:
            images[ref.name] = supplied
    pending = [
        ref for ref in references
        if FILE[ref.kind] == "image" and (
            ref.image is None or ref.kind == "character" and prepare_character is not None
        )
    ]

    async def one(reference: StoryReference) -> ImageAsset:
        if reference.image is not None:
            assert prepare_character is not None
            return await prepare_character(reference)
        return await generate(reference)

    tasks = [asyncio.create_task(one(reference)) for reference in pending]
    ready = 0
    try:
        for finished in asyncio.as_completed(tasks):
            await finished
            while ready < len(tasks) and tasks[ready].done():
                reference, image = pending[ready], tasks[ready].result()
                images[reference.name] = image
                if show is not None:
                    show(reference, image)
                ready += 1
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return images


def image_prompt(reference: StoryReference, style: str = "") -> str:
    parts = [reference.description.strip()]
    if style.strip():
        parts.append(f"style:\n{style.strip()}")
    if reference.kind == "character":
        parts.append(
            "Generate exactly one image: a character design sheet on a plain white background, "
            "with three side-by-side panels of equal width showing the same character. Left: "
            "a full-body front view. Middle: a full-body back view. Right: a close-up of the "
            "head and shoulders, facing the camera with a neutral expression, showing the face "
            "in sharp detail: eyes, eyebrows, nose, lips, skin, hairline and hairstyle. Show "
            "the entire character in both full-body views, in a natural neutral pose. Keep the "
            "character's face, appearance, proportions and clothing consistent across all "
            "three panels."
        )
    else:
        parts.append(
            "Generate an empty environment reference image showing only the architecture, "
            "materials, lighting and defining features of the location. Absolutely no people, "
            "characters, human figures, faces, bodies, crowds, pedestrians, silhouettes or "
            "human-like subjects may appear anywhere in the image. Do not include labels, "
            "text, panels or character references."
        )
    parts.append(
        "Follow the visual style explicitly requested in the description or style section. "
        "If no visual style is specified, use a realistic photographic look. "
        "Use anime styling only when explicitly requested."
    )
    return "\n\n".join(parts)


def validate_references(references: Sequence[StoryReference]) -> dict[str, StoryReference]:
    # The render children enforce this same policy after model preparation; refuse here first.
    # Videos attach only where named, so the H3 limit on them is per call (`_segment_call`).
    videos = sum(ref.kind == "video" for ref in references)
    _policy(
        [_POLICY[ref.kind] for ref in references if ref.kind != "video"]
        + ["video"] * min(videos, MAX_VIDEO_REFERENCES)
    )
    by_name: dict[str, StoryReference] = {}
    for reference in references:
        key = name_key(reference.name)
        if key in by_name:
            raise InvalidRequest(
                f"duplicate reference name: {reference.name}", fields=["references"]
            )
        field = FILE[reference.kind]
        for other in {"image", "audio", "video"} - {field}:
            if getattr(reference, other) is not None:
                raise InvalidRequest(
                    f"{reference.kind} reference {reference.name} takes {field}, not {other}",
                    fields=["references"],
                )
        if getattr(reference, field) is None and (field != "image" or not reference.description.strip()):
            raise InvalidRequest(
                f"reference {reference.name} needs {'an image or a nonempty description' if field == 'image' else f'its {field}'}",
                fields=["references"],
            )
        if reference.retention_analysis is not None and not reference.retention_analysis.strip():
            raise InvalidRequest(
                f"reference {reference.name} has blank retention_analysis; omit it for the default",
                fields=["references"],
            )
        by_name[key] = reference
    return by_name


def _policy(kinds: list[str], prefix: str = "") -> None:
    try:
        validate_reference_policy(kinds)
    except ValueError as exc:
        raise InvalidRequest(f"{prefix}{exc}", code="reference_policy", fields=["references"]) from exc


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


def _numbering(references: Sequence[StoryReference]) -> dict[tuple[str, int], StoryReference]:
    """`<Picture N>`, `<Audio N>` and `<Video N>`: each kind counted in the given order."""
    counts = dict.fromkeys(("picture", "audio", "video"), 0)
    slots: dict[tuple[str, int], StoryReference] = {}
    for ref in references:
        slot = _SLOT[ref.kind]
        counts[slot] += 1
        slots[slot, counts[slot]] = ref
    return slots


def _segment_call(
    text: str,
    references: Sequence[StoryReference],
    shared: dict[str, str],
    index: int,
) -> SegmentCall:
    """Backgrounds always; characters, audio and videos only where the text names them.

    A token names a reference by its name first (any case, `_`, `-` and space alike), else as
    the global `<Picture N>`/`<Audio N>`/`<Video N>`; these are renumbered for the call. A voice
    (audio whose description names a character) follows its character. Videos go last, so
    the audio references keep their numbers ahead of a clip's soundtrack.
    """
    by_key = {name_key(ref.name): ref for ref in references}
    by_slot = _numbering(references)

    def lookup(token: re.Match[str]) -> StoryReference | None:
        named = by_key.get(name_key(token.group(1)))
        media = _MEDIA.fullmatch(name_key(token.group(1)))
        if named is not None or media is None:
            return named
        return by_slot.get((media.group(1).lower(), int(media.group(2))))

    def mentioned(span: str) -> list[StoryReference]:
        found = {ref.name: ref for token in _TOKEN.finditer(span) if (ref := lookup(token))}
        return list(found.values())

    unmatched = dict.fromkeys(
        token.group(0)
        for token in _TOKEN.finditer(text)
        if _MEDIA.fullmatch(name_key(token.group(1))) and lookup(token) is None
    )
    warnings = (
        [f"Segment {index + 1}: {', '.join(unmatched)} match no reference; left as written."]
        if unmatched else []
    )
    voices: dict[str, list[StoryReference]] = {}
    owners: dict[str, StoryReference] = {}
    for ref in references:
        owner = [o for o in mentioned(ref.description) if o.kind not in ("audio", "video")]
        if ref.kind == "audio" and owner:
            voices.setdefault(owner[0].name, []).append(ref)
            owners[ref.name] = owner[0]
    attached = {ref.name: ref for ref in references if ref.kind == "background"}
    for ref in mentioned(text):
        attached.setdefault(ref.name, ref)
        for voice in voices.get(ref.name, []):
            attached.setdefault(voice.name, voice)
    used = list(attached.values())
    if all(ref.kind == "audio" for ref in used):
        # H3's ref2va needs a picture: with no background and no character named, send every
        # reference but the videos.
        used = [ref for ref in references if ref.kind != "video"]
        warnings.append(
            f"Segment {index + 1} names no character and there is no background, so it receives "
            f"all {len(used)} of its image and audio references."
        )
    used = [ref for ref in used if ref.kind != "video"] + [ref for ref in used if ref.kind == "video"]
    _policy([_POLICY[ref.kind] for ref in used], f"segment {index + 1}: ")
    _, start, end = _sections(text).get("detailed_description", (0, 0, 0))
    markers = list(_SHOT.finditer(text, start, end))
    spans = pairwise([marker.start() for marker in markers] + [end])
    shots = [
        (m.group(1), {ref.name for ref in mentioned(text[a:b])})
        for m, (a, b) in zip(markers, spans, strict=True)
    ]
    _, start, end = _sections(text).get("retention_analysis", (0, 0, 0))
    retained = {ref.name for ref in mentioned(text[start:end])}

    def speaker(owner: StoryReference, span: str) -> str:
        """The `(Sx)` the text gives the voice's character, never a new one (guide §5.4)."""
        for token in _TOKEN.finditer(span):
            if lookup(token) is owner and (sx := _SPEAKER.match(span, token.end())):
                return f" ({sx.group(1)})"
        return ""

    child = {ref.name: slot for slot, ref in _numbering(used).items()}

    def renumber(token: re.Match[str]) -> str:
        ref = lookup(token) if _MEDIA.fullmatch(name_key(token.group(1))) else None
        if ref is None or ref.name not in child:
            return token.group(0)
        kind, number = child[ref.name]
        return f"<{kind.capitalize()} {number}>"

    voiced = {ref.name: speaker(owners[ref.name], text) or speaker(owners[ref.name], ref.description)
              for ref in used if ref.name in owners}
    text = _TOKEN.sub(renumber, text)
    definitions, retention = [], []
    tasks = set()
    for ref in used:
        kind, number = child[ref.name]
        slot = f"<{kind.capitalize()} {number}>"
        label = slot
        if ref.name in owners:
            owner = f"<{owners[ref.name].name}>"
            definitions.append(f"{slot} is the voice-timbre reference for {owner}{voiced[ref.name]}.")
            default = (f"reference - {owner}'s voice follows {slot}'s timbre and delivery without "
                       "copying the original signal.")
        elif ref.kind == "audio":
            definitions.append(f"{slot} is the supplied audio reference for <{ref.name}>.")
            default = f"reference - its sound guides the target audio without copying the original signal."
        elif ref.kind == "video":
            definitions.append(f"{slot} is the supplied action video." if _MEDIA.fullmatch(ref.name)
                               else f"<{ref.name}> is the action shown in {slot}.")
            label = f"{slot} (motion and choreography)"
            default = f"attribute_transfer - {MOTION}."
        else:
            definitions.append(f"<{ref.name}> is the {ref.kind} shown in {slot}.")
            preserved = (
                "keep the face, hair, and outfit of the referenced character. Ignore the white background"
                if ref.kind == "character"
                else "the architecture, materials and defining features of the referenced environment"
            )
            default = f"fully_preserved - {preserved}."
            appears = [f"[Shot {n}]" for n, names in shots if ref.name in names]
            label = f"<{ref.name}>" + (f" (appears in {', '.join(appears)})" if appears else "")
        if ref.description:
            definitions[-1] += f" {ref.description}"
        line = ref.retention_analysis or default
        if ref.kind == "audio":
            tasks.add("audio reuse" if line.split(" ", 1)[0] in ("fully_copy", "partially_copy")
                      else "audio reference")
        if ref.kind != "video" or ref.retention_analysis or ref.name not in retained:
            retention.append(f"{label}: {line}")
    _, start, end = _sections(text).get("summary", (0, 0, 0))
    if prefix := _PREFIX.match(text, start, end):
        named = [part.strip() for part in prefix.group(1).split("+")]
        missing = sorted(tasks - set(named))
        if missing:
            text = text[: prefix.start(1)] + " + ".join([*named, *missing]) + text[prefix.end(1):]
    additions = {
        "subject_definitions": "\n".join(definitions),
        "retention_analysis": "\n".join(retention),
        **shared,
    }
    return SegmentCall(_fill(text, additions), used, warnings)


def _shot_texts(text: str) -> list[str]:
    """The description's `[Shot N]` spans, or the whole description as one shot."""
    _, start, end = _sections(text).get("detailed_description", (0, 0, 0))
    marks = [m.start() for m in _SHOT.finditer(text, start, end)] or [start]
    return [text[a:b] for a, b in pairwise([*marks, end])]


def opening_cut_warnings(
    prompts: Sequence[str], references: Sequence[StoryReference], carries_picture: Sequence[bool]
) -> list[str]:
    """A hint, not a gate: a segment carrying the previous picture whose first shot cuts to
    characters the previous segment's last shot does not name. The carried frames open the
    new clip, so H3 keeps the old characters and the cut is lost."""
    by_key = {name_key(r.name): r.name for r in references if r.kind == "character"}
    by_slot = {key: ref.name for key, ref in _numbering(references).items() if ref.kind == "character"}

    def characters(shot: str) -> set[str]:
        found = set()
        for token in _TOKEN.finditer(shot):
            media = _MEDIA.fullmatch(token.group(1))
            name = (by_key.get(name_key(token.group(1))) if media is None
                    else by_slot.get((media.group(1).lower(), int(media.group(2)))))
            if name is not None:
                found.add(name)
        return found

    warnings = []
    for index, (previous, text) in enumerate(pairwise(prompts), start=1):
        if not carries_picture[index] or not text.strip():
            continue
        opening = _shot_texts(text)[0]
        new = sorted(characters(opening) - characters(_shot_texts(previous)[-1]))
        if new and _CUT.search(opening):
            names = ", ".join(f"<{name}>" for name in new)
            warnings.append(
                f"Segment {index + 1} carries the previous picture but opens with a cut to "
                f"{names}, not in segment {index}'s last shot; the carried frames keep its "
                "characters. For a clean picture cut with continuous sound, set "
                "context_frames: false."
            )
    return warnings


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
