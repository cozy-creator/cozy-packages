"""Shared reference assets and H3 segment prompts: authored text plus the global sections."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from itertools import pairwise
from typing import Annotated, Literal, NamedTuple

import msgspec
from cozy_runtime.author import (
    AssetBound,
    AudioAsset,
    AudioPreparation,
    ImageAsset,
    InvalidRequest,
    VideoAsset,
    VideoPreparation,
)

from minimax_h3.official import MAX_VIDEO_REFERENCES, Task, validate_reference_policy

#: MiniMax's full-reference sections in their canonical order. A `style` section is read and
#: moved into the description's opening, where the guide puts it (ref guide 5.2).
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
#: Reference clips: a video longer than 15 s is truncated to its first 15 s (Runtime records the
#: adjustment); audio outside 2-15 s refuses. Every input that carries a reference declares these.
REFERENCE_VIDEO = VideoPreparation(min_seconds=2, max_seconds=15, excess="truncate")
REFERENCE_AUDIO = AudioPreparation(min_seconds=2, max_seconds=15, excess="refuse")
#: A reference video's upload: a source of any length is admitted and truncated, not refused.
VIDEO_MAX_BYTES = 1 << 30
#: Read bounds. A video streams, one decoded frame at a time, straight to its motion size
#: (`read_reference_video`), so a frame of any real source fits. A prepared image is at most
#: 16,777,216 RGB8 pixels; audio is 15 s, mono or stereo.
VIDEO_READ_BYTES = 16 << 30
IMAGE_READ_BYTES = 64 << 20
AUDIO_READ_BYTES = 64 << 20
#: Each kind's file field, its H3 reference kind and its `<Picture|Audio|Video N>` label.
FILE = {"character": "image", "background": "image", "audio": "audio", "video": "video"}
_SLOT = {"character": "picture", "background": "picture", "audio": "audio", "video": "video"}
_POLICY = {"character": "image", "background": "image", "audio": "audio", "video": "video"}
_SPEAKER = re.compile(r"\s*\((S\d+)\)")
_PREFIX = re.compile(r"\s*\[([^\]\n]*)\]")
#: The summary's task types (ref guide 3); a summary without one opens with the first.
TASKS = ("reference generation", "keyframe completion", "video editing", "video continuation",
         "audio reuse", "audio reference")
#: A clip guides a description shorter than this (the guide asks 350-500 words): H3 then fills
#: the gaps from the clip and copies its people and setting (runs 10706, 10783 vs 10707).
SHORT_DESCRIPTION = 150


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
        AssetBound(max_bytes=256 << 20, max_decoded_bytes=AUDIO_READ_BYTES),
        REFERENCE_AUDIO,
    ] = None
    video: Annotated[
        VideoAsset | None,
        AssetBound(max_bytes=VIDEO_MAX_BYTES, max_decoded_bytes=VIDEO_READ_BYTES),
        REFERENCE_VIDEO,
    ] = None
    #: A video's own sound: dropped unless set, so the clip guides motion alone.
    use_audio: bool = False
    #: A character's voice: a timbre reference wherever the character is named.
    voice: Annotated[
        AudioAsset | None,
        AssetBound(max_bytes=256 << 20, max_decoded_bytes=AUDIO_READ_BYTES),
        REFERENCE_AUDIO,
    ] = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None

    def __post_init__(self) -> None:
        # Prompts quote the name as <name>: any text reads as a name once its brackets and
        # extra whitespace go.
        self.name = " ".join(self.name.replace("<", " ").replace(">", " ").split())
        if not self.name:
            raise ValueError("a reference name needs at least one visible character")


def voices(references: Sequence[StoryReference]) -> tuple[list[StoryReference], dict[str, StoryReference]]:
    """Every reference with each character's voice after it, as the audio reference
    "<Name>'s voice", and each voice's character."""
    pool: list[StoryReference] = []
    owners: dict[str, StoryReference] = {}
    for ref in references:
        pool.append(ref)
        if ref.voice is not None:
            voice = StoryReference(f"{ref.name}'s voice", "audio", audio=ref.voice)
            pool.append(voice)
            owners[voice.name] = ref
    return pool, owners


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
    for ref in voices(references)[0]:
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
    pool = voices(references)[0]
    videos = sum(ref.kind == "video" for ref in pool)
    _policy(
        [_POLICY[ref.kind] for ref in pool if ref.kind != "video"]
        + ["video"] * min(videos, MAX_VIDEO_REFERENCES)
    )
    by_name: dict[str, StoryReference] = {}
    for reference in pool:
        key = name_key(reference.name)
        if key in by_name:
            raise InvalidRequest(
                f"duplicate reference name: {reference.name}", fields=["references"]
            )
        field = FILE[reference.kind]
        if reference.voice is not None and reference.kind != "character":
            raise InvalidRequest(
                f"only a character takes a voice, not {reference.kind} {reference.name}",
                fields=["references"],
            )
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
    overrides = [None] * len(prompts) if segment_music is None else segment_music
    calls = []
    for index, (text, override) in enumerate(zip(prompts, overrides, strict=True)):
        music = non_diegetic_music if override is None else override
        text = _fill(
            text, {"non_diegetic_music": music.strip() or "N/A"},
            replace_music=override is not None,
        )
        calls.append(_segment_call(text, references, style.strip(), overall_soundscape.strip(), index))
    sent = {ref.name for call in calls for ref in call.references}
    for ref in references:
        if calls and ref.kind == "video" and ref.name not in sent:
            calls[0].warnings.append(
                f"<{ref.name}> is never named in a prompt, so H3 never sees it; name the clip "
                "where its motion applies."
            )
    return calls


def _numbering(references: Sequence[StoryReference]) -> dict[tuple[str, int], StoryReference]:
    """`<Picture N>`, `<Audio N>` and `<Video N>`: each kind counted in the given order, as H3
    labels them. A `use_audio` clip's soundtrack takes an `<Audio N>` just before its video."""
    counts = dict.fromkeys(("picture", "audio", "video"), 0)
    slots: dict[tuple[str, int], StoryReference] = {}
    for ref in references:
        for slot in ("audio", "video") if ref.kind == "video" and ref.use_audio else (_SLOT[ref.kind],):
            counts[slot] += 1
            slots[slot, counts[slot]] = ref
    return slots


def _segment_call(
    text: str,
    references: Sequence[StoryReference],
    style: str,
    soundscape: str,
    index: int,
) -> SegmentCall:
    """Backgrounds always; characters, audio and videos only where the text names them, all in
    request order, as H3 packs and labels them.

    A token names a reference by its name first (any case, `_`, `-` and space alike), else as
    the global `<Picture N>`/`<Audio N>`/`<Video N>`; these are renumbered for the call. A
    character's voice follows the character. The prompt takes MiniMax's full-reference form
    (ref guide): a summary task type, each clip bound to the characters who perform it, and
    `style` as the description's opening sentence.
    """
    pool, owners = voices(references)
    by_key = {name_key(ref.name): ref for ref in pool}
    by_slot = _numbering(pool)
    text, style = _open_with_style(text, style)
    words = len(_body(text, "detailed_description").split()) - len(style.split())

    def lookup(token: re.Match[str]) -> StoryReference | None:
        named = by_key.get(name_key(token.group(1)))
        media = _MEDIA.fullmatch(name_key(token.group(1)))
        if named is not None or media is None:
            return named
        return by_slot.get((media.group(1).lower(), int(media.group(2))))

    def part(token: re.Match[str], ref: StoryReference) -> tuple[str, str]:
        """The reference and which of its labels a token names: a clip's sound or picture."""
        media = _MEDIA.fullmatch(name_key(token.group(1)))
        named = by_key.get(name_key(token.group(1))) is not None
        return ref.name, media.group(1).lower() if media and not named else _SLOT[ref.kind]

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
    named = {ref.name for ref in mentioned(text)}
    named |= {voice.name for voice in pool if voice.name in owners and owners[voice.name].name in named}
    used = [ref for ref in pool if ref.kind == "background" or ref.name in named]
    if all(ref.kind == "audio" for ref in used):
        # H3's ref2va needs a picture: with no background and no character named, send every
        # reference but the videos.
        used = [ref for ref in pool if ref.kind != "video"]
        warnings.append(
            f"Segment {index + 1} names no character and there is no background, so it receives "
            f"all {len(used)} of its image and audio references."
        )
    _policy([_POLICY[ref.kind] for ref in used], f"segment {index + 1}: ")
    sections = _sections(text)
    _, start, end = sections.get("detailed_description", (0, 0, 0))
    markers = list(_SHOT.finditer(text, start, end))
    spans = pairwise([marker.start() for marker in markers] + [end])
    shots = [
        (m.group(1), {ref.name for ref in mentioned(text[a:b])})
        for m, (a, b) in zip(markers, spans, strict=True)
    ]
    _, start, end = sections.get("retention_analysis", (0, 0, 0))
    retained = {
        part(token, ref) for token in _TOKEN.finditer(text, start, end) if (ref := lookup(token))
    }
    given = _tasks(text)
    editing = bool({"video editing", "video continuation"} & {task.casefold() for task in given})

    def speaker(owner: StoryReference, span: str) -> str:
        """The `(Sx)` the text gives the voice's character, never a new one (guide §5.4)."""
        for token in _TOKEN.finditer(span):
            if lookup(token) is owner and (sx := _SPEAKER.match(span, token.end())):
                return f" ({sx.group(1)})"
        return ""

    child = {(ref.name, kind): number for (kind, number), ref in _numbering(used).items()}

    def label(ref: StoryReference, kind: str | None = None) -> str:
        kind = kind or _SLOT[ref.kind]
        return f"<{kind.capitalize()} {child[ref.name, kind]}>"

    def renumber(token: re.Match[str]) -> str:
        ref = lookup(token) if _MEDIA.fullmatch(name_key(token.group(1))) else None
        if ref is None or part(token, ref) not in child:
            return token.group(0)
        return label(ref, part(token, ref)[1])

    def names(refs: Sequence[StoryReference]) -> str:
        tokens = [f"<{ref.name}>" for ref in refs]
        return ", ".join(tokens[:-1]) + " and " + tokens[-1] if len(tokens) > 1 else "".join(tokens)

    characters = [ref for ref in used if ref.kind == "character"]
    places = [ref for ref in used if ref.kind == "background"]
    clips = [ref for ref in used if ref.kind == "video"]

    def performers(clip: StoryReference) -> list[StoryReference]:
        """The characters a motion clip's shots name, else every character in the call."""
        cast = set().union(*(found for _, found in shots if clip.name in found))
        return [ref for ref in characters if ref.name in cast] or characters

    motion = {} if editing else {clip.name: performers(clip) for clip in clips}
    voiced = {ref.name: speaker(owners[ref.name], text) for ref in used if ref.name in owners}
    text = _TOKEN.sub(renumber, text)
    definitions, retention = [], []
    tasks = set()
    for ref in used:
        slot = label(ref)
        line_label = slot
        if ref.name in owners:
            owner = f"<{owners[ref.name].name}>"
            definitions.append(f"{slot} is the voice-timbre reference for {owner}{voiced[ref.name]}.")
            default = (f"reference - {owner}'s voice follows {slot}'s timbre and delivery without "
                       "copying the original signal.")
        elif ref.kind == "audio":
            definitions.append(f"{slot} is the supplied audio reference for <{ref.name}>.")
            default = f"partially_copy - {slot} is copied into the target audio, with other sounds around it."
        elif ref.kind == "video":
            role = "source video" if editing else "motion and camera reference"
            who = f" for {names(motion[ref.name])}" if motion.get(ref.name) else ""
            definitions.append(f"{slot} is the {role}{who}." if _MEDIA.fullmatch(ref.name)
                               else f"<{ref.name}> is the {role} in {slot}{who}.")
            if ref.use_audio:
                sound = label(ref, "audio")
                definitions.append(f"{sound} is the synchronized audio track of {slot}.")
                if (ref.name, "audio") not in retained:
                    retention.append(f"{sound}: partially_copy - the synchronized audio track of {slot} "
                                     "is copied into the target audio, with other sounds around it.")
                    tasks.add("audio reuse")
            if editing:
                line_label = f"{slot} (source video editing)"
                default = ("fully_preserved - its framing, setting and subjects are kept except where "
                           "the description changes them.")
            else:
                line_label = f"{slot} (motion, choreography and camera movement)"
                cast = names(motion[ref.name]) or "the target video's subjects"
                where = f" in {names(places)}" if places else ""
                default = (f"attribute_transfer - its movements, timing and camera work are performed "
                           f"by {cast}{where}.")
        else:
            moves = [f"<{clip.name}>" if not _MEDIA.fullmatch(clip.name) else label(clip)
                     for clip in clips if ref in motion.get(clip.name, ())]
            binding = f", whose movements come from {' and '.join(moves)}" if moves else ""
            definitions.append(f"<{ref.name}> is the {ref.kind} shown in {slot}{binding}.")
            preserved = (
                "keep the face, hair, and outfit of the referenced character. Ignore the white background"
                if ref.kind == "character"
                else "the architecture, materials and defining features of the referenced environment"
            )
            default = f"fully_preserved - {preserved}."
            appears = [f"[Shot {n}]" for n, found in shots if ref.name in found]
            line_label = f"<{ref.name}>" + (f" (appears in {', '.join(appears)})" if appears else "")
        if ref.description:
            definitions[-1 - (ref.kind == "video" and ref.use_audio)] += f" {ref.description}"
        line = ref.retention_analysis or default
        if ref.kind == "audio":
            tasks.add("audio reuse" if line.split(" ", 1)[0] in ("fully_copy", "partially_copy")
                      else "audio reference")
        if ref.kind != "video" or ref.retention_analysis or (ref.name, "video") not in retained:
            retention.append(f"{line_label}: {line}")
    if clips and not editing and words < SHORT_DESCRIPTION:
        warnings.append(
            f"Segment {index + 1}: {names(clips)} guide{'s' * (len(clips) == 1)} a description of "
            f"{words} words. H3 copies a clip's own people and setting unless the description "
            "establishes each character, the background and the beats in order; MiniMax's guide "
            "asks 350-500 words."
        )
    if "summary" in (sections := _sections(text)):
        _, start, end = sections["summary"]
        tag = " + ".join([*(given or ["reference generation"]), *sorted(tasks - set(given))])
        if prefix := _PREFIX.match(text, start, end) if given else None:
            text = text[: prefix.start(1)] + tag + text[prefix.end(1):]
        elif body := text[start:end].lstrip():
            text = text[: end - len(body)] + f"[{tag}] " + body + text[end:]
        else:
            text = text[:start] + f"\n[{tag}]" + text[start:]
    additions = {
        "subject_definitions": "\n".join(definitions),
        "retention_analysis": "\n".join(retention),
        "overall_soundscape": soundscape,
    }
    return SegmentCall(_fill(text, additions), used, warnings)


def _tasks(text: str) -> list[str]:
    """The summary's `[task + task]` prefix, when it opens with one."""
    _, start, end = _sections(text).get("summary", (0, 0, 0))
    prefix = _PREFIX.match(text, start, end)
    given = [task.strip() for task in prefix.group(1).split("+")] if prefix else []
    return given if all(task.casefold() in TASKS for task in given) else []


def _body(text: str, name: str) -> str:
    _, start, end = _sections(text).get(name, (0, 0, 0))
    return text[start:end]


def _open_with_style(text: str, style: str) -> tuple[str, str]:
    """The segment's own `style` section, else the global style, as the description's opening
    sentence (ref guide 5.2) rather than a section of its own; and that style."""
    found = _sections(text)
    if "style" in found:
        head, start, end = found["style"]
        style = text[start:end].strip() or style
        text = text[:head] + text[end:]
    if not style:
        return text, style
    found = _sections(text)
    if "detailed_description" not in found:
        return _fill(text, {"detailed_description": style}), style
    head, start, end = found["detailed_description"]
    if head == start:
        return f"{style}\n{text}", style
    rest = text[start:end].lstrip(" \t")
    return text[:start] + "\n" + style + ("" if rest.startswith("\n") else "\n") + rest + text[end:], style


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
    a segment's own music is kept unless a structured music override is given. A description
    without a heading gains `detailed_description:` once sections are inserted ahead of it.
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
            if body and (name == "non_diegetic_music" or addition.upper() == "N/A"):
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
    if found.get("detailed_description", (1, 0, 0))[:2] == (0, 0) and any(at == 0 for at, *_ in edits):
        edits.append((0, len(additions), 0, "detailed_description:\n"))
    for start, _, end, insert in sorted(edits, reverse=True):
        text = text[:start] + insert + text[end:]
    return text
