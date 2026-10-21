"""Descriptions of supplied references for video prompts: a character's identity in one
sentence, a place's look, a motion clip's people, beats and camera; and plain text writing."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from typing import Annotated

import msgspec
from PIL import Image as PILImage
from PIL.Image import Image
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    DecodedVideoFrame,
    ImageAsset,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
    VideoPreparation,
    Weights,
    WeightsOutput,
    invocable,
)
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

from .vlm import Describer

app = App()

IDENTITY_PROMPT = (
    "Describe only how the single main character in this image looks, in exactly one concise "
    "sentence. Include distinctive visible identity, species or character type, hair, face, colors, "
    "clothing, armor, accessories, and carried or worn items when visible. Say nothing about what "
    "the character is doing or where they are: do not describe the background, setting, scenery, "
    "other characters, pose, posture, action, facial expression, camera, lighting, image quality, or "
    "art style. Do not invent hidden details or a name. Return only the sentence, without a label "
    "or introduction."
)
# Owner ruling (se-066): appearance only. A caption naming any of these is regenerated once with
# the words pointed out; clauses still naming one are then dropped.
SCENE_OR_POSE = re.compile(
    r"\b(?:background|backdrop|setting|scene|scenery|surroundings|room|indoors|outdoors|studio|wall|"
    r"mirror|selfie|stand(?:s|ing)?|sit(?:s|ting)?|seated|pos(?:e|es|ed|ing)|lean(?:s|ing)?|"
    r"kneel(?:s|ing)?|crouch(?:es|ing)?|walk(?:s|ing)?|looking|facing|gaz(?:es|ing)|confident(?:ly)?)\b",
    re.IGNORECASE,
)
MAX_NEW_TOKENS = 160
PLACE_PROMPT = (
    "Describe only the place in this image, in exactly one concise sentence: the kind of place, "
    "its architecture, materials, colors, furnishings and lighting. Say nothing about any person, "
    "character or animal, and nothing about the camera, image quality or art style. Return only the "
    "sentence, without a label or introduction."
)
#: A clip is watched as video: about 2 fps, at least 16 and at most 32 frames (a 3 s chop still
#: gets 16), each at most 448 px on its long edge.
CLIP_FPS = 2
CLIP_FRAMES = (16, 32)
CLIP_EDGE = 448
CLIP_SECONDS = 15
CLIP_PARTS = (
    "PEOPLE: one line per person, \"P1: <where in the frame at the start> - <build, hair, "
    "clothing>\".\n"
    "BEATS: one line per beat in time order, \"<start>-<end> s: <who does what to whom>; <which "
    "way each person turns, falls or moves>; <each person's facing relative to the camera>\".\n"
    "CAMERA: the framing and each camera move with its time.\n"
)
CLIP_CUTS = "CUTS: the time of each cut, or none.\n"
CLIP_RULES = (
    "Name people only as P1, P2 and so on. Describe only what is visible; do not guess names, "
    "intentions or the place's story."
)
CLIP_TOKENS = 700
#: A frame whose ffmpeg scene score (`select`, on frames 270 px high) reaches CUT_SCENE starts a
#: new shot; detections within CUT_GAP seconds of a cut are that cut. On se-072's 15 clips the
#: non-cuts peak at 0.16 and the 10 cuts start at 0.26: 0.2 finds all 10 and no false cut.
CUT_SCENE = 0.2
CUT_GAP = 0.25
SECTION = re.compile(r"^\s*(PEOPLE|BEATS|CAMERA|CUTS)\s*:\s*", re.IGNORECASE | re.MULTILINE)
PERSON = re.compile(r"\b(P\d+)\s*:\s*(.+?)(?=\s*\bP\d+\s*:|\Z)", re.DOTALL)


class DescribeCharacterInput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ]


class DescribeCharacterOutput(msgspec.Struct):
    description: Annotated[str, msgspec.Meta(min_length=1, max_length=2048)]


class DescribeCharactersInput(msgspec.Struct):
    images: Annotated[
        list[ImageAsset],
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
        msgspec.Meta(min_length=1, max_length=16),
    ]


class DescribeCharactersOutput(msgspec.Struct):
    #: One per image, in order; empty where the image shows no character's appearance.
    descriptions: list[Annotated[str, msgspec.Meta(max_length=2048)]]


class DescribeClipInput(msgspec.Struct):
    video: Annotated[
        VideoAsset,
        AssetBound(max_bytes=1 << 30, max_decoded_bytes=16 << 30),
        VideoPreparation(min_seconds=1, max_seconds=CLIP_SECONDS, excess="truncate"),
    ]


class DescribeClipOutput(msgspec.Struct):
    description: Annotated[str, msgspec.Meta(min_length=1, max_length=8192)]
    #: The frames the model watched, at their seconds.
    seconds: list[float]


class WriteTextsInput(msgspec.Struct):
    prompts: Annotated[
        list[Annotated[str, msgspec.Meta(min_length=1, max_length=65536)]],
        msgspec.Meta(min_length=1, max_length=16),
    ]
    max_new_tokens: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 1024


class WriteTextsOutput(msgspec.Struct):
    #: One reply per prompt, in order.
    texts: list[str]


#: The int8 lane: every projection of the 32 language layers (gated-delta-net and attention)
#: and their MLPs. The vision tower, embeddings, the decay/gate projections, norms and the LM
#: head keep their source bytes.
INT8 = QuantizationPlan(
    components=("model",),
    include=(
        "model.model.language_model.layers.*.linear_attn.in_proj_[qz]*.weight",
        "model.model.language_model.layers.*.linear_attn.out_proj.weight",
        "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
        "model.model.language_model.layers.*.mlp.*_proj.weight",
    ),
    output_precision="preserve",
)


@invocable(memo_version="character-describer-int8/1")
async def int8(ctx: Context, *, source: Weights[Describer], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8-convrot256/1`): W8A8 on int8 tensor cores; see `INT8`."""
    return quantize_into(ctx, tel, source=source, plan=INT8, encoding="int8-convrot256/1", output="int8")


app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=24 << 30),), accelerator=False)


@app.entrypoint(
    defaults={"model": "qwen3.5-9b@0.2.0"},
    memoize=True,
    memo_version="identity-v4",
)
def describe_character(
    ctx: Context,
    payload: DescribeCharacterInput,
    model: Describer,
    tel: Telemetry,
) -> DescribeCharacterOutput:
    """Describe one supplied character's appearance in one sentence."""
    ctx.raise_if_cancelled()
    with tel.stage("describing character"):
        (text,) = _describe(model, [payload.image.read()], tel)
    if not text:
        raise UnsupportedInput("the character description names no appearance", code="output_integrity")
    return DescribeCharacterOutput(text)


@app.entrypoint(
    defaults={"model": "qwen3.5-9b@0.2.0"},
    memoize=True,
    memo_version="identity-batch-v1",
)
def describe_characters(
    ctx: Context,
    payload: DescribeCharactersInput,
    model: Describer,
    tel: Telemetry,
) -> DescribeCharactersOutput:
    """Describe each supplied character's appearance in one sentence, all images in one pass."""
    ctx.raise_if_cancelled()
    with tel.stage("describing characters"):
        texts = _describe(model, [image.read() for image in payload.images], tel)
    return DescribeCharactersOutput(texts)


@app.entrypoint(
    defaults={"model": "qwen3.5-9b@0.2.0"},
    memoize=True,
    memo_version="place-batch-v1",
)
def describe_places(
    ctx: Context,
    payload: DescribeCharactersInput,
    model: Describer,
    tel: Telemetry,
) -> DescribeCharactersOutput:
    """Describe each supplied place in one sentence, people left out, all images in one pass."""
    ctx.raise_if_cancelled()
    with tel.stage("describing places"):
        images = [image.read() for image in payload.images]
        texts = model.describe(images, [PLACE_PROMPT] * len(images), max_new_tokens=MAX_NEW_TOKENS)
    return DescribeCharactersOutput(texts)


@app.entrypoint(
    defaults={"model": "qwen3.5-9b@0.2.0"},
    memoize=True,
    memo_version="clip-v2",
)
def describe_clip(
    ctx: Context,
    payload: DescribeClipInput,
    model: Describer,
    tel: Telemetry,
) -> DescribeClipOutput:
    """A motion clip's people, beats, camera and cuts. ffmpeg finds the cuts; each shot is watched
    as timestamped video, its people kept under the labels earlier shots gave them."""
    ctx.raise_if_cancelled()
    with tel.stage("reading clip"):
        frames, seconds = clip_frames(payload.video)
    with tel.stage("finding cuts"):
        cuts = scene_cuts(payload.video, tel)
    with tel.stage("describing clip"):
        text, shots = describe_shots(ctx, model, frames, seconds, cuts, float(payload.video.probe().duration))
    tel.log("clip described", frames=len(frames), shots=shots, description=text[:2000])
    if not text:
        raise UnsupportedInput("the clip description is empty", code="output_integrity")
    return DescribeClipOutput(text, seconds)


@app.entrypoint(
    defaults={"model": "qwen3.5-9b@0.2.0"},
    memoize=True,
    memo_version="write-v1",
)
def write_texts(
    ctx: Context,
    payload: WriteTextsInput,
    model: Describer,
    tel: Telemetry,
) -> WriteTextsOutput:
    """One greedy reply per text prompt, all in one pass."""
    ctx.raise_if_cancelled()
    with tel.stage("writing"):
        texts = model.write(list(payload.prompts), max_new_tokens=payload.max_new_tokens)
    return WriteTextsOutput(texts)


def clip_frames(video: VideoAsset) -> tuple[list[Image], list[float]]:
    """About `CLIP_FPS` frames a second, `CLIP_FRAMES` at least and at most, an even count,
    each the frame showing at the middle of its slot, at most `CLIP_EDGE` px long."""
    duration = float(video.probe().duration)
    count = min(max(round(duration * CLIP_FPS), CLIP_FRAMES[0]), CLIP_FRAMES[1])
    count += count % 2
    seconds = [(index + 0.5) * duration / count for index in range(count)]
    frames: list[Image] = []
    with video.stream(end=duration, audio=False) as events:
        for event in events:
            if not isinstance(event, DecodedVideoFrame):
                continue
            end = float(event.start_time + event.duration * event.time_base)
            picture = None
            while len(frames) < count and seconds[len(frames)] < end:
                if picture is None:
                    picture = PILImage.frombuffer("RGB", (event.width, event.height), event.rgb, "raw", "RGB", 0, 1)
                    scale = CLIP_EDGE / max(picture.size)
                    if scale < 1:
                        picture = picture.resize(
                            (round(picture.width * scale), round(picture.height * scale)), PILImage.Resampling.LANCZOS
                        )
                frames.append(picture)
    if not frames:
        raise UnsupportedInput("the clip has no frames", code="media_decode")
    frames += [frames[-1]] * (count - len(frames))
    return frames, seconds


def scene_cuts(video: VideoAsset, tel: Telemetry) -> list[float] | None:
    """The clip's cut times in seconds, from ffmpeg's scene score; None, with a warning, when
    ffmpeg is missing or cannot read the clip: the model then judges the cuts itself."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        tel.log("no ffmpeg on this machine: the model judges cuts itself (cozy tools install ffmpeg)", level="warning")
        return None
    duration = float(video.probe().duration)
    with tempfile.NamedTemporaryFile() as clip:
        clip.write(video.read_bytes())
        clip.flush()
        found = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-t", f"{duration:.3f}", "-i", clip.name, "-an", "-vf",
             f"setpts=PTS-STARTPTS,scale=-2:270,select='gte(scene,{CUT_SCENE})',metadata=print:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
    if found.returncode:
        tel.log("ffmpeg cannot read the clip: the model judges cuts itself", level="warning", error=found.stderr[-400:])
        return None
    cuts: list[float] = []
    for t in map(float, re.findall(r"pts_time:([\d.]+)", found.stdout)):
        if 0 < t < duration and (not cuts or t - cuts[-1] >= CUT_GAP):
            cuts.append(t)
    return cuts


def clip_shots(seconds: list[float], cuts: list[float], duration: float) -> list[tuple[float, float, list[int]]]:
    """Each shot's span and the watched frames inside it. A shot holding fewer than two of them is
    described with its neighbour (the one before it, or for the first shot the one after)."""
    bounds = [0.0, *cuts, duration]
    shots: list[tuple[float, float, list[int]]] = []
    for start, end in zip(bounds, bounds[1:], strict=False):
        held = [i for i, t in enumerate(seconds) if start <= t < end]
        if shots and (len(held) < 2 or len(shots[-1][2]) < 2):
            before, _, earlier = shots.pop()
            shots.append((before, end, earlier + held))
        else:
            shots.append((start, end, held))
    return shots


def clip_prompt(span: tuple[float, float] | None, people: list[str]) -> str:
    """The instruction for one shot (its span in the clip's seconds), naming the people earlier
    shots labelled; with no span the whole clip in one pass, cuts included."""
    shot = f"This is one shot of it, from {span[0]:.1f} s to {span[1]:.1f} s; give times in the clip's seconds. " if span else ""
    parts = CLIP_PARTS if span else CLIP_PARTS + CLIP_CUTS
    known = (
        "People labelled in earlier shots:\n" + "\n".join(people)
        + "\nKeep these labels for anyone who appears again; give anyone new the next free label.\n"
        if people else ""
    )
    return (
        "This clip's motion will be performed again by other characters in another place. "
        f"{shot}Describe its structure, not its story, in exactly these {'three' if span else 'four'} parts, "
        f"each starting on its own line:\n{parts}{known}{CLIP_RULES}"
    )


def describe_shots(
    ctx: Context, model: Describer, frames: list[Image], seconds: list[float], cuts: list[float] | None, duration: float
) -> tuple[str, int]:
    """One PEOPLE / BEATS / CAMERA / CUTS answer on the clip's timeline, and how many shots it
    took. A clip without cuts is one shot; with `cuts` None the model judges them in one pass."""
    shots = clip_shots(seconds, cuts or [], duration)
    people: dict[str, str] = {}
    beats: list[str] = []
    camera: list[str] = []
    judged = "none"
    for start, end, held in shots:
        ctx.raise_if_cancelled()
        held = held + held[-1:] * (len(held) % 2)
        reply = model.describe_video(
            [frames[i] for i in held], [seconds[i] for i in held],
            clip_prompt(None if cuts is None else (start, end), [f"{k}: {v}" for k, v in people.items()]),
            max_new_tokens=CLIP_TOKENS,
        )
        pieces = SECTION.split(reply)
        parts: dict[str, str] = {}
        for name, body in zip(pieces[1::2], pieces[2::2], strict=True):
            parts.setdefault(name.upper(), body.strip())
        listed = parts.get("PEOPLE", "")
        unlabelled = [(f"{start:.1f}-{end:.1f} s", listed)] if listed.strip(" .").lower() not in ("", "none") else []
        for label, text in PERSON.findall(listed) or unlabelled:
            people.setdefault(label, " ".join(text.split()))
        beats.append(parts.get("BEATS", "" if parts else reply.strip()))
        view = parts.get("CAMERA", "")
        camera.append(view if len(shots) == 1 or view[:1].isdigit() else f"{start:.1f}-{end:.1f} s: {view}")
        judged = parts.get("CUTS", judged)
    cut_text = judged if cuts is None else ", ".join(f"{t:.1f} s" for t in cuts) or "none"
    answer = {
        "PEOPLE": "\n".join(f"{k}: {v}" for k, v in people.items()),
        "BEATS": "\n".join(b for b in beats if b),
        "CAMERA": "\n".join(c for c in camera if c),
        "CUTS": cut_text,
    }
    return "\n\n".join(f"{name}:\n{body}" for name, body in answer.items()), len(shots)


def _describe(model: Describer, images: list[Image], tel: Telemetry) -> list[str]:
    texts = model.describe(images, [IDENTITY_PROMPT] * len(images), max_new_tokens=MAX_NEW_TOKENS)
    named = {i: sorted({w.lower() for w in SCENE_OR_POSE.findall(text)}) for i, text in enumerate(texts)}
    named = {i: words for i, words in named.items() if words}
    if not named:
        return texts
    for i, words in named.items():
        tel.log("description named setting or pose; regenerating", words=", ".join(words), first=texts[i][:400])
    retries = [
        f'{IDENTITY_PROMPT} An earlier answer was "{texts[i]}". It mentioned {", ".join(words)}, which '
        "describe the pose or surroundings, not the character. Rewrite it without them."
        for i, words in named.items()
    ]
    again = model.describe([images[i] for i in named], retries, max_new_tokens=MAX_NEW_TOKENS)
    for i, text in zip(named, again, strict=True):
        texts[i] = strip_scene_or_pose(text)
    return texts


def strip_scene_or_pose(text: str) -> str:
    """Drop each comma-separated clause that still names a setting or pose."""
    kept = [clause for clause in text.rstrip(". ").split(", ") if not SCENE_OR_POSE.search(clause)]
    return ", ".join(kept) + "." if kept else ""
