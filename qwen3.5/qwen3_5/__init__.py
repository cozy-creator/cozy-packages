"""Qwen3.5: single-turn questions about text, images and videos, answered greedily."""

from __future__ import annotations

from typing import Annotated

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    DecodedVideoFrame,
    ImageAsset,
    InvalidRequest,
    QuantizationPlan,
    Recipe,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
)
from PIL import Image as PILImage
from PIL.Image import Image

from .model import Qwen35, Question, Watched

app = App()

Images = Annotated[
    list[ImageAsset],
    AssetBound(
        max_bytes=64 << 20, max_decoded_bytes=256 << 20, media_types=("image/png", "image/jpeg", "image/webp")
    ),
    msgspec.Meta(max_length=16),
]
Videos = Annotated[
    list[VideoAsset], AssetBound(max_bytes=1 << 30, max_decoded_bytes=16 << 30), msgspec.Meta(max_length=4)
]
Text = Annotated[str, msgspec.Meta(max_length=65536)]


class Prompt(msgspec.Struct):
    text: Annotated[str, msgspec.Meta(min_length=1, max_length=65536)]
    images: Images = []
    videos: Videos = []


class AskInput(msgspec.Struct):
    #: One question, shown `images` then `videos`; or `prompts`, several decoded together.
    prompt: Text = ""
    images: Images = []
    videos: Videos = []
    prompts: Annotated[list[Prompt], msgspec.Meta(max_length=16)] = []
    system: Text = ""
    max_tokens: Annotated[int, msgspec.Meta(ge=1, le=16384)] = 1024
    #: Subtracted from every token already generated (vLLM's).
    presence_penalty: Annotated[float, msgspec.Meta(ge=-2, le=2)] = 0.0
    #: An answer ends before the first of these.
    stop: Annotated[list[Annotated[str, msgspec.Meta(min_length=1, max_length=256)]], msgspec.Meta(max_length=8)] = []
    #: Video reading, as Qwen-VL's: about `fps` frames a second, at least `min_frames` and at most
    #: `max_frames`; null takes the checkpoint's video processor (2, 4, 768).
    fps: Annotated[float, msgspec.Meta(gt=0, le=30)] | None = None
    min_frames: Annotated[int, msgspec.Meta(ge=2, le=768)] | None = None
    max_frames: Annotated[int, msgspec.Meta(ge=2, le=768)] | None = None
    #: Frames longer than this on their long edge are downscaled first (Lanczos); null: as decoded.
    max_edge: Annotated[int, msgspec.Meta(ge=32, le=4096)] | None = None
    #: Each video is watched up to this second; null: to its end.
    video_end: Annotated[float, msgspec.Meta(gt=0)] | None = None


class AskOutput(msgspec.Struct):
    #: One answer per prompt, in order.
    texts: list[str]


#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): every projection of the 32 language
#: layers (gated-delta-net and attention) and their MLPs. The vision tower, embeddings, the
#: decay/gate projections, norms and the LM head keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=Qwen35,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            "model.model.language_model.layers.*.linear_attn.in_proj_[qz]*.weight",
            "model.model.language_model.layers.*.linear_attn.out_proj.weight",
            "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "model.model.language_model.layers.*.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)


@app.entrypoint(defaults={"model": "qwen3.5-9b@0.2.0"}, memoize=True, memo_version="ask-v1")
def ask(ctx: Context, payload: AskInput, model: Qwen35, tel: Telemetry) -> AskOutput:
    """Answer each prompt about its text, images and videos, all prompts in one pass."""
    ctx.raise_if_cancelled()
    if payload.prompts and (payload.prompt or payload.images or payload.videos):
        raise InvalidRequest("send prompt (with images and videos) or prompts, not both", code="ask_prompts")
    prompts = payload.prompts or [Prompt(payload.prompt, payload.images, payload.videos)]
    if not prompts[0].text:
        raise InvalidRequest("prompt is empty", code="ask_prompts")
    video = model.processor.video_processor
    fps = payload.fps or float(video.fps)
    lo, hi = payload.min_frames or int(video.min_frames), payload.max_frames or int(video.max_frames)
    if lo > hi:
        raise InvalidRequest(f"min_frames {lo} is above max_frames {hi}", code="ask_frames")
    questions = []
    with tel.stage("reading"):
        for p in prompts:
            watched = [watch(v, fps, (lo, hi), payload.max_edge, payload.video_end) for v in p.videos]
            questions.append(Question(p.text, [image.read() for image in p.images], watched))
    with tel.stage("answering"):
        texts = model.ask(
            questions, system=payload.system, max_tokens=payload.max_tokens,
            presence_penalty=payload.presence_penalty, stop=list(payload.stop),
        )
    return AskOutput(texts)


def watch(
    video: VideoAsset, fps: float, frames: tuple[int, int], max_edge: int | None, end: float | None
) -> Watched:
    """About `fps` frames a second, `frames` at least and at most, an even count (Qwen-VL reads
    frame pairs), each the frame showing at the middle of its slot, downscaled to `max_edge`."""
    duration = float(video.probe().duration)
    if end is not None:
        duration = min(duration, end)
    count = min(max(round(duration * fps), frames[0]), frames[1])
    count += count % 2
    seconds = [(index + 0.5) * duration / count for index in range(count)]
    shown: list[Image] = []
    with video.stream(end=duration, audio=False) as events:
        for event in events:
            if not isinstance(event, DecodedVideoFrame):
                continue
            until = float(event.start_time + event.duration * event.time_base)
            picture = None
            while len(shown) < count and seconds[len(shown)] < until:
                if picture is None:
                    picture = PILImage.frombuffer("RGB", (event.width, event.height), event.rgb, "raw", "RGB", 0, 1)
                    scale = max_edge / max(picture.size) if max_edge else 1
                    if scale < 1:
                        picture = picture.resize(
                            (round(picture.width * scale), round(picture.height * scale)), PILImage.Resampling.LANCZOS
                        )
                shown.append(picture)
    if not shown:
        raise UnsupportedInput("the video has no frames", code="media_decode")
    shown += [shown[-1]] * (count - len(shown))
    return Watched(shown, seconds)
