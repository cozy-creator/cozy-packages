"""FLUX.2 [dev], its Turbo LoRA, and FLUX.2 [klein] 9B and 4B: text to image, or an edit of up to ten
ordered references."""

from __future__ import annotations

import math
import secrets
from enum import IntEnum
from functools import partial
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    AssetBound,
    Context,
    ImageAsset,
    Memoize,
    Outputs,
    Shape,
    Telemetry,
    WeightsOutput,
)
from PIL import Image, ImageOps

from .model import MAX_PIXELS, MAX_SIDE, MIN_SIDE, PATCH, TURBO_SIGMAS, Flux2, prepare_references, schedule
from .quantize import int8, nvfp4
from .turbo import Flux2TurboLoRA
from .upsampler import Flux2Upsampler

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=64 << 30),), accelerator=False)
app.entrypoint(nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=48 << 30),), accelerator=False)


class Megapixels(IntEnum):
    MP1 = 1
    MP2 = 2
    MP4 = 4


#: Admission's bound per tier: its area (1024² per megapixel) plus 1/16 for the 16 px grid. With no
#: tier the first reference decides, at most 4 MiP.
_DEMAND: dict[Megapixels | None, tuple[int, int]] = {
    tier: (1088 * (tier or Megapixels.MP4), 1024) for tier in (*Megapixels, None)
}
_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
_REFERENCES = Annotated[
    list[ImageAsset],
    AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20, media_types=("image/png", "image/jpeg", "image/webp")),
    msgspec.Meta(max_length=10),
]


class GenerateInput(msgspec.Struct):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or composes them, in order.
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    #: Any W:H. None: the first reference's, else 1:1.
    aspect_ratio: _RATIO | None = None
    #: None: the first reference's pixel count, else 1.
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    #: None: the checkpoint's (README).
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] | None = None
    #: [dev]'s embedded guidance or [klein] Base's CFG scale. None: the checkpoint's.
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] | None = None
    #: [klein] Base's CFG negative.
    negative_prompt: str = ""
    #: Rewrite the prompt first with BFL's upsampler ([dev]'s Mistral; its first use fetches it).
    prompt_upsampling: bool = False
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(max_bytes=64 << 20, max_decoded_bytes=64 << 20, media_types=("image/png",))]
    width: int
    height: int
    seed: int
    #: The prompt the image was generated from: the upsampler's when `prompt_upsampling` is on.
    prompt: str


def _snap(value: float) -> int:
    return max(PATCH, round(value / PATCH) * PATCH)


def output_size(ratio: float, area: float) -> tuple[int, int]:
    """`area` pixels at W/H `ratio` on the 16 px grid, held to 64-4096 px sides and 4 MiP."""
    area = min(area, MAX_PIXELS)
    widest = min(MAX_SIDE**2 / area, area / MIN_SIDE**2)
    ratio = min(max(ratio, 1 / widest), widest)
    width = min(max(_snap(math.sqrt(area * ratio)), MIN_SIDE), MAX_SIDE)
    height = min(max(_snap(math.sqrt(area / ratio)), MIN_SIDE), MAX_SIDE)
    while width * height > _DEMAND[Megapixels.MP4][0] * _DEMAND[Megapixels.MP4][1]:
        width, height = (width - PATCH, height) if width >= height else (width, height - PATCH)
    return width, height


def _layout(adj: Adjustments, payload: GenerateInput, first: Image.Image | None) -> tuple[int, int]:
    """The output size: the ratio given (else the first reference's, else 1:1) at the tier given (else
    the first reference's pixel count, else 1 MP); out of range, the nearest the model admits."""
    w, h = map(int, payload.aspect_ratio.split(":")) if payload.aspect_ratio else first.size if first else (1, 1)
    area = payload.megapixels * 1024 * 1024 if payload.megapixels else first.width * first.height if first else 1 << 20
    width, height = output_size(w / h, area)
    if area > MAX_PIXELS:
        adj.clamp("megapixels", round(area / (1 << 20), 2), round(width * height / (1 << 20), 2), "the model's 4 MP maximum")
    elif (width, height) != (_snap(math.sqrt(area * w / h)), _snap(math.sqrt(area * h / w))):
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "the nearest size the model admits")
    return width, height


def _generate(ctx: Context, payload: GenerateInput, model: Flux2, turbo: Flux2TurboLoRA | None,
              upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    images = [ImageOps.exif_transpose(asset.read()) for asset in payload.reference_images]
    width, height = _layout(adj, payload, images[0] if images else None)
    turbo = turbo if model.dev else None
    # The checkpoint's defaults: [dev] 50 steps at guidance 4, its Turbo fal's 8 sigmas at 2.5,
    # [klein] Base 50 steps of CFG 4, [klein] distilled 4 steps without guidance (BFL `util.py`).
    steps = 8 if turbo else 4 if model.distilled else 50
    if payload.steps is not None and turbo and payload.steps != steps:
        adj.clamp("steps", payload.steps, steps, "the Turbo LoRA's own 8-step schedule")
    elif payload.steps is not None:
        steps = payload.steps
    guidance = None if model.distilled else payload.guidance if payload.guidance is not None else 2.5 if turbo else 4.0
    if payload.guidance is not None and model.distilled:
        adj.warn("guidance", "FLUX.2 [klein] distilled takes no guidance; ignored")
    cfg = not model.dev and not model.distilled and guidance > 1
    if payload.negative_prompt and not cfg:
        adj.warn("negative_prompt", "only FLUX.2 [klein] Base runs CFG; ignored")
    prompt = payload.prompt
    if payload.prompt_upsampling:
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            prompt = upsampler.upsample(payload.prompt, images)
        tel.log("upsampled prompt", caption=prompt[:2000])
    ctx.raise_if_cancelled()
    references = None
    if images:
        with tel.stage("encoding reference images", overall_range=(0.15, 0.2)):
            references = model.encode_references(prepare_references(images))
    with tel.stage("encoding prompt", overall_range=(0.2, 0.25)):
        embeds = model.encode(prompt)
        negative = model.encode(payload.negative_prompt) if cfg else None
    ctx.raise_if_cancelled()
    tokens = (height // PATCH) * (width // PATCH)
    timesteps = schedule(steps, tokens, TURBO_SIGMAS if turbo else None)
    denoise = partial(
        model.denoise, embeds, negative, references, width=width, height=height, timesteps=timesteps,
        guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled,
        on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.25, 0.95)),
    )
    with tel.stage("generating image", overall_range=(0.25, 0.95)):
        latents = turbo.applied(model.graph.components["transformer"], denoise) if turbo else denoise()
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed, prompt)


_DEV = "flux-2-dev@0.1.0"
#: The upsampler is [dev]'s text encoder, built and fetched only when `prompt_upsampling` is on; the
#: Turbo LoRA only with [dev].
_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("text_encoder",)}
_TURBO_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("text_encoder",),
                     "turbo_lora": ("model",)}


@app.entrypoint(defaults={"model": _DEV, "upsampler": _DEV}, components=_COMPONENTS, on_demand=("upsampler",),
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, model: Flux2, upsampler: Flux2Upsampler, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """The full model: FLUX.2 [dev] (50 steps, guidance 4), or a [klein] Base by `model=` (50 steps, CFG 4)."""
    return _generate(ctx, payload, model, None, upsampler, out, tel, adj)


@app.entrypoint(defaults={"model": _DEV, "turbo_lora": "flux-2-dev-turbo@0.1.0", "upsampler": _DEV},
                components=_TURBO_COMPONENTS, on_demand=("upsampler", "turbo_lora"), memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: GenerateInput, model: Flux2, turbo_lora: Flux2TurboLoRA,
                   upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """The fast model: FLUX.2 [dev] with fal's Turbo LoRA (8 steps, guidance 2.5), or a [klein] by
    `model=` (4 steps)."""
    return _generate(ctx, payload, model, turbo_lora, upsampler, out, tel, adj)
