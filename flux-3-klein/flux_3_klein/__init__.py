"""FLUX.3 Image Klein text-to-image and reference editing, distilled and base, with BFL's prompt upsampler."""

from __future__ import annotations

import math
import secrets
from enum import IntEnum
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
from PIL import Image

from .klein import MAX_SIDE, MIN_SIDE, Klein, prepare_multi_reference, prepare_reference
from .quantize import drafter_head, int8, nvfp4, upsampler_int8, upsampler_nvfp4
from .upsampler import Drafter, Upsampler

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=12 << 30),), accelerator=False)
app.entrypoint(nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=12 << 30),), accelerator=False)
app.entrypoint(upsampler_int8, weights=(WeightsOutput("int8", max_new_bytes=12 << 30),), accelerator=False)
app.entrypoint(upsampler_nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=8 << 30),), accelerator=False)
app.entrypoint(drafter_head, weights=(WeightsOutput("bf16", max_new_bytes=1 << 30),), accelerator=False)


class Megapixels(IntEnum):
    MP1 = 1
    MP2 = 2
    MP4 = 4


#: Admission's bound per tier: its area (1024² per megapixel) plus 1/16 for rounding to 32 px. An
#: edit with no size takes its first reference's, at most 4 MiP, so it is bounded at the largest.
_DEMAND: dict[Megapixels | None, tuple[int, int]] = {
    tier: (1088 * (tier or Megapixels.MP4), 1024) for tier in (*Megapixels, None)
}
_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]


def _rounded(area: int, ratio: float) -> tuple[int, int]:
    """Qwen-Image's `calculate_dimensions`: the area at that ratio, each side rounded to 32 px."""
    width = math.sqrt(area * ratio)
    return round(width / 32) * 32, round(width / ratio / 32) * 32


def output_size(megapixels: Megapixels, ratio: str) -> tuple[int, int, bool]:
    """The tier's area at `W:H`, held inside BFL's 64-4096 px sides and the tier's demand; True
    when that moved it off the rounded size."""
    area, (w, h) = megapixels * 1024 * 1024, map(int, ratio.split(":"))
    widest = min(MAX_SIDE**2 / area, area / MIN_SIDE**2)
    width, height = _rounded(area, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, MIN_SIDE), MAX_SIDE), min(max(height, MIN_SIDE), MAX_SIDE)
    while width * height > _DEMAND[megapixels][0] * _DEMAND[megapixels][1]:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != _rounded(area, w / h)


def _nearest_tier(pixels: int) -> Megapixels:
    return min(Megapixels, key=lambda tier: abs(tier * 1024 * 1024 - pixels))


def _sized(adj: Adjustments, ratio: str, megapixels: Megapixels) -> tuple[int, int]:
    width, height, held = output_size(megapixels, ratio)
    if held:
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "nearest size the model admits")
    return width, height


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    aspect_ratio: _RATIO = "1:1"
    megapixels: Annotated[Megapixels, Shape(pixels=_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 16
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


class GenerateBaseInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    aspect_ratio: _RATIO = "1:1"
    megapixels: Annotated[Megapixels, Shape(pixels=_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] = 50
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 3.5
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


class EditInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20,
                   media_types=("image/png", "image/jpeg", "image/webp")),
        msgspec.Meta(min_length=1, max_length=10),
    ]
    aspect_ratio: _RATIO | None = None
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 16
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


class EditBaseInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20,
                   media_types=("image/png", "image/jpeg", "image/webp")),
        msgspec.Meta(min_length=1, max_length=10),
    ]
    aspect_ratio: _RATIO | None = None
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] = 50
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 3.5
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=64 << 20, max_decoded_bytes=64 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int
    #: The caption the image was generated from: the upsampler's when `prompt_upsampling` is on.
    prompt: str


def _render(
    ctx: Context,
    model: Klein,
    upsampler: Upsampler,
    drafter: Drafter,
    out: Outputs,
    tel: Telemetry,
    *,
    prompt: str,
    upsample: bool,
    draft: bool,
    images: list[Image.Image],
    prepared: list[Image.Image],
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    canvas: str | None,
    steps: int,
    guidance: float | None,
    seed: int | None,
) -> ImageOutput:
    seed = seed if seed is not None else secrets.randbits(53)
    width, height = size
    caption = prompt
    if upsample:
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            def run() -> str:
                return upsampler.upsample(prompt, images, canvas)
            caption = drafter.drafting(run) if draft else run()
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    references = None
    if prepared:
        with tel.stage("encoding reference images", overall_range=(0.15, 0.2)):
            references = model.encode_references(prepared)
    with tel.stage("encoding prompt", overall_range=(0.2, 0.25)):
        embeds = model.encode(caption)
        negative = model.encode("") if guidance is not None else None
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.25, 0.95)):
        tokens = model.denoise(
            embeds, negative, references,
            width=width, height=height, steps=steps, guidance=guidance, seed=seed,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.25, 0.95)),
            cancel=ctx.raise_if_cancelled,
        )
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(tokens, width=width, height=height)
        if crop is not None:
            image = image.crop(crop)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, image.width, image.height, seed, caption)


def _references(
    adj: Adjustments, ratio: str | None, megapixels: Megapixels | None, images: list[Image.Image]
) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int, int, int] | None]:
    """The prepared references, the output size and the crop after decoding. Given neither size
    input, BFL's default: the first prepared reference's own size, a lone one's padding cropped
    off. Otherwise the ratio (else the first reference's) at the tier (else the one nearest it)."""
    box = None
    if len(images) == 1:
        lone, box = prepare_reference(images[0])
        prepared = [lone]
    else:
        prepared = [prepare_multi_reference(image) for image in images]
    if ratio is None and megapixels is None:
        return prepared, prepared[0].size, box
    first = images[0]
    tier = megapixels or _nearest_tier(first.width * first.height)
    return prepared, _sized(adj, ratio or f"{first.width}:{first.height}", tier), None


# Qwen3.5-9B's MTP head alone (`drafter_head` makes it): BFL's upsampler checkpoint has none.
_DRAFTER = "qwen3.5-9b-mtp@0.2.0"
_DISTILLED = {"model": "flux-3-klein@0.1.0", "upsampler": "flux-3-klein-upsampler@0.2.0", "drafter": _DRAFTER}
_BASE = {"model": "flux-3-klein-base@0.1.0", "upsampler": "flux-3-klein-upsampler@0.2.0", "drafter": _DRAFTER}
_COMPONENTS = {
    "model": ("transformer", "text_encoder", "vae"),
    "upsampler": ("model",),
    "drafter": ("model",),
}
#: Built and fetched on first use: `prompt_upsampling=False` never pulls the upsampler, and
#: `prompt_drafting=False` never pulls the drafter.
_ON_DEMAND = ("upsampler", "drafter")


@app.entrypoint(defaults=_DISTILLED, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Text to image with the distilled Klein: 16 steps, no guidance."""
    return _render(ctx, model, upsampler, drafter, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   draft=payload.prompt_drafting,
                   images=[], prepared=[], size=_sized(adj, payload.aspect_ratio, payload.megapixels), crop=None,
                   canvas=payload.aspect_ratio, steps=payload.steps, guidance=None, seed=payload.seed)


@app.entrypoint(defaults=_BASE, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def generate_base(ctx: Context, payload: GenerateBaseInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
                  out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Text to image with Klein Base: 50 steps, CFG 3.5."""
    return _render(ctx, model, upsampler, drafter, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   draft=payload.prompt_drafting,
                   images=[], prepared=[], size=_sized(adj, payload.aspect_ratio, payload.megapixels), crop=None,
                   canvas=payload.aspect_ratio, steps=payload.steps, guidance=payload.guidance,
                   seed=payload.seed)


@app.entrypoint(defaults=_DISTILLED, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def edit(ctx: Context, payload: EditInput, drafter: Drafter, model: Klein, upsampler: Upsampler, out: Outputs,
         tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Edit or compose one to ten ordered reference images with the distilled Klein."""
    images = [asset.read() for asset in payload.reference_images]
    prepared, size, crop = _references(adj, payload.aspect_ratio, payload.megapixels, images)
    return _render(ctx, model, upsampler, drafter, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   draft=payload.prompt_drafting,
                   images=images, prepared=prepared, size=size, crop=crop,
                   canvas=None, steps=payload.steps, guidance=None, seed=payload.seed)


@app.entrypoint(defaults=_BASE, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def edit_base(ctx: Context, payload: EditBaseInput, drafter: Drafter, model: Klein, upsampler: Upsampler, out: Outputs,
              tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Edit or compose one to ten ordered reference images with Klein Base."""
    images = [asset.read() for asset in payload.reference_images]
    prepared, size, crop = _references(adj, payload.aspect_ratio, payload.megapixels, images)
    return _render(ctx, model, upsampler, drafter, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   draft=payload.prompt_drafting,
                   images=images, prepared=prepared, size=size, crop=crop,
                   canvas=None, steps=payload.steps, guidance=payload.guidance, seed=payload.seed)
