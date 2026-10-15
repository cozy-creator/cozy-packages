"""FLUX.3 Image Klein text-to-image and reference editing, distilled and base, with BFL's prompt upsampler."""

from __future__ import annotations

import math
import secrets
from enum import Enum, IntEnum
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    MediaDecoder,
    Memoize,
    ModelArtifact,
    Outputs,
    Shape,
    Telemetry,
    WeightsOutput,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource
from PIL import Image

from .klein import MAX_SIDE, MIN_SIDE, PATCH, Klein, prepare_multi_reference, prepare_reference, repack
from .upsampler import Upsampler, prepare

app = App()


@invocable(memoize=True)
async def normalize_upsampler(ctx: Context, *, source: QuantizationSource, tel: Telemetry) -> ModelArtifact:
    """The servable upsampler checkpoint from BFL's raw weights (see README, Prepare the models)."""
    return prepare(source, ctx, tel)


@invocable(memoize=True)
async def normalize(ctx: Context, *, source: QuantizationSource, tel: Telemetry) -> ModelArtifact:
    """A Klein checkpoint as served, from its as-is upload (see README, Prepare the models)."""
    return repack(source, ctx, tel)


app.job(normalize_upsampler, weights=(WeightsOutput("model", max_new_bytes=64 << 20),), accelerator=False)
app.job(normalize, weights=(WeightsOutput("model", max_new_bytes=16 << 20),), accelerator=False)


class AspectRatio(Enum):
    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    PORTRAIT = "3:4"
    PHOTO = "3:2"
    PORTRAIT_PHOTO = "2:3"
    WIDE = "16:9"
    TALL = "9:16"
    ULTRAWIDE = "21:9"
    ULTRATALL = "9:21"


class Megapixels(IntEnum):
    MP1 = 1
    MP2 = 2
    MP4 = 4


# The Qwen-Image-2 buckets: multiples of 32, so of Klein's 16-pixel token grid.
_BUCKETS: dict[tuple[AspectRatio, Megapixels], tuple[int, int]] = {
    (AspectRatio.SQUARE, Megapixels.MP1): (1024, 1024),
    (AspectRatio.LANDSCAPE, Megapixels.MP1): (1216, 896),
    (AspectRatio.PORTRAIT, Megapixels.MP1): (896, 1216),
    (AspectRatio.PHOTO, Megapixels.MP1): (1280, 864),
    (AspectRatio.PORTRAIT_PHOTO, Megapixels.MP1): (864, 1280),
    (AspectRatio.WIDE, Megapixels.MP1): (1376, 768),
    (AspectRatio.TALL, Megapixels.MP1): (768, 1376),
    (AspectRatio.ULTRAWIDE, Megapixels.MP1): (1568, 672),
    (AspectRatio.ULTRATALL, Megapixels.MP1): (672, 1568),
    (AspectRatio.SQUARE, Megapixels.MP2): (1440, 1440),
    (AspectRatio.LANDSCAPE, Megapixels.MP2): (1696, 1280),
    (AspectRatio.PORTRAIT, Megapixels.MP2): (1280, 1696),
    (AspectRatio.PHOTO, Megapixels.MP2): (1792, 1216),
    (AspectRatio.PORTRAIT_PHOTO, Megapixels.MP2): (1216, 1792),
    (AspectRatio.WIDE, Megapixels.MP2): (1952, 1088),
    (AspectRatio.TALL, Megapixels.MP2): (1088, 1952),
    (AspectRatio.ULTRAWIDE, Megapixels.MP2): (2208, 960),
    (AspectRatio.ULTRATALL, Megapixels.MP2): (960, 2208),
    (AspectRatio.SQUARE, Megapixels.MP4): (2048, 2048),
    (AspectRatio.LANDSCAPE, Megapixels.MP4): (2400, 1792),
    (AspectRatio.PORTRAIT, Megapixels.MP4): (1792, 2400),
    (AspectRatio.PHOTO, Megapixels.MP4): (2528, 1696),
    (AspectRatio.PORTRAIT_PHOTO, Megapixels.MP4): (1696, 2528),
    (AspectRatio.WIDE, Megapixels.MP4): (2752, 1536),
    (AspectRatio.TALL, Megapixels.MP4): (1536, 2752),
    (AspectRatio.ULTRAWIDE, Megapixels.MP4): (3136, 1344),
    (AspectRatio.ULTRATALL, Megapixels.MP4): (1344, 3136),
}

_TIER_DEMAND: dict[Megapixels, tuple[int, int]] = {
    tier: max((size for (_, t), size in _BUCKETS.items() if t is tier), key=lambda s: s[0] * s[1])
    for tier in Megapixels
}

class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 16
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True


class GenerateBaseInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] = 50
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 3.5
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True


class EditInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20,
                   media_types=("image/png", "image/jpeg", "image/webp")),
        msgspec.Meta(min_length=1, max_length=10),
    ]
    aspect_ratio: AspectRatio | None = None
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 16
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True


class EditBaseInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20,
                   media_types=("image/png", "image/jpeg", "image/webp")),
        msgspec.Meta(min_length=1, max_length=10),
    ]
    aspect_ratio: AspectRatio | None = None
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] = 50
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 3.5
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True


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


def output_size(aspect_ratio: AspectRatio | None, megapixels: Megapixels, first: Image.Image | None) -> tuple[int, int]:
    """A bucket, or with no aspect ratio the first reference's, at the tier's area on the 16-pixel grid."""
    if aspect_ratio is not None or first is None:
        return _BUCKETS[(aspect_ratio or AspectRatio.SQUARE, megapixels)]
    width, height = _BUCKETS[(AspectRatio.SQUARE, megapixels)]
    ratio = first.width / first.height
    side = lambda v: min(MAX_SIDE, max(MIN_SIDE, round(v / PATCH) * PATCH))  # noqa: E731
    return side(math.sqrt(width * height * ratio)), side(math.sqrt(width * height / ratio))


def _render(
    ctx: Context,
    model: Klein,
    upsampler: Upsampler,
    out: Outputs,
    tel: Telemetry,
    *,
    prompt: str,
    upsample: bool,
    images: list[Image.Image],
    size: tuple[int, int],
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
            caption = upsampler.upsample(prompt, images, canvas)
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    references = None
    if images:
        with tel.stage("encoding reference images", overall_range=(0.15, 0.2)):
            prepared = (
                [prepare_reference(images[0])[0]] if len(images) == 1
                else [prepare_multi_reference(image) for image in images]
            )
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
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed, caption)


_DISTILLED = {
    "model": [{"gpu": "*", "gpus": 1, "lane": "flux-3-klein@0.1.0/original"}],
    "upsampler": [{"gpu": "*", "gpus": 1, "lane": "flux-3-klein-upsampler@0.1.0/original"}],
}
_BASE = {
    "model": [{"gpu": "*", "gpus": 1, "lane": "flux-3-klein-base@0.1.0/original"}],
    "upsampler": [{"gpu": "*", "gpus": 1, "lane": "flux-3-klein-upsampler@0.1.0/original"}],
}
_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("model",)}


@app.entrypoint(defaults=_DISTILLED, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, model: Klein, upsampler: Upsampler,
             out: Outputs, tel: Telemetry) -> ImageOutput:
    """Text to image with the distilled Klein: 16 steps, no guidance."""
    return _render(ctx, model, upsampler, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   images=[], size=output_size(payload.aspect_ratio, payload.megapixels, None),
                   canvas=payload.aspect_ratio.value, steps=payload.steps, guidance=None, seed=payload.seed)


@app.entrypoint(defaults=_BASE, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def generate_base(ctx: Context, payload: GenerateBaseInput, model: Klein, upsampler: Upsampler,
                  out: Outputs, tel: Telemetry) -> ImageOutput:
    """Text to image with Klein Base: 50 steps, CFG 3.5."""
    return _render(ctx, model, upsampler, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   images=[], size=output_size(payload.aspect_ratio, payload.megapixels, None),
                   canvas=payload.aspect_ratio.value, steps=payload.steps, guidance=payload.guidance,
                   seed=payload.seed)


@app.entrypoint(defaults=_DISTILLED, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def edit(ctx: Context, payload: EditInput, model: Klein, upsampler: Upsampler, decoder: MediaDecoder,
         out: Outputs, tel: Telemetry) -> ImageOutput:
    """Edit or compose one to ten ordered reference images with the distilled Klein."""
    images = [decoder.value(asset) for asset in payload.reference_images]
    return _render(ctx, model, upsampler, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   images=images, size=output_size(payload.aspect_ratio, payload.megapixels, images[0]),
                   canvas=None, steps=payload.steps, guidance=None, seed=payload.seed)


@app.entrypoint(defaults=_BASE, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def edit_base(ctx: Context, payload: EditBaseInput, model: Klein, upsampler: Upsampler, decoder: MediaDecoder,
              out: Outputs, tel: Telemetry) -> ImageOutput:
    """Edit or compose one to ten ordered reference images with Klein Base."""
    images = [decoder.value(asset) for asset in payload.reference_images]
    return _render(ctx, model, upsampler, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   images=images, size=output_size(payload.aspect_ratio, payload.megapixels, images[0]),
                   canvas=None, steps=payload.steps, guidance=payload.guidance, seed=payload.seed)
