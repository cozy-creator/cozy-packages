"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

import secrets
from enum import Enum, IntEnum
from typing import Annotated, Any, Literal

import msgspec
import numpy as np
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    MediaDecoder,
    Outputs,
    Shape,
    Telemetry,
)
from cozy_runtime.models.qwen_image21 import QwenImage21Model
from PIL import Image

app = App()
Background = Literal["normal", "white"]
CacheMode = Literal["enabled", "disabled"]


class AspectRatio(Enum):
    """Explicit output buckets; unknown ratios are refused, never rounded."""

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
    """Nominal image-area tiers; exact dimensions are listed in the package README."""

    MP1 = 1
    MP2 = 2
    MP4 = 4


# Tier 4 uses Qwen2.1 native recommendations; lower tiers scale and snap to 32px.
# Ultrawide/tall extend the grid at the same area and latent patch stride.
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
    tier: max(
        (size for (_, t), size in _BUCKETS.items() if t is tier),
        key=lambda size: size[0] * size[1],
    )
    for tier in Megapixels
}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 40
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    background: Background = "normal"
    kv_cache_mode: Literal["auto", "enabled", "disabled"] = "auto"
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(
            max_bytes=32 << 20,
            max_decoded_bytes=20 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
        msgspec.Meta(max_length=10),
    ] = msgspec.field(default_factory=list)

    def resolved_seed(self) -> int:
        return self.seed if self.seed is not None else secrets.randbits(53)

    def dimensions(self) -> tuple[int, int]:
        return _BUCKETS[(self.aspect_ratio, self.megapixels)]


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=20 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int
    kv_cache_mode: CacheMode


def reference_prompt(prompt: str, background: Background) -> str:
    if background == "white":
        return (
            f"{prompt}\nSingle subject isolated against a seamless pure white studio background. "
            "Show the whole subject clearly, with no scenery, text, labels or other people."
        )
    return prompt


def rgb_image(decoded: Any) -> Image.Image:
    """Preserve native RGBA semantics and flatten alpha over white for H3 references."""
    # The model returns B,C,H,W in [-1,1], just as upstream VaeImageProcessor consumes.
    pixels = decoded[0].detach().float().cpu().permute(1, 2, 0).numpy()
    if not np.isfinite(pixels).all():
        raise ValueError("Qwen Image decoder produced non-finite pixels")
    pixels = ((pixels / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)
    image = Image.fromarray(pixels)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    if image.mode != "RGB":
        raise ValueError(f"Qwen Image decoder produced unsupported {image.mode} pixels")
    return image


@app.entrypoint(defaults={"model": [{"gpu": "*", "gpus": 1, "lane": "reference-image@0.1.0/original"}]})
def generate_image(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    seed = payload.resolved_seed()
    cache_mode: CacheMode | None = None

    def selected_cache(enabled: bool) -> None:
        nonlocal cache_mode
        cache_mode = "enabled" if enabled else "disabled"

    width, height = payload.dimensions()
    ctx.raise_if_cancelled()
    reference_images, reference_latents, image_pad_mask = None, None, None
    if payload.reference_images:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.08)):
            reference_images = [decoder.value(asset) for asset in payload.reference_images]
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.08 if reference_images else 0.0, 0.1)):
        prompt = reference_prompt(payload.prompt, payload.background)
        if reference_images is None:
            embeds, mask = model.encode(prompt)
        else:
            embeds, mask, image_pad_mask = model.encode_image_prompt(prompt, conditioning_images)
    with tel.stage("generating image", overall_range=(0.1, 0.9)):
        latents = model.denoise(
            embeds,
            mask,
            width=width,
            height=height,
            steps=payload.steps,
            seed=seed,
            on_step=tel.step_callback(
                payload.steps, stage="generating image", overall_range=(0.1, 0.9)
            ),
            cancel=ctx.raise_if_cancelled,
            reference_images=reference_images,
            reference_latents=reference_latents,
            image_pad_mask=image_pad_mask,
            use_kv_cache=None
            if payload.kv_cache_mode == "auto"
            else payload.kv_cache_mode == "enabled",
            on_cache_mode=selected_cache,
        )
    if cache_mode is None:
        raise RuntimeError("Qwen denoising did not report its request cache mode")
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.9, 0.98)):
        decoded = model.decode(latents, width=width, height=height)
        image = rgb_image(decoded)
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed, cache_mode)
