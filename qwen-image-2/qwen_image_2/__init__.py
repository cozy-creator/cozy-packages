"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

import secrets
from enum import IntEnum
from typing import Annotated, Any

import msgspec
import numpy as np
from cozy_runtime.author import (
    Adjustments,
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImagePreparation,
    Memoize,
    Outputs,
    Shape,
    Telemetry,
    WeightsOutput,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from PIL import Image

from .model import QwenImage21Model
from .quantize import int8

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=24 << 30),), accelerator=False)


class Megapixels(IntEnum):
    """Output area: 1024² pixels per megapixel, as upstream sizes it."""

    MP1 = 1
    MP2 = 2
    MP4 = 4


#: Upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
_MAX_SIDE = 2048 * 16
#: Admission's bound per tier: its area plus 1/16 for upstream's rounding to 32 px. An omitted
#: tier is decided by the references after admission, so it is bounded at the largest.
_DEMAND: dict[Megapixels | None, tuple[int, int]] = {
    tier: (1088 * (tier or Megapixels.MP4), 1024) for tier in (*Megapixels, None)
}


def _nearest_tier(pixels: int) -> Megapixels:
    return min(Megapixels, key=lambda tier: abs(tier * 1024 * 1024 - pixels))


def output_size(tier: Megapixels, ratio: str) -> tuple[int, int, bool]:
    """Upstream's `calculate_dimensions` at the tier's area, held inside the rotary range, 32 px
    sides and the tier's demand; True when that moved it off upstream's size."""
    area, (w, h) = tier * 1024 * 1024, map(int, ratio.split(":"))
    widest = min(_MAX_SIDE**2 / area, area / 32**2)
    width, height, _ = calculate_dimensions(area, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, 32), _MAX_SIDE), min(max(height, 32), _MAX_SIDE)
    while width * height > _DEMAND[tier][0] * _DEMAND[tier][1]:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != calculate_dimensions(area, w / h)[:2]


class GenerateInput(msgspec.Struct):
    prompt: str
    # Any W:H. None: the last reference's ratio, else 1:1.
    aspect_ratio: Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")] | None = None
    # None: the tier nearest the last reference's pixel count, else 4 MP, the model's native 2K.
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    # None: the checkpoint's own schedule (Turbo's 8 steps), else 40.
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] | None = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(
            max_bytes=32 << 20,
            max_decoded_bytes=20 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
        ImagePreparation(max_pixels=4 << 20),
        msgspec.Meta(max_length=10),
    ] = msgspec.field(default_factory=list)

    def resolved_seed(self) -> int:
        return self.seed if self.seed is not None else secrets.randbits(53)

    def shape(self, reference: tuple[int, int] | None = None) -> tuple[Megapixels, str]:
        """The tier and W:H asked for; `reference` is the last reference's displayed size."""
        if self.megapixels is not None:
            tier = self.megapixels
        else:
            tier = _nearest_tier(reference[0] * reference[1]) if reference else Megapixels.MP4
        ratio = f"{reference[0]}:{reference[1]}" if reference else "1:1"
        return tier, self.aspect_ratio or ratio


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=20 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int


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
    return image


def _generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Both functions' body: the checkpoint decides the schedule."""
    reference_steps = ("vae",) if payload.reference_images else ()
    sigmas = model.sample_sigmas
    steps = len(sigmas) if sigmas else payload.steps or 40
    if sigmas and payload.steps not in (None, steps):
        adj.clamp("steps", payload.steps, steps, "the checkpoint's own schedule")
    ctx.plan(*reference_steps, "text_encoder", *(["transformer"] * steps), "vae")
    seed = payload.resolved_seed()
    ctx.raise_if_cancelled()
    reference_images, reference_latents, image_pad_mask = None, None, None
    if payload.reference_images:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.08)):
            reference_images = [asset.read() for asset in payload.reference_images]
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    tier, ratio = payload.shape(reference_images[-1].size if reference_images else None)
    width, height, held = output_size(tier, ratio)
    if held:
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "nearest size the model admits")
    with tel.stage("encoding prompt", overall_range=(0.08 if reference_images else 0.0, 0.1)):
        if reference_images is None:
            embeds, mask = model.encode(payload.prompt)
        else:
            embeds, mask, image_pad_mask = model.encode_image_prompt(
                payload.prompt, conditioning_images
            )
    with tel.stage("generating image", overall_range=(0.1, 0.9)):
        latents = model.denoise(
            embeds,
            mask,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.1, 0.9)),
            cancel=ctx.raise_if_cancelled,
            reference_images=reference_images,
            reference_latents=reference_latents,
            image_pad_mask=image_pad_mask,
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.9, 0.98)):
        decoded = model.decode(latents, width=width, height=height)
        image = rgb_image(decoded)
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed)


_MEMO = ("qwen_image_2", "qwen_image_2.model")


@app.entrypoint(
    defaults={"model": "qwen-image-2.1@0.1.0"},
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate_image(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Qwen-Image-2.1, the 40-step base model."""
    return _generate(ctx, payload, model, out, tel, adj)


@app.entrypoint(
    defaults={"model": "qwen-image-2.1-turbo@0.1.0"},
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate_image_turbo(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Qwen-Image-2.1-Turbo: the same inputs, on its checkpoint's 8-step schedule."""
    return _generate(ctx, payload, model, out, tel, adj)
