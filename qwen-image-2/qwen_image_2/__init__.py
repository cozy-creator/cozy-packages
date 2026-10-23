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
from .quantize import int8, w4a8

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=24 << 30),), accelerator=False)
app.entrypoint(w4a8, weights=(WeightsOutput("w4a8", max_new_bytes=24 << 30),), accelerator=False)


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


#: An edit with no tier keeps its first reference's pixel count inside upstream's 1 MP default and
#: the 4 MP tier.
_EDIT_PIXELS = (Megapixels.MP1 << 20, Megapixels.MP4 << 20)


def _bound(tier: Megapixels | None) -> int:
    return _DEMAND[tier][0] * _DEMAND[tier][1]


def fitted_size(pixels: int, ratio: str, bound: int) -> tuple[int, int, bool]:
    """Upstream's `calculate_dimensions` at `pixels`, held inside the rotary range, 32 px sides and
    `bound` pixels; True when that moved it off upstream's size."""
    w, h = map(int, ratio.split(":"))
    widest = min(_MAX_SIDE**2 / pixels, pixels / 32**2)
    width, height, _ = calculate_dimensions(pixels, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, 32), _MAX_SIDE), min(max(height, 32), _MAX_SIDE)
    while width * height > bound:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != calculate_dimensions(pixels, w / h)[:2]


def output_size(tier: Megapixels, ratio: str) -> tuple[int, int, bool]:
    """The tier's area at `W:H`, inside the rotary range and the tier's demand."""
    return fitted_size(tier << 20, ratio, _bound(tier))


class GenerateInput(msgspec.Struct):
    prompt: str
    # Any W:H. None: the first reference's ratio, else 1:1.
    aspect_ratio: Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")] | None = None
    # None: the first reference's pixel count (1-4 MP), else 4 MP, the model's native 2K.
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    # None: the checkpoint's own schedule (Turbo's 8 steps), else 40.
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] | None = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    reference_images: Annotated[
        list[ImageAsset],
        AssetBound(max_bytes=32 << 20),
        ImagePreparation(max_pixels=4 << 20),
        msgspec.Meta(max_length=10),
    ] = msgspec.field(default_factory=list)

    def resolved_seed(self) -> int:
        return self.seed if self.seed is not None else secrets.randbits(53)

    def size(self, adj: Adjustments, first: tuple[int, int] | None = None) -> tuple[int, int]:
        """The output size; `first` is the first reference's displayed size. What is not given is the
        first reference's: its ratio, and its pixel count inside 1-4 MP. Upstream takes the LAST
        reference's; every Cozy image package takes the first, the image being edited."""
        if first is None:
            ratio, pixels = self.aspect_ratio or "1:1", (self.megapixels or Megapixels.MP4) << 20
        else:
            ratio, own = self.aspect_ratio or f"{first[0]}:{first[1]}", first[0] * first[1]
            pixels = self.megapixels << 20 if self.megapixels else min(max(own, _EDIT_PIXELS[0]), _EDIT_PIXELS[1])
            if not self.megapixels and pixels != own:
                adj.clamp("megapixels", f"{own / (1 << 20):.2f}", f"{pixels / (1 << 20):.2f}",
                          "the first reference's pixel count, inside the 1-4 MP Qwen admits")
        width, height, held = fitted_size(pixels, ratio, _bound(self.megapixels))
        if held:
            adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "nearest size the model admits")
        return width, height


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=20 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int


#: The decoder's alpha never quite reaches 0 or 255: a transparent background sits at 1–4 and an
#: opaque subject at 251–255 (runs 12867, 12878, 12891). Those bands snap to exact.
_ALPHA_NOISE = 4


def decoded_image(decoded: Any) -> Image.Image:
    """The decoder's pixels: RGBA when its alpha has any transparency, else RGB."""
    # The model returns B,C,H,W in [-1,1], just as upstream VaeImageProcessor consumes.
    pixels = decoded[0].detach().float().cpu().permute(1, 2, 0).numpy()
    if not np.isfinite(pixels).all():
        raise ValueError("Qwen Image decoder produced non-finite pixels")
    pixels = ((pixels / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)
    if pixels.shape[-1] == 4:
        alpha = pixels[..., 3]
        alpha[alpha <= _ALPHA_NOISE] = 0
        alpha[alpha >= 255 - _ALPHA_NOISE] = 255
        if alpha.min() == 255:
            pixels = pixels[..., :3]
    return Image.fromarray(np.ascontiguousarray(pixels))


def _generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Both functions' body: the checkpoint decides the schedule."""
    sigmas = model.sample_sigmas
    steps = len(sigmas) if sigmas else payload.steps or 40
    if sigmas and payload.steps not in (None, steps):
        adj.clamp("steps", payload.steps, steps, "the checkpoint's own schedule")
    seed = payload.resolved_seed()
    ctx.raise_if_cancelled()
    reference_images, reference_latents, image_pad_mask = None, None, None
    if payload.reference_images:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.08)):
            # Upstream feeds the VAE RGBA; its vision encoder sees alpha composited over white.
            reference_images = [asset.read(mode="RGBA") for asset in payload.reference_images]
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    width, height = payload.size(adj, reference_images[0].size if reference_images else None)
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
        image = decoded_image(decoded)
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed)


_MEMO = ("qwen_image_2", "qwen_image_2.model")


@app.entrypoint(
    defaults={"model": "qwen-image-2.1@0.1.0"},
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate(
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
def generate_turbo(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Qwen-Image-2.1-Turbo: the same inputs, on its checkpoint's 8-step schedule."""
    return _generate(ctx, payload, model, out, tel, adj)
