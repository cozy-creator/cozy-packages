"""Qwen-Image-2.1 generation and reference editing with Runtime-owned execution."""

from __future__ import annotations

from typing import Annotated, Any

import msgspec
import numpy as np
from cozy_runtime.author import (
    MEGAPIXEL,
    Adjustments,
    App,
    Context,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    ReferenceImages,
    Telemetry,
    held,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from PIL import Image

from .model import QwenImage21Model
from . import quantize as quantize  # its recipes

app = App()

#: Upstream's rotary table centres at most 2048 tokens of 16 px on an axis.
_MAX_SIDE = 2048 * 16
#: The pixel counts Qwen serves, in megapixels: upstream's 1 MP default to its native 2K (4 MP), the
#: default without references.
_MEGAPIXELS = (1.0, 4.0)
#: The largest output, 4 MP plus 1/16 for upstream's rounding to 32 px.
_BOUND = 1088 * 4 * 1024
#: Steps without a checkpoint schedule (Turbo names its own 8), and the most admitted.
_STEPS, _MAX_STEPS = 40, 100


def fitted_size(pixels: float, ratio: tuple[int, int]) -> tuple[int, int, bool]:
    """Upstream's `calculate_dimensions` at `pixels`, held inside the rotary range, 32 px sides and
    4 MP; True when that moved it off upstream's size."""
    w, h = ratio
    widest = min(_MAX_SIDE**2 / pixels, pixels / 32**2)
    width, height, _ = calculate_dimensions(pixels, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, 32), _MAX_SIDE), min(max(height, 32), _MAX_SIDE)
    while width * height > _BOUND:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != calculate_dimensions(pixels, w / h)[:2]


class GenerateInput(ImageRequest):
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )


def output_size(adj: Adjustments, payload: ImageRequest, first: tuple[int, int] | None) -> tuple[int, int]:
    """What is not given is the first reference's (its ratio and pixel count), else 1:1 at 4 MP;
    the pixel count held to 1-4 MP. Upstream takes the LAST reference's; every Cozy image package
    takes the first, the image being edited."""
    target = payload.target(first, megapixels=_MEGAPIXELS[1])
    megapixels = held(adj, "megapixels", target.pixels / MEGAPIXEL, *_MEGAPIXELS, "the 1-4 MP Qwen-Image serves")
    width, height, moved = fitted_size(megapixels * MEGAPIXEL, target.ratio)
    if moved:
        ratio = f"{target.ratio[0]}:{target.ratio[1]}"
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "nearest size the model admits")
    return width, height


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
) -> ImageResult:
    """Both functions' body: the checkpoint decides the schedule."""
    payload.ignored(adj, "guidance", "negative_prompt", "prompt_upsampling",
                    reason="Qwen-Image-2.1 samples without CFG and has no prompt upsampler")
    sigmas = model.sample_sigmas
    if sigmas:
        steps = len(sigmas)
        if payload.steps not in (None, steps):
            adj.clamp("steps", payload.steps, steps, "the checkpoint's own schedule")
    else:
        steps = held(adj, "steps", payload.steps or _STEPS, 1, _MAX_STEPS)
    seed = payload.drawn_seed()
    ctx.raise_if_cancelled()
    reference_images, reference_latents, image_pad_mask = None, None, None
    if payload.reference_images:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.08)):
            # Upstream feeds the VAE RGBA; its vision encoder sees alpha composited over white.
            reference_images = [asset.read(mode="RGBA") for asset in payload.reference_images]
            conditioning_images, reference_latents = model.encode_reference_images(reference_images)
        ctx.raise_if_cancelled()
    width, height = output_size(adj, payload, reference_images[0].size if reference_images else None)
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
    return ImageResult(asset, width, height, seed, payload.prompt)


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
) -> ImageResult:
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
) -> ImageResult:
    """Qwen-Image-2.1-Turbo: the same inputs, on its checkpoint's 8-step schedule."""
    return _generate(ctx, payload, model, out, tel, adj)
