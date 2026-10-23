"""FLUX.2 [dev], its Turbo LoRA, and FLUX.2 [klein] 9B and 4B: text to image, or an edit of up to ten
ordered references."""

from __future__ import annotations

import math
from functools import partial
from typing import Annotated

import msgspec
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
from PIL import Image

from .model import MAX_PIXELS, MAX_SIDE, MIN_SIDE, PATCH, TURBO_SIGMAS, Flux2, prepare_references, schedule
from . import quantize as quantize  # its recipes
from .turbo import Flux2TurboLoRA
from .upsampler import Flux2Upsampler

app = App()

#: The largest output: 4 MiP plus 1/16 for the 16 px grid.
_BOUND = 1088 * 4 * 1024


class GenerateInput(ImageRequest):
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )


def _snap(value: float) -> int:
    return max(PATCH, round(value / PATCH) * PATCH)


def output_size(ratio: float, area: float) -> tuple[int, int]:
    """`area` pixels at W/H `ratio` on the 16 px grid, held to 64-4096 px sides and 4 MiP."""
    area = min(area, MAX_PIXELS)
    widest = min(MAX_SIDE**2 / area, area / MIN_SIDE**2)
    ratio = min(max(ratio, 1 / widest), widest)
    width = min(max(_snap(math.sqrt(area * ratio)), MIN_SIDE), MAX_SIDE)
    height = min(max(_snap(math.sqrt(area / ratio)), MIN_SIDE), MAX_SIDE)
    while width * height > _BOUND:
        width, height = (width - PATCH, height) if width >= height else (width, height - PATCH)
    return width, height


def _layout(adj: Adjustments, payload: GenerateInput, first: Image.Image | None) -> tuple[int, int]:
    """The output size: the ratio given (else the first reference's, else 1:1) at the size given (else
    the first reference's pixel count, else 1 MP); out of range, the nearest the model admits."""
    target = payload.target(first.size if first else None, megapixels=1.0)
    (w, h), area = target.ratio, held(adj, "megapixels", target.pixels / MEGAPIXEL, MIN_SIDE**2 / MEGAPIXEL,
                                      MAX_PIXELS / MEGAPIXEL, "the model's 4 MP maximum") * MEGAPIXEL
    width, height = output_size(w / h, area)
    if (width, height) != (_snap(math.sqrt(area * w / h)), _snap(math.sqrt(area * h / w))):
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "the nearest size the model admits")
    return width, height


def _generate(ctx: Context, payload: GenerateInput, model: Flux2, turbo: Flux2TurboLoRA | None,
              upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    seed = payload.drawn_seed()
    images = [asset.read() for asset in payload.reference_images]
    width, height = _layout(adj, payload, images[0] if images else None)
    turbo = turbo if model.dev else None
    # The checkpoint's defaults: [dev] 50 steps at guidance 4, its Turbo fal's 8 sigmas at 2.5,
    # [klein] Base 50 steps of CFG 4, [klein] distilled 4 steps without guidance (BFL `util.py`).
    steps = 8 if turbo else 4 if model.distilled else 50
    if payload.steps is not None and turbo and payload.steps != steps:
        adj.clamp("steps", payload.steps, steps, "the Turbo LoRA's own 8-step schedule")
    elif payload.steps is not None:
        steps = held(adj, "steps", payload.steps, 1, 100)
    if model.distilled:
        guidance = None
        payload.ignored(adj, "guidance", reason="FLUX.2 [klein] distilled takes no guidance")
    else:
        default = 2.5 if turbo else 4.0
        guidance = held(adj, "guidance", default if payload.guidance is None else payload.guidance, 1.0, 20.0)
    cfg = not model.dev and guidance is not None and guidance > 1  # distilled: no guidance
    if not cfg:
        payload.ignored(adj, "negative_prompt", reason="only FLUX.2 [klein] Base runs CFG")
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
        negative = model.encode(payload.negative_prompt or "") if cfg else None
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
    return ImageResult(asset, width, height, seed, prompt)


_DEV = "flux-2-dev@0.1.0"
#: The upsampler is [dev]'s text encoder, built and fetched only when `prompt_upsampling` is on; the
#: Turbo LoRA only with [dev].
_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("text_encoder",)}
_TURBO_COMPONENTS = {"model": ("transformer", "text_encoder", "vae"), "upsampler": ("text_encoder",),
                     "turbo_lora": ("model",)}


@app.entrypoint(defaults={"model": _DEV, "upsampler": _DEV}, components=_COMPONENTS, on_demand=("upsampler",),
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, model: Flux2, upsampler: Flux2Upsampler, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The full model: FLUX.2 [dev] (50 steps, guidance 4), or a [klein] Base by `model=` (50 steps, CFG 4)."""
    return _generate(ctx, payload, model, None, upsampler, out, tel, adj)


@app.entrypoint(defaults={"model": _DEV, "turbo_lora": "flux-2-dev-turbo@0.1.0", "upsampler": _DEV},
                components=_TURBO_COMPONENTS, on_demand=("upsampler", "turbo_lora"), memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: GenerateInput, model: Flux2, turbo_lora: Flux2TurboLoRA,
                   upsampler: Flux2Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageResult:
    """The fast model: FLUX.2 [dev] with fal's Turbo LoRA (8 steps, guidance 2.5), or a [klein] by
    `model=` (4 steps)."""
    return _generate(ctx, payload, model, turbo_lora, upsampler, out, tel, adj)
