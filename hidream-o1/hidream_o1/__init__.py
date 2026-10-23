"""HiDream-O1-Image text-to-image and reference editing, full and Dev."""

from __future__ import annotations

import math
from typing import Annotated

import msgspec
from cozy_runtime.author import (
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

from .model import HiDreamO1, Recipe, area_resize, grid_size, reference_size
from . import quantize as quantize  # its recipes

app = App()

#: 1 MP is 1024² pixels. The model was refined at 2048² (tech report §4.1) and upstream serves only
#: about 4 MP: 2560x1440 (3.52 MP) to 2048² (4). At 1024² and below it falls apart (upstream #9).
MP = 1 << 20
MIN_PIXELS, NATIVE_PIXELS = 2560 * 1440, 4 * MP
MIN_SIDE, MAX_SIDE = 256, 4096
FULL_STEPS, DEV_STEPS = 50, 28
GUIDANCE = 5.0
#: The full model's shift; upstream's README runs two or more references at 1.
FULL_SHIFT, SUBJECT_SHIFT = 3.0, 1.0
#: The Dev model's flash sampler: noise scale 7.5, clipped at 2.5 standard deviations.
DEV_FLASH = Recipe("flash", noise=7.5, clip=2.5)
#: Upstream's unconditional caption.
UNCONDITIONAL = " "



class GenerateInput(ImageRequest):
    #: Given, the image edits them, or composes them in order.
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=10)] = msgspec.field(
        default_factory=list
    )
    #: CFG over the first 80% of the steps only (limited-interval guidance): with CFG to the end,
    #: flat areas show a 32 px grid of slightly miscoloured patches. False: CFG on every step, as upstream.
    guidance_interval: bool = True


def fitted(adj: Adjustments, ratio: tuple[int, int], pixels: int) -> tuple[int, int]:
    """`pixels` at `ratio` on the 32 px grid (at most `pixels`), the ratio held to 256-4096 px sides."""
    w, h = ratio
    widest = min(MAX_SIDE**2 / pixels, pixels / MIN_SIDE**2)
    aspect = min(max(w / h, 1 / widest), widest)
    width, height = grid_size(aspect, 1.0, pixels)
    width, height = min(max(width, MIN_SIDE), MAX_SIDE), min(max(height, MIN_SIDE), MAX_SIDE)
    if aspect != w / h:
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "HiDream-O1's 256-4096 px sides")
    return width, height


def layout(
    adj: Adjustments, payload: GenerateInput, images: list[Image.Image]
) -> tuple[tuple[int, int], list[Image.Image]]:
    """The output size and the prepared references. What is not given is the first reference's
    (ratio, pixel count), else 1:1 at 4 MP; pixels are held to 3.52-4 MP. A lone reference given
    neither is upstream's `keep_original_aspect`: resized to the output's area and the output takes
    its size. Otherwise each is resized to the area of a square on the output's longer side, smaller
    with more references (upstream's `generate_image`)."""
    target = payload.target(images[0].size if images else None, megapixels=NATIVE_PIXELS / MP)
    pixels = held(adj, "megapixels", target.pixels / MP, MIN_PIXELS / MP, NATIVE_PIXELS / MP,
                  "HiDream-O1's 3.52-4 MP") * MP
    if len(images) == 1 and payload.aspect_ratio is None and payload.megapixels is None:
        lone = area_resize(images[0], math.sqrt(pixels))
        if MIN_SIDE <= min(lone.size) and max(lone.size) <= MAX_SIDE:
            return lone.size, [lone]
    width, height = fitted(adj, target.ratio, int(pixels))
    side = reference_size(max(width, height), len(images))
    return (width, height), [area_resize(image, side) for image in images]


def _generate(
    ctx: Context, model: HiDreamO1, out: Outputs, tel: Telemetry, adj: Adjustments, *,
    payload: GenerateInput, recipe: Recipe, steps: int, guidance: float, negative: str | None,
    interval: bool = False,
) -> ImageResult:
    payload.ignored(adj, "prompt_upsampling", reason="HiDream-O1 has no prompt upsampler")
    seed = payload.drawn_seed()
    images = [asset.read() for asset in payload.reference_images]
    (width, height), references = layout(adj, payload, images)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.0, 0.98)):
        image = model.generate(
            payload.prompt, negative, references, width=width, height=height, recipe=recipe, steps=steps,
            guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled, interval=interval,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.0, 0.98)),
        )
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageResult(asset, width, height, seed, payload.prompt)


_FULL = {"model": "hidream-o1@0.1.0"}
_DEV = {"model": "hidream-o1-dev@0.1.0"}
_COMPONENTS = {"model": ("model",)}
_MEMO = ("hidream_o1", "hidream_o1.model", "hidream_o1.sampling")


@app.entrypoint(defaults=_FULL, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: HiDreamO1, out: Outputs,
             tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image: 50 UniPC steps with CFG 5.0 over the first 40. The model HiDream recommends
    for editing."""
    shift = SUBJECT_SHIFT if len(payload.reference_images) > 1 else FULL_SHIFT
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0)
    cfg = guidance > 1.0
    negative = (payload.negative_prompt if payload.negative_prompt is not None else UNCONDITIONAL) if cfg else None
    steps = held(adj, "steps", payload.steps or FULL_STEPS, 1, 200)
    return _generate(ctx, model, out, tel, adj, payload=payload, recipe=Recipe("unipc", shift),
                     steps=steps, guidance=guidance, negative=negative, interval=payload.guidance_interval)


@app.entrypoint(defaults=_DEV, components=_COMPONENTS, memoize=Memoize(requires=["seed"]),
                memo_dependencies=_MEMO)
def generate_turbo(ctx: Context, payload: GenerateInput, model: HiDreamO1, out: Outputs,
                   tel: Telemetry, adj: Adjustments) -> ImageResult:
    """HiDream-O1-Image-Dev: 28 distilled steps, no CFG. A one-reference edit samples flow-match Euler."""
    if payload.steps not in (None, DEV_STEPS):
        adj.clamp("steps", payload.steps, DEV_STEPS, "the Dev model's distilled schedule")
    payload.ignored(adj, "guidance", "negative_prompt", reason="HiDream-O1-Dev samples without CFG")
    if not payload.guidance_interval:
        adj.warn("guidance_interval", "ignored: HiDream-O1-Dev samples without CFG")
    recipe = Recipe("euler") if len(payload.reference_images) == 1 else DEV_FLASH
    return _generate(ctx, model, out, tel, adj, payload=payload, recipe=recipe, steps=DEV_STEPS,
                     guidance=0.0, negative=None)
