"""MOSS-Image (OpenMOSS) text-to-image, with layout boxes for objects and rendered text."""

from __future__ import annotations

import json
import math
import secrets
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImagePreparation,
    Memoize,
    Outputs,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
)

from .model import MossImage
from .quantize import int8

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=16 << 30),), accelerator=False)

MP = 1 << 20
GRID = 16
#: The user guide: best at 1-2 MP, extrapolates from 0.25 MP (512²) to 4 MP; the pipeline's default 1024².
MIN_PIXELS, NATIVE_PIXELS, MAX_PIXELS = MP // 4, MP, 4 * MP
MIN_SIDE, MAX_SIDE = 256, 4096
STEPS, MOST_STEPS = 50, 200
GUIDANCE = 3.0

_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
_SEED = Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)]
#: `[x0, y0, x1, y1]`: corners normalized 0-1 on the output canvas, origin top-left.
Box = tuple[float, float, float, float]


class Region(msgspec.Struct):
    box: Box
    #: What goes in the box.
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: The literal text to render in the box (`\n` breaks a line).
    text: Annotated[str, msgspec.Meta(min_length=1)] | None = None


class GenerateInput(msgspec.Struct):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: MOSS-Image takes no reference images: warned and ignored.
    reference_images: Annotated[
        list[ImageAsset], AssetBound(max_bytes=32 << 20), ImagePreparation(max_pixels=16 << 20),
        msgspec.Meta(max_length=16),
    ] = msgspec.field(default_factory=list)
    #: Placement of `reference_images`: warned and ignored, as they are.
    reference_boxes: list[Box | None] = msgspec.field(default_factory=list)
    #: Objects and text placed in boxes; the prompt then describes the whole image.
    regions: Annotated[list[Region], msgspec.Meta(max_length=64)] = msgspec.field(default_factory=list)
    #: Any W:H. None: 1:1.
    aspect_ratio: _RATIO | None = None
    #: None: 1. Held to 0.25-4.
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    #: None: 50.
    steps: Annotated[int, msgspec.Meta(ge=1)] | None = None
    #: None: 3.0. 1 runs without CFG.
    guidance: Annotated[float, msgspec.Meta(ge=0)] | None = None
    #: None: empty, upstream's default.
    negative_prompt: str | None = None
    seed: _SEED | None = None
    #: MOSS-Image has no prompt upsampler: warned and ignored.
    prompt_upsampling: bool | None = None


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset, AssetBound(max_bytes=64 << 20, max_decoded_bytes=256 << 20, media_types=("image/png",))
    ]
    width: int
    height: int
    seed: int
    #: The prompt the image was made from: the layout JSON when `regions` are given.
    prompt: str


def held[T: (int, float)](adj: Adjustments, field: str, requested: T, low: T, high: T, reason: str) -> T:
    applied = min(max(requested, low), high)
    if applied != requested:
        adj.clamp(field, round(requested, 3) if isinstance(requested, float) else requested, applied, reason)
    return applied


def output_size(adj: Adjustments, ratio: str | None, megapixels: float | None) -> tuple[int, int]:
    """The ratio (else 1:1) at the pixel count (else 1 MP), held to 0.25-4 MP and 256-4096 px sides,
    on the 16 px grid."""
    w, h = map(int, ratio.split(":")) if ratio else (1, 1)
    pixels = held(adj, "megapixels", megapixels if megapixels is not None else 1.0, MIN_PIXELS / MP,
                  MAX_PIXELS / MP, "MOSS-Image's 0.25-4 MP") * MP
    widest = min(MAX_SIDE**2 / pixels, pixels / MIN_SIDE**2)
    aspect = min(max(w / h, 1 / widest), widest)
    width = min(max(round(math.sqrt(pixels * aspect) / GRID) * GRID, MIN_SIDE), MAX_SIDE)
    height = min(max(round(math.sqrt(pixels / aspect) / GRID) * GRID, MIN_SIDE), MAX_SIDE)
    if aspect != w / h:
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "MOSS-Image's 256-4096 px sides")
    return width, height


def pixel_box(box: Box, width: int, height: int, where: str) -> list[int]:
    """A normalized `[x0, y0, x1, y1]` as MOSS's `yxyx` pixel box from the canvas centre: left and top
    floored, right and bottom ceiled, as upstream's layout compiler encloses a box."""
    x0, y0, x1, y1 = box
    if not (all(math.isfinite(v) for v in box) and 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise UnsupportedInput(f"{where} must be [x0, y0, x1, y1] with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1",
                               code="regions")
    return [math.floor(round(y0 * height, 9)) - height // 2, math.floor(round(x0 * width, 9)) - width // 2,
            math.ceil(round(y1 * height, 9)) - height // 2, math.ceil(round(x1 * width, 9)) - width // 2]


def layout_prompt(prompt: str, regions: list[Region], width: int, height: int) -> str:
    """The prompt as given, or with regions MOSS's layout JSON: the prompt is the scene's caption, each
    region an item (`text` when it renders text, else `object`), compact as upstream compiles it."""
    if not regions:
        return prompt
    items: list[dict[str, object]] = []
    for index, region in enumerate(regions):
        item: dict[str, object] = {"label": "object" if region.text is None else "text", "caption": region.prompt}
        if region.text is not None:
            item["text"] = region.text
        item["bbox"] = pixel_box(region.box, width, height, f"regions[{index}].box")
        items.append(item)
    layout = {"summary": {"caption": prompt.strip()}, "items": items, "bbox_coord_space": "abs_centralized",
              "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def _ignored(adj: Adjustments, payload: GenerateInput) -> None:
    if payload.reference_images:
        adj.warn("reference_images", "ignored: MOSS-Image generates from text alone")
    if payload.reference_boxes:
        adj.warn("reference_boxes", "ignored: MOSS-Image takes no reference images; place objects with regions")
    if payload.prompt_upsampling:
        adj.warn("prompt_upsampling", "ignored: MOSS-Image has no prompt upsampler")


_MEMO = ("moss_image", "moss_image.model")


@app.entrypoint(defaults={"model": "moss-image@0.1.0"}, components={"model": ("transformer", "text_encoder", "vae")},
                memoize=Memoize(requires=["seed"]), memo_dependencies=_MEMO)
def generate(ctx: Context, payload: GenerateInput, model: MossImage, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> ImageOutput:
    """MOSS-Image SFT: 50 Euler steps at CFG 3 (the negative branch skips block 22), with `regions` laid
    out as upstream's layout JSON."""
    _ignored(adj, payload)
    steps = held(adj, "steps", payload.steps or STEPS, 1, MOST_STEPS, "MOSS-Image's 1-200 steps")
    guidance = held(adj, "guidance", GUIDANCE if payload.guidance is None else payload.guidance, 1.0, 20.0,
                    "MOSS-Image's CFG 1-20")
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    width, height = output_size(adj, payload.aspect_ratio, payload.megapixels)
    prompt = layout_prompt(payload.prompt, payload.regions, width, height)
    prompts = [prompt] if guidance == 1.0 else [prompt, payload.negative_prompt or ""]
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.0, 0.05)):
        encoded = model.encode(prompts)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.05, 0.97)):
        latents = model.denoise(
            encoded[0], encoded[1] if len(encoded) > 1 else None, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.05, 0.97)),
        )
    with tel.stage("decoding image", overall_range=(0.97, 0.99)):
        image = model.decode(latents)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed, prompt)
