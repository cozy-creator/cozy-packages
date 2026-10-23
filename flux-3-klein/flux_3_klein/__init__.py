"""FLUX.3 Image Klein text-to-image and reference editing, distilled and base, with BFL's prompt upsampler."""

from __future__ import annotations

import json
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
    InvalidRequest,
    Memoize,
    Outputs,
    Shape,
    Telemetry,
    WeightsOutput,
)
from PIL import Image, ImageOps

from .klein import (
    MATCH_MAX_PIXELS,
    MAX_SIDE,
    MIN_SIDE,
    Klein,
    prepare_multi_reference,
    prepare_reference,
    reference_canvas,
)
from .quantize import drafter_head, int8, nvfp4, w4a8
from .upsampler import Drafter, Row, Upsampler, canvas_box, is_finished, with_table

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=12 << 30),), accelerator=False)
app.entrypoint(nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=12 << 30),), accelerator=False)
app.entrypoint(w4a8, weights=(WeightsOutput("w4a8", max_new_bytes=12 << 30),), accelerator=False)
app.entrypoint(drafter_head, weights=(WeightsOutput("bf16", max_new_bytes=1 << 30),), accelerator=False)


class Megapixels(IntEnum):
    MP1 = 1
    MP2 = 2
    MP4 = 4


#: Admission's bound per tier: its area (1024² per megapixel) plus 1/16 for rounding to 32 px. No
#: tier is decided after admission (by the references, at most 4 MiP), so it is bounded at the largest.
_DEMAND: dict[Megapixels | None, tuple[int, int]] = {
    tier: (1088 * (tier or Megapixels.MP4), 1024) for tier in (*Megapixels, None)
}
_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]


def _rounded(area: int, ratio: float) -> tuple[int, int]:
    """Qwen-Image's `calculate_dimensions`: the area at that ratio, each side rounded to 32 px."""
    width = math.sqrt(area * ratio)
    return round(width / 32) * 32, round(width / ratio / 32) * 32


def _bound(tier: Megapixels | None) -> int:
    return _DEMAND[tier][0] * _DEMAND[tier][1]


def fitted_size(pixels: int, ratio: str, bound: int) -> tuple[int, int, bool]:
    """`pixels` at `W:H`, held inside BFL's 64-4096 px sides and `bound` pixels; True when that moved
    it off the rounded size."""
    w, h = map(int, ratio.split(":"))
    widest = min(MAX_SIDE**2 / pixels, pixels / MIN_SIDE**2)
    width, height = _rounded(pixels, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, MIN_SIDE), MAX_SIDE), min(max(height, MIN_SIDE), MAX_SIDE)
    while width * height > bound:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != _rounded(pixels, w / h)


def output_size(megapixels: Megapixels, ratio: str) -> tuple[int, int, bool]:
    """The tier's area at `W:H`, inside BFL's sides and the tier's demand."""
    return fitted_size(megapixels << 20, ratio, _bound(megapixels))


def _sized(adj: Adjustments, ratio: str, pixels: int, bound: int) -> tuple[int, int]:
    width, height, held = fitted_size(pixels, ratio, bound)
    if held:
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "nearest size the model admits")
    return width, height


def _megapixels(pixels: int) -> str:
    return f"{pixels / (1 << 20):.2f}"


_REFERENCES = Annotated[
    list[ImageAsset],
    AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20, media_types=("image/png", "image/jpeg", "image/webp")),
    msgspec.Meta(max_length=10),
]
_UNIT = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]
#: `[x0, y0, x1, y1]`: corners, 0-1 of the output canvas, origin top-left, x0 < x1, y0 < y1.
Box = Annotated[list[_UNIT], msgspec.Meta(min_length=4, max_length=4)]


class Region(msgspec.Struct, forbid_unknown_fields=True):
    box: Box
    #: What goes in the box.
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: The literal text to render there.
    text: Annotated[str, msgspec.Meta(min_length=1)] | None = None


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or composes them, in order.
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    #: Any W:H. None: the first reference's, else 1:1.
    aspect_ratio: _RATIO | None = None
    #: None: the first reference's pixel count (at most 4 MP), else 1 MP.
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    #: One per reference, in order: where it goes in the output, or null for none.
    reference_boxes: list[Box | None] | None = None
    #: Elements and rendered text placed by box, described in words.
    regions: list[Region] | None = None
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] = 50
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 4.0
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    prompt_upsampling: bool = True
    #: Upsample with Qwen3.5-9B's MTP head drafting tokens (1.1-1.2× faster a caption on a 4090);
    #: its first use downloads the 0.5 GB head.
    prompt_drafting: bool = True


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or composes them, in order.
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    #: Any W:H. None: the first reference's, else 1:1.
    aspect_ratio: _RATIO | None = None
    #: None: the first reference's pixel count (at most 4 MP), else 1 MP.
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    #: One per reference, in order: where it goes in the output, or null for none.
    reference_boxes: list[Box | None] | None = None
    #: Elements and rendered text placed by box, described in words.
    regions: list[Region] | None = None
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 16
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
    rows: list[Row],
    steps: int,
    guidance: float | None,
    seed: int | None,
) -> ImageOutput:
    seed = seed if seed is not None else secrets.randbits(53)
    width, height = size
    caption = with_table(prompt, rows) if rows else prompt
    if upsample and (rows or not is_finished(prompt)):
        table = json.dumps(rows, ensure_ascii=False) if rows else ""
        with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
            def run() -> str:
                return upsampler.upsample(prompt, images, canvas, table)
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


def _layout(
    adj: Adjustments, ratio: str | None, megapixels: Megapixels | None, images: list[Image.Image], placed: bool
) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int, int, int] | None, str | None]:
    """The prepared references, the output size, the crop after decoding and the upsampler's canvas.
    No references, or references placed by box (elements, not the canvas): the ratio (1:1) at the
    tier (1 MP). Otherwise what is not given is the first reference's: given neither, its own size,
    fitted to the 4 MiP Klein admits and generated on the 16 px grid, then cropped back; given one,
    its ratio or its pixel count."""
    if len(images) == 1:
        prepared = [prepare_reference(images[0])[0]]
    else:
        prepared = [prepare_multi_reference(image) for image in images]
    if not images or placed:
        ratio, tier = ratio or "1:1", megapixels or Megapixels.MP1
        return prepared, _sized(adj, ratio, tier << 20, _bound(tier)), None, None if images else ratio
    first = ImageOps.exif_transpose(images[0]).size
    if ratio is None and megapixels is None:
        fit, canvas = reference_canvas(*first)
        if fit != first:
            adj.clamp("megapixels", _megapixels(first[0] * first[1]), _megapixels(fit[0] * fit[1]),
                      "the first reference's size, fitted to the 4 MiP and 4096 px sides Klein admits")
        left, top = (canvas[0] - fit[0]) // 2, (canvas[1] - fit[1]) // 2
        return prepared, canvas, None if fit == canvas else (left, top, left + fit[0], top + fit[1]), None
    if megapixels is None:
        pixels = min(max(first[0] * first[1], MIN_SIDE**2), MATCH_MAX_PIXELS)
        if pixels != first[0] * first[1]:
            adj.clamp("megapixels", _megapixels(first[0] * first[1]), _megapixels(pixels),
                      "the first reference's pixel count, inside the sizes Klein admits")
    else:
        pixels = megapixels << 20
    return prepared, _sized(adj, ratio or f"{first[0]}:{first[1]}", pixels, _bound(megapixels)), None, None


def _checked(field: str, box: list[float]) -> list[int]:
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        raise InvalidRequest(f"a box is [x0, y0, x1, y1] with x0 < x1 and y0 < y1, got {box}",
                             code="invalid_box", fields=(field,))
    return canvas_box(box)


def layout_rows(payload: GenerateInput | TurboInput) -> list[Row]:
    """Klein's table rows for the request's boxes: a placed reference is `ref_image_<i>` at its box (the
    upsampler describes it from the image); a region is a new element, its text quoted in the desc."""
    boxes, references = payload.reference_boxes, len(payload.reference_images)
    if boxes is not None and len(boxes) != references:
        raise InvalidRequest(f"reference_boxes has {len(boxes)} entries for {references} reference_images",
                             code="reference_boxes_length", fields=("reference_boxes",))
    rows: list[Row] = []
    for i, box in enumerate(boxes or ()):
        if box is not None:
            target = _checked(f"reference_boxes.{i}", box)
            rows.append({"id": f"obj_{len(rows) + 1}", "from": f"ref_image_{i}", "src_bbox": None,
                         "tgt_bbox": target, "desc": ""})
    for i, region in enumerate(payload.regions or ()):
        grid = _checked(f"regions.{i}.box", region.box)
        desc = f'{region.prompt.strip()} "{region.text}"' if region.text else region.prompt.strip()
        row_id = f"obj_{len(rows) + 1}"
        rows.append({"id": row_id, "from": "target", "src_bbox": None, "tgt_bbox": grid, "desc": desc}
                    if references else {"id": row_id, "bbox": grid, "desc": desc})
    return rows


def _generate(ctx: Context, payload: GenerateInput | TurboInput, guidance: float | None, drafter: Drafter,
              model: Klein, upsampler: Upsampler, out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    rows = layout_rows(payload)
    images = [asset.read() for asset in payload.reference_images]
    placed = any(box is not None for box in payload.reference_boxes or ())
    prepared, size, crop, canvas = _layout(adj, payload.aspect_ratio, payload.megapixels, images, placed)
    return _render(ctx, model, upsampler, drafter, out, tel, prompt=payload.prompt, upsample=payload.prompt_upsampling,
                   draft=payload.prompt_drafting, images=images, prepared=prepared, size=size, crop=crop,
                   canvas=canvas, rows=rows, steps=payload.steps, guidance=guidance, seed=payload.seed)


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


@app.entrypoint(defaults=_BASE, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Klein Base, 50 steps with CFG 4: text to image, or an edit of up to ten ordered references."""
    return _generate(ctx, payload, payload.guidance, drafter, model, upsampler, out, tel, adj)


@app.entrypoint(defaults=_DISTILLED, components=_COMPONENTS, on_demand=_ON_DEMAND,
                memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: TurboInput, drafter: Drafter, model: Klein, upsampler: Upsampler,
                   out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """The distilled Klein, 16 steps, no guidance: text to image, or an edit of up to ten ordered references."""
    return _generate(ctx, payload, None, drafter, model, upsampler, out, tel, adj)
