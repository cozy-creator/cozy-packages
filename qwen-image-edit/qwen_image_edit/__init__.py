"""Qwen-Image-Edit-2511 image editing and generation, full and Lightning, Runtime-executed."""

from __future__ import annotations

import math
import secrets
from enum import IntEnum
from functools import partial
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
    Shape,
    Telemetry,
    WeightsOutput,
)
from PIL import Image

from .model import PATCH, QwenImageEdit, QwenImageEditLoRA
from .quantize import int8, nvfp4

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=40 << 30),), accelerator=False)
app.entrypoint(nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=32 << 30),), accelerator=False)


class Megapixels(IntEnum):
    """Output area, 1024² pixels per megapixel."""

    MP1 = 1
    MP2 = 2
    MP4 = 4


#: The sizes the model serves: 256² to 2 MiP (Qwen-Image trains from 256² up to 1328², 1.7 MiP),
#: sides of 128-4096 px on its 16 px grid. Text-to-image defaults to upstream's 1024² edit area.
MIN_SIDE, MAX_SIDE = 128, 4096
MIN_PIXELS, MAX_PIXELS = 256 * 256, 2 << 20
NATIVE_PIXELS = 1 << 20
#: Admission's bound per tier, 1/16 over for the grid; a size the references decide is at most 2 MiP.
_DEMAND: dict[Megapixels | None, tuple[int, int]] = {
    tier: (1088 * min(tier or 2, 2), 1024) for tier in (*Megapixels, None)
}
_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
_REFERENCES = Annotated[
    list[ImageAsset],
    AssetBound(max_bytes=32 << 20),
    ImagePreparation(max_pixels=4 << 20),
    msgspec.Meta(max_length=10),
]
_SEED = Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)]


def output_size(pixels: int, ratio: float) -> tuple[int, int, int, float]:
    """(width, height) on the 16 px grid for `pixels` at `ratio`, with the area and ratio actually
    served: the area held to 256²-2 MiP, the ratio to what keeps both sides within 128-4096 px."""
    area = min(max(pixels, MIN_PIXELS), MAX_PIXELS)
    widest = min(MAX_SIDE**2 / area, area / MIN_SIDE**2)
    held = min(max(ratio, 1 / widest), widest)
    width = round(math.sqrt(area * held) / PATCH) * PATCH
    height = round(math.sqrt(area / held) / PATCH) * PATCH
    return min(max(width, MIN_SIDE), MAX_SIDE), min(max(height, MIN_SIDE), MAX_SIDE), area, held


def _megapixels(pixels: int) -> float:
    return round(pixels / (1 << 20), 2)


def sized(
    adj: Adjustments, aspect_ratio: str | None, megapixels: Megapixels | None, first: Image.Image | None
) -> tuple[int, int]:
    """The output size: an explicit ratio or tier, else the first reference's ratio and pixel
    count, else 1:1 at 1 MiP. Anything outside the model's range is a `clamp` adjustment."""
    if aspect_ratio is not None:
        w, h = map(int, aspect_ratio.split(":"))
    else:
        w, h = first.size if first is not None else (1, 1)
    if megapixels is not None:
        pixels = int(megapixels) << 20
    else:
        pixels = first.width * first.height if first is not None else NATIVE_PIXELS
    width, height, area, ratio = output_size(pixels, w / h)
    if area != pixels:
        adj.clamp("megapixels", megapixels or _megapixels(pixels), _megapixels(area),
                  "the model serves 0.06-2 MiP")
    if ratio != w / h:
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "nearest ratio the model serves")
    return width, height


class GenerateInput(msgspec.Struct):
    prompt: str
    #: Given, the image edits or composes them, in order ("Picture 1", "Picture 2", ...).
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    #: Any W:H. None: the first reference's, else 1:1.
    aspect_ratio: _RATIO | None = None
    #: None: the first reference's pixel count, else 1 MiP.
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    #: None: 40. At least 2: the schedule stretches its last step to the terminal sigma.
    steps: Annotated[int, msgspec.Meta(ge=2, le=100)] | None = None
    seed: _SEED | None = None
    #: True CFG scale; 1 runs no negative pass.
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 4.0
    negative_prompt: str = " "


class TurboInput(msgspec.Struct):
    prompt: str
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    aspect_ratio: _RATIO | None = None
    megapixels: Annotated[Megapixels | None, Shape(pixels=_DEMAND)] = None
    #: None: 8, the steps this Lightning LoRA is distilled for.
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] | None = None
    seed: _SEED | None = None


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=32 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int
    #: The caption the image was made from: the prompt as given.
    prompt: str


def _render(
    ctx: Context,
    model: QwenImageEdit,
    lora: QwenImageEditLoRA | None,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    payload: GenerateInput | TurboInput,
    *,
    steps: int,
    guidance: float,
    negative: str,
) -> ImageOutput:
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    # Upright, with any transparency over white, as upstream's RGB references.
    images = [asset.read() for asset in payload.reference_images]
    width, height = sized(adj, payload.aspect_ratio, payload.megapixels, images[0] if images else None)
    ctx.raise_if_cancelled()
    references = None
    if images:
        with tel.stage("encoding reference images", overall_range=(0.0, 0.05)):
            references = model.encode_references(images)
    with tel.stage("encoding prompt", overall_range=(0.05, 0.1)):
        embeds, mask = model.encode(payload.prompt, images)
        unconditional = model.encode(negative, images) if guidance > 1 else None
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.1, 0.95)):
        denoise = model.denoise if lora is None else partial(lora.denoise, model)
        latents = denoise(
            embeds, mask, unconditional, references, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed, cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.1, 0.95)),
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(latents, width=width, height=height)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, width, height, seed, payload.prompt)


_MEMO = ("qwen_image_edit", "qwen_image_edit.model")


@app.entrypoint(
    defaults={"model": "qwen-image-edit-2511@0.1.0"},
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImageEdit,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Qwen-Image-Edit-2511: 40 steps with true CFG 4.0. Edits when given references."""
    return _render(ctx, model, None, out, tel, adj, payload, steps=payload.steps or 40,
                   guidance=payload.guidance, negative=payload.negative_prompt)


@app.entrypoint(
    defaults={
        "model": "qwen-image-edit-2511@0.1.0",
        "turbo_lora": "qwen-image-edit-2511-lightning-lora@0.1.0",
    },
    memoize=Memoize(requires=["seed"]),
    memo_dependencies=_MEMO,
)
def generate_turbo(
    ctx: Context,
    payload: TurboInput,
    model: QwenImageEdit,
    turbo_lora: QwenImageEditLoRA,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Qwen-Image-Edit-2511 with lightx2v's 8-step Lightning LoRA: 8 steps, no CFG."""
    return _render(ctx, model, turbo_lora, out, tel, adj, payload, steps=payload.steps or 8,
                   guidance=1.0, negative="")
