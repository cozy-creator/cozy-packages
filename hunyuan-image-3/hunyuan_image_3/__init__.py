"""Tencent's HunyuanImage 3.0 Instruct: text to image, or an edit or fusion of up to three ordered references,
with the model's own think-and-rewrite stage. `generate` is Instruct, `generate_turbo` Instruct-Distil."""

from __future__ import annotations

import secrets
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    Adjustments,
    App,
    AssetBound,
    Context,
    ImageAsset,
    Memoize,
    Outputs,
    Telemetry,
    WeightsOutput,
)

from .model import MAX_NEW_TOKENS, HunyuanImage3, caption
from .quantize import int8, nvfp4
from .tencent.tokenization_hunyuan_image_3 import Resolution, ResolutionGroup

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=96 << 30),), accelerator=False)
app.entrypoint(nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=64 << 30),), accelerator=False)

#: The model's sizes, as Tencent's image processor builds them (`image_base_size` 1024): 33 ratios from
#: 1:4 to 4:1 on a 64 px step with width + height = 2048, plus four common ones; all about 1 MP.
SIZES = ResolutionGroup(base_size=1024, align=16, extra_resolutions=[
    Resolution("1024x768"), Resolution("1280x720"), Resolution("768x1024"), Resolution("720x1280"),
])
_WIDEST = max(r.w / r.h for r in SIZES.data)
_TALLEST = min(r.w / r.h for r in SIZES.data)
_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
_REFERENCES = Annotated[
    list[ImageAsset],
    AssetBound(max_bytes=32 << 20, max_decoded_bytes=64 << 20, media_types=("image/png", "image/jpeg", "image/webp")),
    msgspec.Meta(max_length=3),
]
_SEED = Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)]


def requested_size(adj: Adjustments, ratio: str) -> tuple[int, int]:
    """The model size nearest `W:H` (Tencent's `get_target_size`); a ratio past 1:4 or 4:1 is a `clamp`."""
    w, h = map(int, ratio.split(":"))
    width, height = SIZES.get_target_size(w, h)
    if not _TALLEST <= w / h <= _WIDEST:
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "the model's sizes run from 1:4 to 4:1")
    return width, height


def check_pixels(adj: Adjustments, megapixels: float | None, size: tuple[int, int]) -> None:
    """The model makes about 1 MP: `megapixels` more than 10% off the size made is a `clamp`."""
    made = size[0] * size[1]
    if megapixels is not None and abs(megapixels * (1 << 20) - made) > 0.1 * made:
        adj.clamp("megapixels", megapixels, round(made / (1 << 20), 2), "HunyuanImage 3.0 makes about 1 MP")


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or fuses them, in order.
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    #: Any W:H, served at the model's nearest size. None: the model picks its size.
    aspect_ratio: _RATIO | None = None
    #: The model makes ~1 MP only; another value is reported as a clamp.
    megapixels: Annotated[float, msgspec.Meta(gt=0, le=64)] | None = None
    #: None: 50.
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] | None = None
    #: Classifier-free guidance; the conditional and unconditional halves always run together.
    guidance: Annotated[float, msgspec.Meta(gt=1.0, le=20.0)] = 2.5
    seed: _SEED | None = None
    #: The model thinks, then rewrites the prompt, before it draws.
    prompt_upsampling: bool = True


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or fuses them, in order.
    reference_images: _REFERENCES = msgspec.field(default_factory=list)
    #: Any W:H, served at the model's nearest size. None: the model picks its size.
    aspect_ratio: _RATIO | None = None
    #: The model makes ~1 MP only; another value is reported as a clamp.
    megapixels: Annotated[float, msgspec.Meta(gt=0, le=64)] | None = None
    #: None: 8.
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] | None = None
    #: The guidance the distilled model is conditioned on.
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = 2.5
    seed: _SEED | None = None
    #: The model thinks, then rewrites the prompt, before it draws.
    prompt_upsampling: bool = True


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=64 << 20, max_decoded_bytes=64 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int
    #: The caption the image was generated from: the model's rewrite when `prompt_upsampling` is on.
    prompt: str


def _generate(ctx: Context, payload: GenerateInput | TurboInput, steps: int, model: HunyuanImage3, out: Outputs,
              tel: Telemetry, adj: Adjustments) -> ImageOutput:
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    references = [asset.read() for asset in payload.reference_images]
    # No aspect ratio: the model picks its size, as Tencent's `image_size="auto"`.
    size = requested_size(adj, payload.aspect_ratio) if payload.aspect_ratio else None
    cot = None
    if payload.prompt_upsampling:
        with tel.stage("thinking", overall_range=(0.0, 0.3)):
            cot, picked = model.think(payload.prompt, references, seed=seed, pick_size=size is None,
                                      cancel=ctx.raise_if_cancelled,
                                      on_token=tel.step_callback(MAX_NEW_TOKENS, stage="thinking",
                                                                 overall_range=(0.0, 0.3)))
        size = size or picked
        tel.log("rewritten prompt", caption=caption(cot, payload.prompt)[:2000])
    elif size is None:
        with tel.stage("picking size", overall_range=(0.0, 0.05)):
            size = model.pick_size(payload.prompt, references, seed=seed)
    assert size is not None
    check_pixels(adj, payload.megapixels, size)
    with tel.stage("generating image", overall_range=(0.3, 0.97)):
        image = model.paint(payload.prompt, references, cot, size=size, steps=steps, guidance=payload.guidance,
                            seed=seed, cancel=ctx.raise_if_cancelled,
                            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.3, 0.95)))
    with tel.stage("saving image", overall_range=(0.97, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, image.width, image.height, seed, caption(cot, payload.prompt))


_COMPONENTS = {"model": ("model",)}


@app.entrypoint(defaults={"model": "hunyuan-image-3-instruct@0.1.0"}, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, model: HunyuanImage3, out: Outputs, tel: Telemetry,
             adj: Adjustments) -> ImageOutput:
    """HunyuanImage 3.0 Instruct, 50 steps with CFG 2.5: text to image, or an edit of up to three references."""
    return _generate(ctx, payload, payload.steps or 50, model, out, tel, adj)


@app.entrypoint(defaults={"model": "hunyuan-image-3-instruct-distil@0.1.0"}, components=_COMPONENTS,
                memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: TurboInput, model: HunyuanImage3, out: Outputs, tel: Telemetry,
                   adj: Adjustments) -> ImageOutput:
    """HunyuanImage 3.0 Instruct-Distil, 8 steps: text to image, or an edit of up to three references."""
    return _generate(ctx, payload, payload.steps or 8, model, out, tel, adj)
