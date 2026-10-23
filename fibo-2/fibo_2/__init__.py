"""Bria Fibo 2.0 text-to-image and image editing, Dev and Edit Turbo, with Bria's prompt-to-JSON VLMs."""

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
from PIL import Image

from .model import Fibo2
from .quantize import int8, nvfp4
from .upsampler import Captioner

app = App()
app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=16 << 30),), accelerator=False)
app.entrypoint(nvfp4, weights=(WeightsOutput("nvfp4", max_new_bytes=16 << 30),), accelerator=False)

#: The area Fibo 2.0 was trained at (model card: "multiples of 16 at about one megapixel").
AREA = 1024 * 1024
MIN_PIXELS = AREA // 4
GRID = 16
#: Diffusers' smallest side (4 latent patches), and the RoPE table's 512 positions an axis.
MIN_SIDE, MAX_SIDE = 4 * GRID, 512 * GRID
DEV_STEPS = 50
TURBO_STEPS = 4
GUIDANCE = 5.0
#: The model card's text-to-image negative: away from soft, featureless, artifact-laden images.
#: Edits take Diffusers' default, the empty caption the model was trained with.
NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified '
    'shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and '
    'featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours",'
    '"neon colour cast","overexposed highlights","lens flare","slight blur","motion blur","haze"]},'
    '"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame",'
    '"focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"",'
    '"artistic_style":""}'
)

_RATIO = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
#: References are prepared to at most 4 MP: the output takes at most 1.
_IMAGES = AssetBound(max_bytes=32 << 20, media_types=("image/png", "image/jpeg", "image/webp"))
_PREPARED = ImagePreparation(max_pixels=4 << 20)
_SEED = Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)]


class GenerateInput(msgspec.Struct):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or composes them, in order (`<image_1>` ... `<image_4>`).
    reference_images: Annotated[list[ImageAsset], _IMAGES, _PREPARED, msgspec.Meta(max_length=4)] = msgspec.field(
        default_factory=list
    )
    #: Any W:H. None: the first reference's, else 1:1.
    aspect_ratio: _RATIO | None = None
    #: None: the first reference's pixel count, else 1. Held to 0.25-1.
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    #: None: 50.
    steps: Annotated[int, msgspec.Meta(ge=1, le=200)] | None = None
    guidance: Annotated[float, msgspec.Meta(ge=1.0, le=20.0)] = GUIDANCE
    #: None: Bria's (the model card's for text-to-image, the empty caption for edits).
    negative_prompt: str | None = None
    seed: _SEED | None = None
    prompt_upsampling: bool = True


class TurboInput(msgspec.Struct):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    #: Given, the image edits or composes them, in order (`<image_1>` ... `<image_4>`).
    reference_images: Annotated[list[ImageAsset], _IMAGES, _PREPARED, msgspec.Meta(max_length=4)] = msgspec.field(
        default_factory=list
    )
    aspect_ratio: _RATIO | None = None
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    #: None: 4.
    steps: Annotated[int, msgspec.Meta(ge=1, le=50)] | None = None
    seed: _SEED | None = None
    prompt_upsampling: bool = True


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(max_bytes=64 << 20, max_decoded_bytes=64 << 20, media_types=("image/png",))]
    width: int
    height: int
    seed: int
    #: The prompt the image was generated from: the VLM's structured caption when `prompt_upsampling` ran.
    prompt: str


def output_size(
    adj: Adjustments, ratio: str | None, megapixels: float | None, first: tuple[int, int] | None
) -> tuple[int, int]:
    """The ratio (else the first reference's, else 1:1) at the pixel count (else the first
    reference's, else 1 MP), held to 0.25-1 MP and 64-8192 px sides, on the 16 px grid."""
    w, h = map(int, ratio.split(":")) if ratio else first or (1, 1)
    pixels = megapixels * AREA if megapixels is not None else (first[0] * first[1] if first else AREA)
    held = min(max(pixels, MIN_PIXELS), AREA)
    if held != pixels:
        adj.clamp("megapixels", round(pixels / AREA, 3), round(held / AREA, 3), "Fibo 2.0's 0.25-1 MP")
    widest = min(MAX_SIDE**2 / held, held / MIN_SIDE**2)
    aspect = min(max(w / h, 1 / widest), widest)
    width = min(max(round(math.sqrt(held * aspect) / GRID) * GRID, MIN_SIDE), MAX_SIDE)
    height = min(max(round(math.sqrt(held / aspect) / GRID) * GRID, MIN_SIDE), MAX_SIDE)
    if aspect != w / h:
        adj.clamp("aspect_ratio", f"{w}:{h}", f"{width}:{height}", "Fibo 2.0's 64-8192 px sides")
    return width, height


def _json_object(prompt: str) -> dict[str, object] | None:
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _caption(
    prompt: str, images: list[Image.Image], upsample: bool, upsampler: Captioner, edit_upsampler: Captioner,
    adj: Adjustments,
) -> str:
    """The prompt Fibo 2.0 reads. A JSON object is already structured; otherwise Bria's VLMs write
    one: `FIBO-vlm` from a prompt, `FIBO-edit-vlm` from a request and its one image. Several images
    take the instruction as typed, a form Fibo 2.0 was also trained on."""
    structured = _json_object(prompt)
    if structured is not None:
        if images and "edit_instruction" not in structured:
            raise UnsupportedInput('an edit\'s JSON prompt needs an "edit_instruction"', code="prompt")
        return prompt
    if not upsample:
        return prompt
    if not images:
        return upsampler.caption(prompt, None)
    if len(images) > 1:
        adj.warn("prompt_upsampling", "Bria's edit VLM reads one image: with several, the instruction goes as typed")
        return prompt
    caption = json.loads(edit_upsampler.caption(prompt, images[0]))
    # FIBO-edit-vlm restates the request as `reworded_prompt`, which Bria's cleaning drops; Fibo 2.0
    # reads the request as `edit_instruction`.
    caption["edit_instruction"] = caption.pop("reworded_prompt", None) or caption.get("edit_instruction") or prompt
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _generate(
    ctx: Context,
    payload: GenerateInput | TurboInput,
    model: Fibo2,
    upsampler: Captioner,
    edit_upsampler: Captioner,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
    *,
    steps: int,
    guidance: float,
    negative: str | None,
) -> ImageOutput:
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    images = [asset.read() for asset in payload.reference_images]
    width, height = output_size(adj, payload.aspect_ratio, payload.megapixels, images[0].size if images else None)
    with tel.stage("upsampling prompt", overall_range=(0.0, 0.15)):
        caption = _caption(payload.prompt, images, payload.prompt_upsampling, upsampler, edit_upsampler, adj)
    if caption != payload.prompt:
        tel.log("upsampled prompt", caption=caption[:2000])
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.15, 0.2)):
        encoded = model.encode(caption, images, width=width, height=height, guidance=guidance, negative=negative)
    latents = []
    if images:
        with tel.stage("encoding reference images", overall_range=(0.2, 0.25)):
            latents = model.encode_images(images, width=width, height=height)
    ctx.raise_if_cancelled()
    with tel.stage("generating image", overall_range=(0.25, 0.95)):
        tokens = model.denoise(
            caption, images, encoded, latents, width=width, height=height, steps=steps, guidance=guidance,
            negative=negative, seed=seed, cancel=ctx.raise_if_cancelled,
            on_step=tel.step_callback(steps, stage="generating image", overall_range=(0.25, 0.95)),
        )
    with tel.stage("decoding image", overall_range=(0.95, 0.99)):
        image = model.decode(tokens)
    with tel.stage("saving image", overall_range=(0.99, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, image.width, image.height, seed, caption)


_MODEL = ("transformer", "text_encoder", "vae")


# A VLM is built and fetched on first use: a request that skips upsampling never pulls one. Model
# parameters are in name order: Runtime refuses a binding set whose paths are not sorted.
@app.entrypoint(defaults={"model": "fibo-2-dev@0.1.0", "upsampler": "fibo-vlm@0.1.0", "edit_upsampler": "fibo-edit-vlm@0.1.0"},
                components={"model": _MODEL, "upsampler": ("model",), "edit_upsampler": ("model",)},
                on_demand=("upsampler", "edit_upsampler"), memoize=Memoize(requires=["seed"]))
def generate(ctx: Context, payload: GenerateInput, edit_upsampler: Captioner, model: Fibo2, upsampler: Captioner,
             out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Fibo 2.0 Dev, 50 steps with CFG 5: text to image, or an edit of up to four ordered references."""
    negative = payload.negative_prompt
    if negative is None and not payload.reference_images:
        negative = NEGATIVE_T2I
    return _generate(ctx, payload, model, upsampler, edit_upsampler, out, tel, adj,
                     steps=payload.steps or DEV_STEPS, guidance=payload.guidance, negative=negative)


@app.entrypoint(defaults={"model": "fibo-2-edit-turbo@0.1.0", "upsampler": "fibo-vlm@0.1.0",
                          "edit_upsampler": "fibo-edit-vlm@0.1.0"},
                components={"model": _MODEL, "upsampler": ("model",), "edit_upsampler": ("model",)},
                on_demand=("upsampler", "edit_upsampler"), memoize=Memoize(requires=["seed"]))
def generate_turbo(ctx: Context, payload: TurboInput, edit_upsampler: Captioner, model: Fibo2, upsampler: Captioner,
                   out: Outputs, tel: Telemetry, adj: Adjustments) -> ImageOutput:
    """Fibo 2.0 Edit Turbo, 4 steps without guidance: text to image, or an edit of up to four ordered references."""
    return _generate(ctx, payload, model, upsampler, edit_upsampler, out, tel, adj,
                     steps=payload.steps or TURBO_STEPS, guidance=1.0, negative=None)
