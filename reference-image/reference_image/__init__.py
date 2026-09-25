"""Text-to-image references using Qwen-Image-2.1, with Runtime-owned model execution."""

from __future__ import annotations

import secrets
from typing import Annotated, Any, Literal

import msgspec
import numpy as np
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    Outputs,
    Telemetry,
)
from cozy_runtime.models.qwen_image21 import QwenImage21Model
from PIL import Image

app = App()
Background = Literal["normal", "white"]


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    width: Annotated[int, msgspec.Meta(ge=256, le=2752, multiple_of=32)] = 1024
    height: Annotated[int, msgspec.Meta(ge=256, le=2752, multiple_of=32)] = 1024
    steps: Annotated[int, msgspec.Meta(ge=1, le=100)] = 40
    seed: Annotated[int, msgspec.Meta(ge=0, le=9223372036854775807)] | None = None
    background: Background = "normal"

    def __post_init__(self) -> None:
        if self.width * self.height > 5_000_000:
            raise ValueError("image area must not exceed 5 million pixels")


class ImageOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=32 << 20, max_decoded_bytes=20 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int
    seed: int


def reference_prompt(prompt: str, background: Background) -> str:
    if background == "white":
        return (
            f"{prompt}\nSingle subject isolated against a seamless pure white studio background. "
            "Show the whole subject clearly, with no scenery, text, labels or other people."
        )
    return prompt


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
    if image.mode != "RGB":
        raise ValueError(f"Qwen Image decoder produced unsupported {image.mode} pixels")
    return image


@app.entrypoint(defaults={"model": [{"gpu": "*", "lane": "paul/reference-image@0.1.0/original"}]})
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: QwenImage21Model,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    seed = payload.seed if payload.seed is not None else secrets.randbits(63)
    ctx.raise_if_cancelled()
    with tel.stage("encoding prompt", overall_range=(0.0, 0.1)):
        embeds, mask = model.encode(reference_prompt(payload.prompt, payload.background))
    with tel.stage("generating image", overall_range=(0.1, 0.9)):
        latents = model.denoise(
            embeds,
            mask,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            seed=seed,
            on_step=tel.step_callback(
                payload.steps, stage="generating image", overall_range=(0.1, 0.9)
            ),
            cancel=ctx.raise_if_cancelled,
        )
    ctx.raise_if_cancelled()
    with tel.stage("decoding image", overall_range=(0.9, 0.98)):
        decoded = model.decode(latents, width=payload.width, height=payload.height)
        image = rgb_image(decoded)
    with tel.stage("saving image", overall_range=(0.98, 1.0)):
        asset = out.save_image(image, format="png")
    return ImageOutput(asset, payload.width, payload.height, seed)
