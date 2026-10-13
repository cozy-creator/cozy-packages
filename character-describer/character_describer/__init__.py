"""One-sentence character identity descriptions from supplied reference images."""

from __future__ import annotations

from typing import Annotated, cast

import msgspec
from PIL.Image import Image
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    MediaDecoder,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource

from .vlm import Describer, prepare

app = App()

IDENTITY_PROMPT = (
    "Describe only the single main character in this image in exactly one concise sentence. "
    "Include distinctive visible identity, species or character type, hair, face, colors, "
    "clothing, armor, accessories, and carried or worn items when visible. "
    "Do not describe the background, setting, scenery, other characters, pose, action, "
    "camera, lighting, image quality, or art style. Do not invent hidden details or a name. "
    "Return only the sentence, without a label or introduction."
)
MAX_NEW_TOKENS = 160


class CaptionImagesInput(msgspec.Struct):
    images: Annotated[
        list[ImageAsset],
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
        msgspec.Meta(min_length=1, max_length=12),
    ]


class Caption(msgspec.Struct):
    index: int
    text: Annotated[str, msgspec.Meta(min_length=1, max_length=2048)]


class CaptionImagesOutput(msgspec.Struct):
    captions: list[Caption]


@invocable(memoize=True)
async def normalize(ctx: Context, *, source: QuantizationSource, tel: Telemetry) -> ModelArtifact:
    """Complete raw upstream weights with the processor and configuration shipped here."""
    return prepare(source, ctx, tel)


app.job(normalize, weights=(WeightsOutput("model", max_new_bytes=64 << 20),), accelerator=False)


@app.entrypoint(
    defaults={"model": [{"gpu": "*", "gpus": 1, "lane": "qwen3.5-9b@0.1.0/bf16"}]},
    memoize=True,
    memo_version="identity-v2",
)
def caption_images(
    ctx: Context,
    payload: CaptionImagesInput,
    model: Describer,
    decoder: MediaDecoder,
    tel: Telemetry,
) -> CaptionImagesOutput:
    """Describe every supplied character independently, in request order."""
    captions: list[Caption] = []
    total = len(payload.images)
    for index, asset in enumerate(payload.images):
        ctx.raise_if_cancelled()
        with tel.stage(f"describing character {index + 1}", overall_range=(index / total, (index + 1) / total)):
            text = model.describe(cast(Image, decoder.value(asset)), IDENTITY_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        if not text:
            raise UnsupportedInput("the character description was empty", code="output_integrity")
        captions.append(Caption(index=index, text=text))
    return CaptionImagesOutput(captions=captions)
