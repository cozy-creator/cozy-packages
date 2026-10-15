"""One-sentence character identity descriptions from supplied reference images."""

from __future__ import annotations

import re
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
    Weights,
    WeightsOutput,
    invocable,
)
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

from .vlm import Describer

app = App()

IDENTITY_PROMPT = (
    "Describe only how the single main character in this image looks, in exactly one concise "
    "sentence. Include distinctive visible identity, species or character type, hair, face, colors, "
    "clothing, armor, accessories, and carried or worn items when visible. Say nothing about what "
    "the character is doing or where they are: do not describe the background, setting, scenery, "
    "other characters, pose, posture, action, facial expression, camera, lighting, image quality, or "
    "art style. Do not invent hidden details or a name. Return only the sentence, without a label "
    "or introduction."
)
# Owner ruling (se-066): appearance only. A caption naming any of these is regenerated once with
# the words pointed out; clauses still naming one are then dropped.
SCENE_OR_POSE = re.compile(
    r"\b(?:background|backdrop|setting|scene|scenery|surroundings|room|indoors|outdoors|studio|wall|"
    r"mirror|selfie|stand(?:s|ing)?|sit(?:s|ting)?|seated|pos(?:e|es|ed|ing)|lean(?:s|ing)?|"
    r"kneel(?:s|ing)?|crouch(?:es|ing)?|walk(?:s|ing)?|looking|facing|gaz(?:es|ing)|confident(?:ly)?)\b",
    re.IGNORECASE,
)
MAX_NEW_TOKENS = 160


class DescribeCharacterInput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ]


class DescribeCharacterOutput(msgspec.Struct):
    description: Annotated[str, msgspec.Meta(min_length=1, max_length=2048)]


#: The int8 lane: every projection of the 32 language layers (gated-delta-net and attention)
#: and their MLPs. The vision tower, embeddings, the decay/gate projections, norms and the LM
#: head keep their source bytes.
INT8 = QuantizationPlan(
    components=("model",),
    include=(
        "model.model.language_model.layers.*.linear_attn.in_proj_[qz]*.weight",
        "model.model.language_model.layers.*.linear_attn.out_proj.weight",
        "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
        "model.model.language_model.layers.*.mlp.*_proj.weight",
    ),
    output_precision="preserve",
)


@invocable(memo_version="character-describer-int8/1")
async def int8(ctx: Context, *, source: Weights[Describer], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8-convrot256/1`): W8A8 on int8 tensor cores; see `INT8`."""
    return quantize_into(ctx, tel, source=source, plan=INT8, encoding="int8-convrot256/1", output="int8")


app.entrypoint(int8, weights=(WeightsOutput("int8", max_new_bytes=24 << 30),), accelerator=False)


@app.entrypoint(
    defaults={"model": [{"gpu": "*", "gpus": 1, "lane": "qwen3.5-9b@0.2.0/bf16"}]},
    memoize=True,
    memo_version="identity-v3",
)
def describe_character(
    ctx: Context,
    payload: DescribeCharacterInput,
    model: Describer,
    decoder: MediaDecoder,
    tel: Telemetry,
) -> DescribeCharacterOutput:
    """Describe one supplied character's appearance in one sentence."""
    ctx.raise_if_cancelled()
    with tel.stage("describing character"):
        text = _describe(model, cast(Image, decoder.value(payload.image)), tel)
    if not text:
        raise UnsupportedInput("the character description names no appearance", code="output_integrity")
    return DescribeCharacterOutput(text)


def _describe(model: Describer, image: Image, tel: Telemetry) -> str:
    text = model.describe(image, IDENTITY_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    words = sorted({word.lower() for word in SCENE_OR_POSE.findall(text)})
    if not words:
        return text
    tel.log("description named setting or pose; regenerating", words=", ".join(words), first=text[:400])
    retry = (
        f'{IDENTITY_PROMPT} An earlier answer was "{text}". It mentioned {", ".join(words)}, which '
        "describe the pose or surroundings, not the character. Rewrite it without them."
    )
    text = model.describe(image, retry, max_new_tokens=MAX_NEW_TOKENS)
    return strip_scene_or_pose(text)


def strip_scene_or_pose(text: str) -> str:
    """Drop each comma-separated clause that still names a setting or pose."""
    kept = [clause for clause in text.rstrip(". ").split(", ") if not SCENE_OR_POSE.search(clause)]
    return ", ".join(kept) + "." if kept else ""
