"""Qwen-Image-2.1's quantized lanes: which weights each encoding takes is this package's recipe;
the encodings and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable
from cozy_runtime.derive.operations import EncodingRule, QuantizationPlan, quantize_into

from .model import QwenImage21Model

#: Every linear of the 32 DiT blocks and of the text encoder's 36 language layers. Embeddings,
#: the vision tower, the timestep/text/image projections, the shared modulation and the output
#: projection keep their source bytes: small, and the ones the image is most sensitive to.
INT8 = QuantizationPlan(
    components=("transformer", "text_encoder"),
    include=(
        "transformer.transformer_blocks.*.attn.to_[qkv].weight",
        "transformer.transformer_blocks.*.attn.to_out.0.weight",
        "transformer.transformer_blocks.*.img_mlp.*.weight",
        "text_encoder.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
        "text_encoder.model.language_model.layers.*.mlp.*_proj.weight",
    ),
    output_precision="preserve",
)


@invocable(memo_version="qwen-image-2-int8/1")
async def int8(ctx: Context, *, source: Weights[QwenImage21Model], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`): W8A8 on int8 tensor cores, the fast lane on
    consumer cards. VAE and every unlisted weight keep their source precision."""
    return quantize_into(
        ctx, tel, source=source, plan=INT8, encoding="int8.convrot", output="int8"
    )


#: The 4-bit lane before Blackwell (xs-062): the int8 lane's weights with the DiT blocks in W4A8
#: (`w4a8.convrot`, multiplied on the int8 tensor cores); the text encoder stays int8.
W4A8 = QuantizationPlan(
    components=INT8.components,
    include=INT8.include,
    encodings=(EncodingRule("transformer.*", "w4a8.convrot"),),
    output_precision="preserve",
)


@invocable(memo_version="qwen-image-2-w4a8/1")
async def w4a8(ctx: Context, *, source: Weights[QwenImage21Model], tel: Telemetry) -> ModelArtifact:
    """The w4a8 lane: the DiT blocks as W4A8 on int8 tensor cores (Ampere, Ada, Hopper), the
    text encoder as the int8 lane has it."""
    return quantize_into(
        ctx, tel, source=source, plan=W4A8, encoding="int8.convrot", output="w4a8"
    )
