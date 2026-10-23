"""Qwen-Image-Edit-2511's quantized lanes: which weights each encoding takes is this package's
recipe; the encodings and their kernels are Runtime's. Both are made from the bf16 lane."""

from __future__ import annotations

from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable
from cozy_runtime.derive.operations import EncodingRule, QuantizationPlan, quantize_into

from .model import QwenImageEdit

_BLOCKS = "transformer.transformer_blocks.*"

#: Every linear of the 60 DiT blocks, their two modulation heads included (6.8B of the 20B
#: weights). The text encoder stays bf16, as in v1's 8-bit lane: its language layers carry the
#: references' image tokens, and with it int8 too a two-reference compose lost Picture 1's
#: identity (run 14702). It runs once per request. The input, timestep and output projections
#: and the VAE keep their source bytes too.
INT8 = QuantizationPlan(
    components=("transformer",),
    include=(
        f"{_BLOCKS}.attn.to_[qkv].weight",
        f"{_BLOCKS}.attn.add_[qkv]_proj.weight",
        f"{_BLOCKS}.attn.to_out.0.weight",
        f"{_BLOCKS}.attn.to_add_out.weight",
        f"{_BLOCKS}_mlp.net.*.weight",
        f"{_BLOCKS}_mod.1.weight",
    ),
    output_precision="preserve",
)

#: The int8 lane with the blocks' attention and MLP linears in NVFP4 (W4A4 on Blackwell's FP4
#: tensor cores); the modulation heads stay int8 and the text encoder bf16.
NVFP4 = QuantizationPlan(
    components=INT8.components,
    include=INT8.include,
    encodings=(
        EncodingRule(f"{_BLOCKS}.attn.*", "nvfp4"),
        EncodingRule(f"{_BLOCKS}_mlp.*", "nvfp4"),
    ),
    output_precision="preserve",
)


@invocable(memo_version="qwen-image-edit-int8/1")
async def int8(ctx: Context, *, source: Weights[QwenImageEdit], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`): W8A8 on int8 tensor cores, the default lane."""
    return quantize_into(ctx, tel, source=source, plan=INT8, encoding="int8.convrot", output="int8")


@invocable(memo_version="qwen-image-edit-nvfp4/1")
async def nvfp4(ctx: Context, *, source: Weights[QwenImageEdit], tel: Telemetry) -> ModelArtifact:
    """The nvfp4 lane: the blocks' attention and MLP as NVFP4, everything else as the int8 lane."""
    return quantize_into(ctx, tel, source=source, plan=NVFP4, encoding="int8.convrot", output="nvfp4")
