"""HunyuanImage 3.0's quantized lanes, written from the original 16-bit checkpoint by `cozy model quantize`:
which weights each encoding takes is this package's recipe; the encodings and their kernels are Runtime's."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable
from cozy_runtime.derive.operations import EncodingRule, QuantizationPlan, quantize_into

_LAYERS = "model.model.layers.*"

#: Every linear of the 32 decoder layers (attention, the shared expert and the 64 routed experts: 99% of
#: the 80B parameters) and the LM head the token loop reads. The routers stay float32, and the embedding,
#: vision tower, VAE and the image patch/timestep layers keep their source bytes.
INT8 = QuantizationPlan(
    components=("model",),
    include=(
        f"{_LAYERS}.self_attn.qkv_proj.weight",
        f"{_LAYERS}.self_attn.o_proj.weight",
        f"{_LAYERS}.mlp.shared_mlp.*_proj.weight",
        f"{_LAYERS}.mlp.experts.*.*_proj.weight",
        "model.lm_head.weight",
    ),
    output_precision="preserve",
)

#: The int8 lane with the routed experts in NVFP4 (Blackwell's FP4 tensor cores): ~45 GB instead of ~83.
NVFP4 = QuantizationPlan(
    components=INT8.components,
    include=INT8.include,
    encodings=(EncodingRule(f"{_LAYERS}.mlp.experts.*", "nvfp4"),),
    output_precision="preserve",
)


@invocable(memo_version="hunyuan-image-3-int8/1")
async def int8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): the default 8-bit lane."""
    return quantize_into(ctx, tel, source=source, plan=INT8, encoding="int8.convrot", output="int8")


@invocable(memo_version="hunyuan-image-3-nvfp4/1")
async def nvfp4(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The nvfp4 lane: the routed experts in NVFP4, everything else as the int8 lane has it."""
    return quantize_into(ctx, tel, source=source, plan=NVFP4, encoding="int8.convrot", output="nvfp4")
