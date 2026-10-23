"""HiDream-O1's lanes, one recipe per lane for both checkpoints the package serves (HiDream-O1's and
its prompt agent's Gemma 4): which weights each encoding takes is this package's recipe; the encodings
and their kernels are Runtime's. HiDream ships f32 weights: every lane is written from them, and every
f32 tensor a lane does not encode is rounded to bf16."""

from __future__ import annotations

from typing import Any

import cozy_runtime.derive as derive
from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

_LAYERS = "model.model.language_model.layers.*"

#: The attention and MLP projections of the decoder layers, in both checkpoints: in HiDream-O1's 36,
#: 6.9B of the 8.2B weights it builds, read every step by every token; in Gemma 4's 60, the bytes
#: each decoded token reads. Patch embedder, timestep MLP, pixel head, token embeddings (Gemma's tied
#: head) and vision tower keep 16 bits.
LINEARS = QuantizationPlan(
    components=("model",),
    include=(f"{_LAYERS}.self_attn.*_proj.weight", f"{_LAYERS}.mlp.*_proj.weight"),
)


@invocable(memo_version="hidream-o1-bf16/1")
async def bf16(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The 16-bit lane: every f32 tensor rounded to bf16, a recorded derivation (#695)."""
    return derive.quantize_artifact(source, derive.plan(), ctx=ctx, tel=tel, output="bf16")


@invocable(memo_version="hidream-o1-int8/1")
async def int8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`), W8A8 on int8 tensor cores: the default lane."""
    return quantize_into(ctx, tel, source=source, plan=LINEARS, encoding="int8.convrot", output="int8")


@invocable(memo_version="hidream-o1-nvfp4/1")
async def nvfp4(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The nvfp4 lane: the same projections W4A4 on Blackwell's FP4 tensor cores, W4A16 elsewhere
    and in the agent's token loop."""
    return quantize_into(ctx, tel, source=source, plan=LINEARS, encoding="nvfp4", output="nvfp4")
