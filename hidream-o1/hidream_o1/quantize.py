"""HiDream-O1's lanes, for both checkpoints (full and Dev): which weights each encoding takes is this
package's recipe; the encodings and their kernels are Runtime's. HiDream ships f32 weights: every lane
is written from them, and every f32 tensor a lane does not encode is rounded to bf16."""

from __future__ import annotations

import cozy_runtime.derive as derive
from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

from .model import HiDreamO1

_LAYERS = "model.model.language_model.layers.*"

#: The q/k, gate and up projections of the 36 decoder layers: 63% of the decoder's linear weights.
#: v_proj, o_proj and down_proj keep 16 bits: with every projection int8, CFG 5 multiplies W8A8's
#: per-pass error into visible grain (teacher-forced CFG x0 17-24 dB vs bf16 at t 999/702/290); keeping
#: these three is the best measured trade (23-36 dB). The patch embedder, timestep MLP, pixel head,
#: embeddings and vision tower keep 16 bits too.
LINEARS = QuantizationPlan(
    components=("model",),
    include=(f"{_LAYERS}.self_attn.[qk]_proj.weight", f"{_LAYERS}.mlp.gate_proj.weight",
             f"{_LAYERS}.mlp.up_proj.weight"),
)


@invocable(memo_version="hidream-o1-bf16/1")
async def bf16(ctx: Context, *, source: Weights[HiDreamO1], tel: Telemetry) -> ModelArtifact:
    """The 16-bit lane: every f32 tensor rounded to bf16, a recorded derivation (#695)."""
    return derive.quantize_artifact(source, derive.plan(), ctx=ctx, tel=tel, output="bf16")


@invocable(memo_version="hidream-o1-int8/2")
async def int8(ctx: Context, *, source: Weights[HiDreamO1], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`), W8A8 on int8 tensor cores: the default lane."""
    return quantize_into(ctx, tel, source=source, plan=LINEARS, encoding="int8.convrot", output="int8")

