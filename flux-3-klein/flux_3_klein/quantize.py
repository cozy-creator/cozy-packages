"""Klein's quantized lanes: which weights each encoding takes is this package's recipe, one per lane for
both its checkpoints (Klein and the upsampler); the encodings and their kernels are Runtime's."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Telemetry, UnsupportedInput, Weights, invocable
from cozy_runtime.derive.operations import (
    EncodingRule,
    QuantizationPlan,
    SelectionPlan,
    quantize_into,
    select_into,
)

from .upsampler import Drafter

#: Every linear of the 43 DiT blocks, the six modulation heads, the context embedder and the output
#: norm, and the q/k/v/o and MLP projections of the 32 text-encoder layers the pipeline reads. On real
#: Klein activations int8.convrot keeps each of them within 0.1-1.3% of bf16 (mean, max 1.6%); plain
#: int8 is 1-5.6% and fp8 rowwise 1.5-2.8%. The input/timestep embedders and the output projection
#: are tiny, and the VAE stays f32.
INT8 = QuantizationPlan(
    components=("transformer", "text_encoder"),
    include=(
        "transformer.*blocks.*.attn.to_[qkv].weight",
        "transformer.*blocks.*.attn.to_out.0.weight",
        "transformer.*blocks.*.ff.linear_*.weight",
        "transformer.*_stream_modulations.*.linear.weight",
        "transformer.context_embedder.weight",
        "transformer.norm_out.linear.weight",
        "text_encoder.model.language_model.layers.[0-9].*_proj.weight",
        "text_encoder.model.language_model.layers.[12][0-9].*_proj.weight",
        "text_encoder.model.language_model.layers.3[01].*_proj.weight",
    ),
    output_precision="preserve",
)


#: The int8 lane's weights with the DiT blocks in NVFP4 (`nvfp4`, Blackwell's FP4 tensor
#: cores): 5-10% per linear on real activations, every image clean in the measured set. The
#: modulation heads, context embedder, output norm and text encoder stay int8.
NVFP4 = QuantizationPlan(
    components=INT8.components,
    include=INT8.include,
    encodings=(EncodingRule("transformer.*blocks.*", "nvfp4"),),
    output_precision="preserve",
)


#: The 4-bit lane before Blackwell (xs-062): the int8 lane's weights with the DiT blocks in W4A8
#: (`w4a8.convrot`: 4-bit codes in int8.convrot's basis, multiplied on the int8 tensor
#: cores), everything else as the int8 lane has it.
W4A8 = QuantizationPlan(
    components=INT8.components,
    include=INT8.include,
    encodings=(EncodingRule("transformer.*blocks.*", "w4a8.convrot"),),
    output_precision="preserve",
)

_LM = "model.model.language_model.layers.*"

#: The upsampler's (Qwen3.5-9B) language linears and its LM head: the bytes each decoded token reads.
#: The embedding, the vision tower and the linear-attention gates keep their source precision.
UPSAMPLER_INT8 = QuantizationPlan(
    components=("model",),
    include=(
        f"{_LM}.mlp.*_proj.weight",
        f"{_LM}.linear_attn.in_proj_qkv.weight",
        f"{_LM}.linear_attn.in_proj_z.weight",
        f"{_LM}.linear_attn.out_proj.weight",
        f"{_LM}.self_attn.[qkvo]_proj.weight",
        "model.lm_head.weight",
    ),
    output_precision="preserve",
)

#: The same weights in NVFP4, served W4A16 below Blackwell and at decode: 0.56 bytes per weight,
#: and the vision tower's linears too (its `mlp.linear_fc2` has K=4304, which no FP4 GEMM tiles).
#: That is what brings the upsampler inside an 8 GB card's pool.
UPSAMPLER_NVFP4 = QuantizationPlan(
    components=UPSAMPLER_INT8.components,
    include=(
        *UPSAMPLER_INT8.include,
        "model.model.visual.blocks.*.attn.qkv.weight",
        "model.model.visual.blocks.*.attn.proj.weight",
        "model.model.visual.blocks.*.mlp.linear_fc1.weight",
        "model.model.visual.merger.linear_fc*.weight",
    ),
    output_precision="preserve",
)


#: Each lane's plan and base encoding by the component that tells the checkpoints apart: Klein's
#: `transformer`, the upsampler's one `model`.
_CHECKPOINTS = {"transformer": "Klein (`transformer`)", "model": "its upsampler (`model`)"}
_LANES: dict[str, dict[str, tuple[QuantizationPlan, str]]] = {
    "int8": {"transformer": (INT8, "int8.convrot"), "model": (UPSAMPLER_INT8, "int8.convrot")},
    "nvfp4": {"transformer": (NVFP4, "int8.convrot"), "model": (UPSAMPLER_NVFP4, "nvfp4")},
    "w4a8": {"transformer": (W4A8, "int8.convrot")},
}


def lane_plan(lane: str, components: Iterable[str]) -> tuple[QuantizationPlan, str]:
    """`lane`'s plan and base encoding for the checkpoint holding `components`."""
    held = set(components)
    found = [plan for component, plan in _LANES[lane].items() if component in held]
    if len(found) != 1:
        raise UnsupportedInput(
            f"{lane} quantizes {' or '.join(_CHECKPOINTS[c] for c in _LANES[lane])}; this checkpoint holds {sorted(held)}",
            code="quantization_source",
        )
    return found[0]


def _quantize(ctx: Context, tel: Telemetry, source: Weights[Any], lane: str) -> ModelArtifact:
    plan, encoding = lane_plan(lane, source.components())
    return quantize_into(ctx, tel, source=source, plan=plan, encoding=encoding, output=lane)


@invocable(memo_version="flux-3-klein-int8/1")
async def int8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`) of Klein or its upsampler: W8A8 on int8 tensor cores, the default
    lane. The VAE and every unlisted weight keep their source precision."""
    return _quantize(ctx, tel, source, "int8")


@invocable(memo_version="flux-3-klein-nvfp4/1")
async def nvfp4(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The nvfp4 lane: Klein's DiT blocks as W4A4 on Blackwell's FP4 tensor cores, everything else as the
    int8 lane has it; the upsampler's linears in NVFP4, W4A16 for its token loop."""
    return _quantize(ctx, tel, source, "nvfp4")


@invocable(memo_version="flux-3-klein-w4a8/1")
async def w4a8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The w4a8 lane: Klein's DiT blocks as W4A8 on int8 tensor cores (Ampere, Ada, Hopper),
    everything else as the int8 lane has it. Klein only: the upsampler's 4-bit lane is nvfp4."""
    return _quantize(ctx, tel, source, "w4a8")


#: The drafter's lane: Qwen3.5-9B's multi-token-prediction head alone, 0.5 GB of the 18 GB model.
DRAFTER_HEAD = SelectionPlan(include=("model.mtp.*",))


@invocable(memo_version="flux-3-klein-drafter-head/1")
async def drafter_head(ctx: Context, *, source: Weights[Drafter], tel: Telemetry) -> ModelArtifact:
    """The drafter's lane from any Qwen3.5-9B lane: its `mtp.*` (bf16 in each), the source's own
    objects, with the model's config."""
    return select_into(ctx, tel, source=source, plan=DRAFTER_HEAD, output="bf16")
