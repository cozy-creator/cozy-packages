"""FLUX.2's quantized lanes, one recipe per lane for every checkpoint ([dev] and [klein]): which weights
each encoding takes is this package's; the encodings and their kernels are Runtime's.

As BFL's own FLUX.2 8-bit and NVFP4 files (`FLUX.2-klein-4b-fp8`, `FLUX.2-dev-NVFP4`), only the DiT blocks'
linears are quantized: the modulation heads, the context, image and time embedders and the final layer keep
their bf16, and NVFP4 leaves the double blocks' text attention at a higher precision (int8 here). The VAE and
Mistral's vision tower keep their bytes too.
"""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Telemetry, UnsupportedInput, Weights, invocable
from cozy_runtime.derive.operations import EncodingRule, QuantizationPlan, quantize_into

#: Every linear of the DiT's blocks: attention and MLPs of both streams.
_DIT = (
    "transformer.*blocks.*.attn.to_*.weight",
    "transformer.*blocks.*.attn.add_*_proj.weight",
    "transformer.*blocks.*.ff*.linear_*.weight",
)
#: Each text encoder's layer projections; [dev]'s LM head too, which its upsampler decodes with.
_TEXT = {
    "mistral3": (
        "text_encoder.language_model.model.layers.*_proj.weight",
        "text_encoder.language_model.lm_head.weight",
    ),
    "qwen3": ("text_encoder.model.layers.*_proj.weight",),
}
#: BFL's NVFP4 [dev] keeps the double blocks' text attention (`txt_attn.qkv`, `.proj`) out of FP4.
_NVFP4 = (
    EncodingRule("transformer.transformer_blocks.*.attn.add_*_proj.weight", "int8.convrot"),
    EncodingRule("transformer.transformer_blocks.*.attn.to_add_out.weight", "int8.convrot"),
    EncodingRule("transformer.*blocks.*", "nvfp4"),
)


def lane_plan(lane: str, family: str) -> QuantizationPlan:
    """`lane`'s plan for a checkpoint whose text encoder is `family`: int8 everywhere it selects, or
    the DiT blocks in nvfp4 beside an int8 rest."""
    return QuantizationPlan(
        components=("transformer", "text_encoder"),
        include=(*_DIT, *_TEXT[family]),
        encodings=_NVFP4 if lane == "nvfp4" else (),
        output_precision="preserve",
    )


def _family(source: Weights[Any]) -> str:
    if "text_encoder" not in source.components():
        raise UnsupportedInput("FLUX.2's lanes quantize a [dev] or [klein] checkpoint (the Turbo LoRA stays 16-bit)",
                               code="quantization_source")
    layer = next(key for key in source.keys("text_encoder") if "layers." in key)
    return "mistral3" if "language_model" in layer else "qwen3"


def _quantize(ctx: Context, tel: Telemetry, source: Weights[Any], lane: str) -> ModelArtifact:
    plan = lane_plan(lane, _family(source))
    return quantize_into(ctx, tel, source=source, plan=plan, encoding="int8.convrot", output=lane)


@invocable(memo_version="flux-2-int8/2")
async def int8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`): W8A8 on int8 tensor cores, the default lane."""
    return _quantize(ctx, tel, source, "int8")


@invocable(memo_version="flux-2-nvfp4/2")
async def nvfp4(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The 4-bit lane: the DiT blocks as W4A4 on Blackwell's FP4 tensor cores, the rest as int8 has it."""
    return _quantize(ctx, tel, source, "nvfp4")
