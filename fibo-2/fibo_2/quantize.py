"""Fibo 2.0's quantized lanes, one recipe per lane for this package's checkpoints (Fibo 2.0 and Bria's
prompt-to-JSON VLMs), chosen by their components. The encodings and their kernels are Runtime's."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Telemetry, UnsupportedInput, Weights, invocable
from cozy_runtime.derive.operations import EncodingRule, QuantizationPlan, quantize_into

_TEXT = "text_encoder.model.language_model.layers.*"

#: Every attention linear of the DiT (its 44 blocks, 2 noise-refiner blocks and 2 Perceiver layers, the
#: five injection blocks' cross-attention included), its FFNs' gate and up projections, and Qwen3-VL's
#: 36 language layers. The FFN down projection (`w2`, 9,536 columns, not a multiple of ConvRot's 256),
#: the adaLN modulation, the embedders, the final layer, the vision tower and the VAE keep their bytes.
INT8 = QuantizationPlan(
    components=("transformer", "text_encoder"),
    include=(
        "transformer.*.to_[qkv].weight",
        "transformer.*.to_out.0.weight",
        "transformer.*.w[13].weight",
        f"{_TEXT}.self_attn.[qkvo]_proj.weight",
        f"{_TEXT}.mlp.*_proj.weight",
    ),
    output_precision="preserve",
)

#: The int8 lane with the DiT's 44 blocks, their down projections too, in NVFP4 (Blackwell).
NVFP4 = QuantizationPlan(
    components=INT8.components,
    include=(*INT8.include, "transformer.layers.*.w2.weight"),
    encodings=(EncodingRule("transformer.layers.*", "nvfp4"),),
    output_precision="preserve",
)

_LM = "model.model.language_model.layers.*"

#: A prompt-to-JSON VLM's language linears: the bytes each decoded token reads. Its embedding (tied
#: to its head) and vision tower keep their source bytes.
CAPTIONER = QuantizationPlan(
    components=("model",),
    include=(f"{_LM}.self_attn.[qkvo]_proj.weight", f"{_LM}.mlp.*_proj.weight"),
    output_precision="preserve",
)

#: Each lane's plan and base encoding by the component that tells the checkpoints apart.
_LANES: dict[str, dict[str, tuple[QuantizationPlan, str]]] = {
    "int8": {"transformer": (INT8, "int8.convrot"), "model": (CAPTIONER, "int8.convrot")},
    "nvfp4": {"transformer": (NVFP4, "int8.convrot"), "model": (CAPTIONER, "nvfp4")},
}


def lane_plan(lane: str, components: tuple[str, ...]) -> tuple[QuantizationPlan, str]:
    """`lane`'s plan and base encoding for the checkpoint holding `components`."""
    found = [plan for component, plan in _LANES[lane].items() if component in components]
    if len(found) != 1:
        raise UnsupportedInput(
            f"{lane} quantizes Fibo 2.0 (`transformer`) or a Bria VLM (`model`); this checkpoint holds "
            f"{sorted(components)}",
            code="quantization_source",
        )
    return found[0]


def _quantize(ctx: Context, tel: Telemetry, source: Weights[Any], lane: str) -> ModelArtifact:
    plan, encoding = lane_plan(lane, source.components())
    return quantize_into(ctx, tel, source=source, plan=plan, encoding=encoding, output=lane)


@invocable(memo_version="fibo-2-int8/1")
async def int8(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default, of Fibo 2.0 or a
    Bria VLM, from its 16-bit release."""
    return _quantize(ctx, tel, source, "int8")


@invocable(memo_version="fibo-2-nvfp4/1")
async def nvfp4(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The nvfp4 lane: Fibo 2.0's DiT blocks W4A4 on Blackwell, the rest as the int8 lane has it; a
    Bria VLM's language linears in NVFP4 (W4A16 for its token loop)."""
    return _quantize(ctx, tel, source, "nvfp4")
