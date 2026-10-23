"""MOSS-Image's int8 lane: which weights it encodes is this package's recipe; the encoding and its
kernels are Runtime's. The upload is the bf16 lane as it is."""

from __future__ import annotations

from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

from .model import MossImage

_TEXT = "text_encoder.language_model.layers.*"

#: The 42 DiT blocks' q, k, v, o, FFN up-gate and down projections (Comfy-Org's int8 file takes the
#: same), and every projection of Qwen3.5's 32 language layers: full attention q/k/v/o, Gated DeltaNet
#: qkv/z/out, and the MLPs. The text refiner, embedders, modulation, final layer, token embedding,
#: DeltaNet's a/b gates, the vision tower and the VAE keep their bytes.
INT8 = QuantizationPlan(
    components=("transformer", "text_encoder"),
    include=(
        "transformer.layers.*.proj.*.weight",
        f"{_TEXT}.self_attn.[qkvo]_proj.weight",
        f"{_TEXT}.linear_attn.in_proj_qkv.weight",
        f"{_TEXT}.linear_attn.in_proj_z.weight",
        f"{_TEXT}.linear_attn.out_proj.weight",
        f"{_TEXT}.mlp.*_proj.weight",
    ),
    output_precision="preserve",
)


@invocable(memo_version="moss-image-int8/1")
async def int8(ctx: Context, *, source: Weights[MossImage], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default, from the bf16 upload."""
    return quantize_into(ctx, tel, source=source, plan=INT8, encoding="int8.convrot", output="int8")
