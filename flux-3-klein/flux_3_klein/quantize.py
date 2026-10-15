"""Klein's quantized lanes: which weights each encoding takes is this package's recipe; the encodings
and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import Context, ModelArtifact, Telemetry, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into
from cozy_runtime.derive.quantization import QuantizationSource

#: Every linear of the 43 DiT blocks, the six modulation heads, the context embedder and the output
#: norm, and the q/k/v/o and MLP projections of the 32 text-encoder layers the pipeline reads. On real
#: Klein activations int8-convrot keeps each of them within 0.1-1.3% of bf16 (mean, max 1.6%); plain
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


@invocable(memoize=True, memo_version="flux-3-klein-int8/1")
async def int8(ctx: Context, *, source: QuantizationSource, tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8-convrot256/1`): W8A8 on int8 tensor cores, the default lane. The VAE and
    every unlisted weight keep their source precision."""
    return quantize_into(
        ctx, tel, source=source, plan=INT8, encoding="int8-convrot256/1", output="int8"
    )
