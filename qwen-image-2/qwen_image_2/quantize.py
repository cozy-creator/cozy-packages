"""Qwen-Image-2.1's quantized lanes: which weights each encoding takes is this package's recipe;
the encodings and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import Context, Loader, Model, ModelArtifact, Telemetry, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

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


class QwenImage21Source(Model[object]):
    """The checkpoint a lane is derived from; never constructed for inference."""

    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True, memo_version="qwen-image-2-int8/1")
async def int8(ctx: Context, *, source: QwenImage21Source, tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8-convrot256/1`): W8A8 on int8 tensor cores, the fast lane on
    consumer cards. VAE and every unlisted weight keep their source precision."""
    return quantize_into(
        ctx, tel, source=source, plan=INT8, encoding="int8-convrot256/1", output="int8"
    )
