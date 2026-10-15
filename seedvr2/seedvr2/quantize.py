"""SeedVR2's int8 lane: which weights take the encoding is this package's recipe; the encoding
and its kernel are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import Context, Loader, Model, ModelArtifact, Telemetry, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into

#: Every attention and MLP linear of the 36 DiT blocks, video and text streams. The patch,
#: text and time embeddings, the output projection, the modulation, the conditioning and the
#: VAE keep their source bytes.
INT8 = QuantizationPlan(
    components=("dit",),
    include=(
        "dit.blocks.*.attn.proj_qkv.*.weight",
        "dit.blocks.*.attn.proj_out.*.weight",
        "dit.blocks.*.mlp.*.proj_*.weight",
    ),
    output_precision="preserve",
)


class SeedVR2Source(Model[object]):
    """The checkpoint a lane is derived from; never constructed for inference."""

    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True, memo_version="seedvr2-int8/1")
async def int8(ctx: Context, *, source: SeedVR2Source, tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8-convrot256/1`): W8A8 on the int8 tensor cores."""
    return quantize_into(
        ctx, tel, source=source, plan=INT8, encoding="int8-convrot256/1", output="int8"
    )
