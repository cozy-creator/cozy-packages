"""SeedVR2's int8 lane: which weights take the encoding is this package's recipe; the encoding
and its kernel are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import RestorationModel

#: Every attention and MLP linear of the 36 DiT blocks, video and text streams, in `int8.convrot`:
#: W8A8 on the int8 tensor cores. The patch, text and time embeddings, the output projection, the
#: modulation, the conditioning and the VAE keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=RestorationModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("dit",),
        include=(
            "dit.blocks.*.attn.proj_qkv.*.weight",
            "dit.blocks.*.attn.proj_out.*.weight",
            "dit.blocks.*.mlp.*.proj_*.weight",
        ),
        output_precision="preserve",
    ),
)
