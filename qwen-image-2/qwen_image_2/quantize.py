"""Qwen-Image-2.1's quantized lanes: which weights each encoding takes is this package's recipe;
the encodings and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import QwenImage21Model

#: Every linear of the 32 DiT blocks and of the text encoder's 36 language layers, in `int8.convrot`:
#: W8A8 on int8 tensor cores, the fast lane on consumer cards. Embeddings, the vision tower, the
#: timestep/text/image projections, the shared modulation and the output projection keep their
#: source bytes: small, and the ones the image is most sensitive to. So does the VAE.
INT8 = Recipe(
    lane="int8",
    model=QwenImage21Model,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            "transformer.transformer_blocks.*.attn.to_[qkv].weight",
            "transformer.transformer_blocks.*.attn.to_out.0.weight",
            "transformer.transformer_blocks.*.img_mlp.*.weight",
            "text_encoder.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "text_encoder.model.language_model.layers.*.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): the int8 lane's weights with the DiT blocks in W4A8
#: (`w4a8.convrot`, multiplied on the int8 tensor cores: Ampere, Ada, Hopper); the text encoder
#: stays int8.
W4A8 = Recipe(
    lane="w4a8",
    model=QwenImage21Model,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule("transformer.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)
