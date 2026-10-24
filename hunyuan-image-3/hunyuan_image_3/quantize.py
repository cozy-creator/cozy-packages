"""HunyuanImage 3.0's quantized lanes, written from the original 16-bit checkpoint by `cozy model quantize`:
which weights each encoding takes is this package's recipe; the encodings and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe

from .model import HunyuanImage3

_LAYERS = "model.model.layers.*"

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default 8-bit lane: every linear of
#: the 32 decoder layers (attention, the shared expert and the 64 routed experts: 99% of the 80B
#: parameters) and the LM head the token loop reads. The routers stay float32, and the embedding,
#: vision tower, VAE and the image patch/timestep layers keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=HunyuanImage3,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            f"{_LAYERS}.self_attn.qkv_proj.weight",
            f"{_LAYERS}.self_attn.o_proj.weight",
            f"{_LAYERS}.mlp.shared_mlp.*_proj.weight",
            f"{_LAYERS}.mlp.experts.*.*_proj.weight",
            "model.lm_head.weight",
        ),
        output_precision="preserve",
    ),
)

#: The int8 lane with the routed experts in NVFP4 (Blackwell's FP4 tensor cores): ~45 GB instead of ~83.
NVFP4 = Recipe(
    lane="nvfp4",
    model=HunyuanImage3,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule(f"{_LAYERS}.mlp.experts.*", "nvfp4"),),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): the routed experts in W4A8 (`w4a8.convrot`, 4-bit codes multiplied
#: on the int8 tensor cores, an RTX 4090's), everything else as the int8 lane has it: half the expert bytes a
#: paged step moves.
W4A8 = Recipe(
    lane="w4a8",
    model=HunyuanImage3,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule(f"{_LAYERS}.mlp.experts.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)
