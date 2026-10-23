"""Klein's quantized lanes, one recipe per lane for each of its checkpoints (Klein and the upsampler):
which weights each encoding takes is this package's; the encodings and their kernels are Runtime's."""

from __future__ import annotations

from cozy_runtime.author import EncodingRule, QuantizationPlan, Recipe, SelectionPlan

from .klein import Klein
from .upsampler import Drafter, Upsampler

#: Every linear of the 43 DiT blocks, the six modulation heads, the context embedder and the output
#: norm, and the q/k/v/o and MLP projections of the 32 text-encoder layers the pipeline reads. On real
#: Klein activations int8.convrot keeps each of them within 0.1-1.3% of bf16 (mean, max 1.6%); plain
#: int8 is 1-5.6% and fp8 rowwise 1.5-2.8%. The input/timestep embedders and the output projection
#: are tiny, and the VAE stays f32. W8A8 on int8 tensor cores, the default lane.
INT8 = Recipe(
    lane="int8",
    model=Klein,
    plan=QuantizationPlan(
        encoding="int8.convrot",
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
    ),
)

#: The int8 lane's weights with the DiT blocks in NVFP4 (`nvfp4`, Blackwell's FP4 tensor cores):
#: 5-10% per linear on real activations, every image clean in the measured set. The modulation
#: heads, context embedder, output norm and text encoder stay int8.
NVFP4 = Recipe(
    lane="nvfp4",
    model=Klein,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule("transformer.*blocks.*", "nvfp4"),),
        output_precision="preserve",
    ),
)

#: The 4-bit lane before Blackwell (xs-062): the int8 lane's weights with the DiT blocks in W4A8
#: (`w4a8.convrot`: 4-bit codes in int8.convrot's basis, multiplied on the int8 tensor cores),
#: everything else as the int8 lane has it. Klein only: the upsampler's 4-bit lane is nvfp4.
W4A8 = Recipe(
    lane="w4a8",
    model=Klein,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=INT8.plan.include,
        encodings=(EncodingRule("transformer.*blocks.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)

_LM = "model.model.language_model.layers.*"

#: The upsampler's (Qwen3.5-9B) language linears and its LM head: the bytes each decoded token reads.
#: The embedding, the vision tower and the linear-attention gates keep their source precision.
UPSAMPLER_INT8 = Recipe(
    lane="int8",
    model=Upsampler,
    plan=QuantizationPlan(
        encoding="int8.convrot",
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
    ),
)

#: The same weights in NVFP4, served W4A16 below Blackwell and at decode: 0.56 bytes per weight,
#: and the vision tower's linears too (its `mlp.linear_fc2` has K=4304, which no FP4 GEMM tiles).
#: That is what brings the upsampler inside an 8 GB card's pool.
UPSAMPLER_NVFP4 = Recipe(
    lane="nvfp4",
    model=Upsampler,
    plan=QuantizationPlan(
        encoding="nvfp4",
        components=UPSAMPLER_INT8.plan.components,
        include=(
            *UPSAMPLER_INT8.plan.include,
            "model.model.visual.blocks.*.attn.qkv.weight",
            "model.model.visual.blocks.*.attn.proj.weight",
            "model.model.visual.blocks.*.mlp.linear_fc1.weight",
            "model.model.visual.merger.linear_fc*.weight",
        ),
        output_precision="preserve",
    ),
)

#: The drafter's lane from any Qwen3.5-9B lane: its multi-token-prediction head alone (`mtp.*`,
#: bf16 in each), 0.5 GB of the 18 GB model, the source's own objects, with the model's config.
DRAFTER_HEAD = Recipe(lane="bf16", model=Drafter, plan=SelectionPlan(include=("model.mtp.*",)))
