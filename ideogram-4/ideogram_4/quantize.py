"""Ideogram 4's lanes and its text encoder's: which weights each encoding takes is this package's
recipe; the encodings and their kernels are Runtime's. Ideogram ships only weight-only fp8 (per-row
scales on every linear), which is the release's fp8 lane as uploaded; the bf16 and int8 lanes are
written from its exact decoded values."""

from __future__ import annotations

from cozy_runtime.author import QuantizationPlan, Recipe

from .model import Ideogram4
from .text_encoder import QwenEncoder

_BLOCKS = "*transformer.layers.*"

#: The 16-bit lane: every fp8 weight decoded (f32 code x scale) and rounded once to bf16.
BF16 = Recipe(lane="bf16", model=Ideogram4, plan=QuantizationPlan())

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores), the default: the attention and SwiGLU
#: linears of both DiTs' 34 blocks (8.9B of each 9.3B). The blocks' adaLN modulation, the input,
#: text, timestep and output projections and the VAE are decoded to bf16.
INT8 = Recipe(
    lane="int8",
    model=Ideogram4,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "unconditional_transformer"),
        include=(
            f"{_BLOCKS}.attention.to_[qkv].weight",
            f"{_BLOCKS}.attention.to_out.0.weight",
            f"{_BLOCKS}.feed_forward.w[123].weight",
        ),
    ),
)

_LM = "model.model.language_model.layers.*"

#: Qwen3-VL-8B-Instruct's language linears in `int8.convrot`; its embedding, head and vision tower
#: keep their bf16 bytes. The same lane serves conditioning and the upsampler.
TEXT_ENCODER_INT8 = Recipe(
    lane="int8",
    model=QwenEncoder,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(f"{_LM}.self_attn.[qkvo]_proj.weight", f"{_LM}.mlp.*_proj.weight"),
        output_precision="preserve",
    ),
)
