"""Qwen3.6: single-turn questions about text, images and videos, answered greedily."""

from __future__ import annotations

from cozy_runtime import vlm
from cozy_runtime.author import App, Context, QuantizationPlan, Recipe, Telemetry
from cozy_runtime.vlm import AskRequest, AskResult, VisionLM

app = App()

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): every attention, gated-delta-net and
#: MLP projection of the language layers and of the multi-token-prediction layer; in 35B-A3B
#: every routed and shared expert. The vision tower, embeddings, routers, the decay/gate
#: projections, norms and the LM head keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=VisionLM,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("model",),
        include=(
            "model.model.language_model.layers.*.linear_attn.in_proj_[qz]*.weight",
            "model.model.language_model.layers.*.linear_attn.out_proj.weight",
            "model.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "model.model.language_model.layers.*.mlp.*_proj.weight",
            "model.mtp.layers.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)


@app.entrypoint(defaults={"model": "qwen3.6-35b-a3b@0.1.0"}, memoize=True, memo_version="ask-v1")
def ask(ctx: Context, payload: AskRequest, model: VisionLM, tel: Telemetry) -> AskResult:
    """Answer each prompt about its text, images and videos, all prompts in one pass."""
    return vlm.ask(ctx, payload, model, tel)
