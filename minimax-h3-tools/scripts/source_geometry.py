"""Real H3 source rows for the CPU proofs, from facts this package already carries.

The text conditioner's 1,058 rows are Qwen3-VL's state-dict geometry under the committed
`model-config.json`; their keys are the reviewed 902-row order plus the 156 source-only
rows, and `lane-proof.py` checks every shape against the banked upstream headers. The video
VAE rows are four real header rows covering each decode-operand normalisation case.
"""

from __future__ import annotations

from h3_tables import job
from h3_tables.model_config import parse_production_config
from h3_tables.order import current_order
from h3_tables.source import text_source_only_keys
from tensorfs.derived import Part, Tensor

Spec = tuple[str, tuple[int, ...]]

VIDEO_VAE_ROWS: dict[str, Spec] = {
    "decoder.proj_in.weight": ("f32", (2048, 24)),
    "decoder.transformer_blocks.0.attn.to_q.weight": ("f32", (2048, 2048)),
    "post_quant_conv.weight": ("f32", (24, 24, 1, 1, 1)),
    "decoder.norm_out.weight": ("f32", (2048,)),
}


def plain(dtype: str, shape: tuple[int, ...]) -> Tensor:
    return Tensor(dtype, shape, job.PLAIN_SPEC, {"value": Part(dtype, shape)})


def conditioner_rows() -> dict[str, Spec]:
    """Every row of the official text conditioner checkpoint, source-only rows included."""
    config = parse_production_config(job._asset("model-config.json"))["text_encoder"]
    text, vision = config["text_config"], config["vision_config"]
    hidden, inner, head = text["hidden_size"], text["intermediate_size"], text["head_dim"]
    q, kv = text["num_attention_heads"] * head, text["num_key_value_heads"] * head
    width, mlp, out = vision["hidden_size"], vision["intermediate_size"], vision["out_hidden_size"]
    merged, patch = width * vision["spatial_merge_size"] ** 2, vision["patch_size"]
    rows: dict[str, tuple[int, ...]] = {
        "lm_head.weight": (text["vocab_size"], hidden),
        "model.language_model.embed_tokens.weight": (text["vocab_size"], hidden),
        "model.language_model.norm.weight": (hidden,),
        "model.visual.patch_embed.proj.weight": (
            width, vision["in_channels"], vision["temporal_patch_size"], patch, patch
        ),
        "model.visual.patch_embed.proj.bias": (width,),
        "model.visual.pos_embed.weight": (vision["num_position_embeddings"], width),
    }
    layer = {
        "input_layernorm": (hidden,),
        "post_attention_layernorm": (hidden,),
        "self_attn.q_norm": (head,),
        "self_attn.k_norm": (head,),
        "self_attn.q_proj": (q, hidden),
        "self_attn.k_proj": (kv, hidden),
        "self_attn.v_proj": (kv, hidden),
        "self_attn.o_proj": (hidden, q),
        "mlp.gate_proj": (inner, hidden),
        "mlp.up_proj": (inner, hidden),
        "mlp.down_proj": (hidden, inner),
    }
    for index in range(text["num_hidden_layers"]):
        rows |= {f"model.language_model.layers.{index}.{n}.weight": s for n, s in layer.items()}
    block = {
        "attn.qkv": (3 * width, width),
        "attn.proj": (width, width),
        "mlp.linear_fc1": (mlp, width),
        "mlp.linear_fc2": (width, mlp),
        "norm1": (width,),
        "norm2": (width,),
    }
    biased = [
        (f"model.visual.blocks.{index}.{name}", shape)
        for index in range(vision["depth"])
        for name, shape in block.items()
    ]
    mergers = ["model.visual.merger"] + [
        f"model.visual.deepstack_merger_list.{index}"
        for index in range(len(vision["deepstack_visual_indexes"]))
    ]
    for prefix in mergers:
        norm = width if prefix == "model.visual.merger" else merged
        biased += [
            (f"{prefix}.norm", (norm,)),
            (f"{prefix}.linear_fc1", (merged, merged)),
            (f"{prefix}.linear_fc2", (out, merged)),
        ]
    for name, shape in biased:
        rows |= {f"{name}.weight": shape, f"{name}.bias": shape[:1]}
    reviewed = {
        key
        for component, key in current_order(job._asset("whole-order.json")).rows
        if component == "text_encoder"
    }
    if set(rows) != reviewed | set(text_source_only_keys()) or len(rows) != 1058:
        raise AssertionError("the derived conditioner is not the reviewed 902 + 156 source rows")
    return {key: ("bf16", shape) for key, shape in rows.items()}
