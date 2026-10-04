"""Actual CPU tokenizer and tiny CLIP comparison; not real-model/GPU qualification."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer


def source_symbols(path, names, scope):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    # Comfy's normal entrypoint opts into CLI parsing before model imports.
    sys.argv = ["conditioning-cpu-proof", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    from comfy import clip_model, ops, sdxl_clip
    from comfy.text_encoders.anima import AnimaTokenizer

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    scope = {
        "Any": Any,
        "Path": Path,
        "json": json,
        "CLIPTokenizer": CLIPTokenizer,
        "__file__": str(root / "sdxl/sdxl/__init__.py"),
    }
    source_symbols(root / "sdxl/sdxl/__init__.py", {"_tokenizer", "_ids", "_tokenize"}, scope)
    assets = SimpleNamespace(names=lambda: ())
    native_sdxl = tuple(scope["_tokenizer"](assets, name) for name in ("tokenizer", "tokenizer_2"))
    stock_sdxl = sdxl_clip.SDXLTokenizer()
    from transformers import PreTrainedTokenizerFast

    anima_scope = {
        "Any": Any,
        "Path": Path,
        "json": json,
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
    }
    source_symbols(root / "anima/anima/__init__.py", {"_tokenizer"}, anima_scope)
    native_qwen = anima_scope["_tokenizer"](root / "anima/anima/tokenizer")
    native_t5 = anima_scope["_tokenizer"](root / "anima/anima/t5_tokenizer")
    stock_anima = AnimaTokenizer()
    matrix = json.loads(args.matrix.read_text())
    token_checks = []
    for cell, pairs in matrix["cells_by_pair"].items():
        for index, row in enumerate(pairs["0"]["requests"]):
            observations = {}
            for field in ("prompt", "negative_prompt"):
                text = row["input"][field]
                if row["model"] == "sdxl":
                    native = [ids[0].tolist() for ids in scope["_tokenize"](native_sdxl, text)]
                    stock = stock_sdxl.tokenize_with_weights(text)
                    assert len(stock["l"]) == len(stock["g"]) == 1
                    observed = [[token for token, weight in stock[key][0]] for key in ("l", "g")]
                    assert all(weight == 1 for key in ("l", "g") for token, weight in stock[key][0])
                    assert native == observed
                else:
                    native = [
                        tok(
                            text,
                            padding="longest",
                            max_length=512,
                            truncation=True,
                            return_tensors="pt",
                        )
                        .input_ids[0]
                        .tolist()
                        for tok in (native_qwen, native_t5)
                    ]
                    stock = stock_anima.tokenize_with_weights(text)
                    assert len(stock["qwen3_06b"]) == len(stock["t5xxl"]) == 1
                    observed = [
                        [token for token, weight in stock[key][0]] for key in ("qwen3_06b", "t5xxl")
                    ]
                    assert all(
                        weight == 1
                        for key in ("qwen3_06b", "t5xxl")
                        for token, weight in stock[key][0]
                    )
                    assert native == observed
                observations[field] = {"exact": True, "ids": native}
            token_checks.append(
                {
                    "cell": cell,
                    "index": index,
                    "model": row["model"],
                    "input": row["input"],
                    "tokenization": observations,
                }
            )
    config = {
        "vocab_size": 49408,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "max_position_embeddings": 8,
        "eos_token_id": 2,
        "bos_token_id": 49406,
        "pad_token_id": 0,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "projection_dim": 32,
    }
    hf_config = CLIPTextConfig(**config)
    hf_config._attn_implementation = "sdpa"
    stock = clip_model.CLIPTextModel(
        config, dtype=torch.float16, device="cpu", operations=ops.manual_cast
    ).eval()
    native = CLIPTextModelWithProjection(hf_config).float().eval()
    generator = torch.Generator().manual_seed(4001)
    state = {}
    for key, value in native.state_dict().items():
        data = torch.randn(value.shape, generator=generator) * 0.01
        if "layer_norm" in key and key.endswith("weight"):
            data += 1
        state[key] = data.half()
    stock.load_state_dict(state, strict=True)
    native.load_state_dict(state, strict=True)
    tokens = torch.tensor(
        [[49406, 10, 11, 49407, 49407, 49407, 49407, 49407], [49406, 30, 49407, 0, 0, 0, 0, 0]]
    )
    observed_dtypes = []
    handle = stock.text_model.encoder.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda module, inputs: observed_dtypes.append(str(inputs[0].dtype))
    )
    with torch.inference_mode():
        a = stock(
            tokens,
            num_tokens=[4, 3],
            intermediate_output=-2,
            final_layer_norm_intermediate=False,
            dtype=torch.float32,
        )
        b = native(tokens, output_hidden_states=True)
    handle.remove()
    hidden_difference = float((a[1] - b.hidden_states[-2]).abs().max())
    pooled_difference = float((a[2] - b.text_embeds).abs().max())
    assert observed_dtypes == ["torch.float32"]
    assert a[1].dtype == b.hidden_states[-2].dtype == torch.float32
    assert hidden_difference < 1e-5 and pooled_difference < 1e-5
    legacy = CLIPTextModelWithProjection(hf_config).half().eval()
    legacy.load_state_dict(state, strict=True)
    with torch.inference_mode():
        old = legacy(tokens, output_hidden_states=True)
    result = {
        "scope": "CPU candidate bundled tokenizer assets and actual tiny Comfy/Transformers CLIP. Actual model asset overrides and real-model/GPU conditioning remain unproven.",
        "torch": torch.__version__,
        "token_checks": token_checks,
        "tiny_clip": {
            "source_weights": "shared half-representable deterministic values",
            "stock_stored_dtype": "float16",
            "stock_compute_dtype": observed_dtypes[0],
            "native_candidate_compute_dtype": "float32",
            "penultimate_unnormalized_max_absolute_difference": hidden_difference,
            "projected_eos_pool_max_absolute_difference": pooled_difference,
            "legacy_fp16_hidden_difference": float(
                (old.hidden_states[-2].float() - a[1]).abs().max()
            ),
        },
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("sdxl/sdxl/__init__.py", "anima/anima/__init__.py")
        },
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "actual_tokenization_checks": len(token_checks),
                "tiny_clip": result["tiny_clip"],
                "scope": result["scope"],
            }
        )
    )


if __name__ == "__main__":
    main()
