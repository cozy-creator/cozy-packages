#!/usr/bin/env python
"""Real tiny HiDream-O1 modules on CPU, through the Runtime test harness.

Run with the package's own interpreter. The cached-prefix forward equals Transformers' own Qwen3-VL
text model run over the whole sequence under the hybrid mask (text causal, the timestep token and the
patches attending to everything), deepstack included; the samplers' schedules are upstream's; sizes
and references follow the package's rules; `generate` and `generate_turbo` run end to end.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from cozy_runtime.author import Config, ModelAssets
from cozy_runtime.author.fakes import granted_input, invoke_with_fakes
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hidream-o1"))

from hidream_o1 import app, layout  # noqa: E402
from hidream_o1.agent import PromptAgent, caption  # noqa: E402
from hidream_o1.model import (  # noqa: E402
    FIX_POINT,
    HiDreamO1,
    O1Graph,
    _predict,
    _prefix,
    area_resize,
    build_processor,
    condition_image,
    patches,
    reference_size,
    sequence,
    unpatch,
)
from hidream_o1.sampling import UniPC  # noqa: E402

#: Qwen3-VL's chat template, as HiDream's checkpoint renders a user turn of images and text.
CHAT = (
    '{% for message in messages %}{{"<|im_start|>" + message["role"] + "\\n"}}'
    '{% if message["content"] is string %}{{message["content"]}}{% else %}{% for item in message["content"] %}'
    '{% if item["type"] == "image" %}{{"<|vision_start|><|image_pad|><|vision_end|>"}}{% else %}{{item["text"]}}'
    '{% endif %}{% endfor %}{% endif %}{{"<|im_end|>\\n"}}{% endfor %}'
    '{% if add_generation_prompt %}{{"<|im_start|>assistant\\n"}}{% endif %}'
)
SPECIAL = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
           "<|boi_token|>", "<|tms_token|>")
#: Upstream's FlowUniPCMultistepScheduler timesteps at 50 steps, shift 3 (`models/fm_solvers_unipc.py`).
UNIPC_50 = [999, 993, 986, 979, 971, 964, 956, 948, 940, 931, 922, 913, 904, 895, 885, 874, 864, 853, 841, 830, 818,
            805, 792, 778, 764, 749, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 396,
            363, 328, 290, 249, 206, 160, 111, 57]


def assets() -> ModelAssets:
    words = "a red fox in snow at dawn the same at night [UNK] [PAD] user assistant"
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words.split()))}
    for token in SPECIAL:
        vocab[token] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.add_special_tokens(list(SPECIAL))
    geometry = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "image_mean": [0.5] * 3,
                "image_std": [0.5] * 3}
    files = {
        "tokenizer.json": PreTrainedTokenizerFast(tokenizer_object=t).backend_tokenizer.to_str().encode(),
        "tokenizer_config.json": json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]"}).encode(),
        "preprocessor_config.json": json.dumps({**geometry, "size": {"longest_edge": 16777216,
                                                                      "shortest_edge": 65536}}).encode(),
        "video_preprocessor_config.json": json.dumps(geometry).encode(),
        "chat_template.json": json.dumps({"chat_template": CHAT}).encode(),
    }
    return ModelAssets(files, lambda _operation: None)


def graph(dtype: torch.dtype) -> O1Graph:
    config = {
        "model_type": "qwen3_vl",
        "text_config": dict(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                            rope_scaling=dict(rope_type="default", mrope_section=[2, 3, 3], mrope_interleaved=True),
                            rope_theta=5000000),
        "vision_config": dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                              patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64,
                              deepstack_visual_indexes=[0, 1, 2]),
    }
    torch.manual_seed(7)
    built = O1Graph(Config(config))
    with torch.no_grad():
        for name, p in built.components["model"].named_parameters():
            p.normal_(0, 0.5 if "embed_tokens" in name else 0.05) if p.dim() >= 2 else p.uniform_(0.8, 1.2)
        built.components["model"].to(dtype)
    return built


def structure() -> None:
    """The prefix cache and the gen pass equal one full-sequence pass of Transformers' text model."""
    built = graph(torch.float32)
    body = built.components["model"].model
    processor = build_processor(assets())
    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rng = np.random.default_rng(3)
    refs = [Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)),
            Image.fromarray(rng.integers(0, 255, (96, 64, 3), dtype=np.uint8))]
    for references in ([], refs[:1], refs):
        rows, columns = 3, 4
        grids = [(r.height // 32, r.width // 32) for r in references]
        seq = sequence(processor, "a red fox in snow", [condition_image(r, len(references)) for r in references],
                       grids, rows, columns)
        n = rows * columns
        inputs = torch.randn(1, n + sum(h * w for h, w in grids), 3072)
        t = torch.tensor(0.37)
        with torch.inference_mode():
            vision = None
            if seq.pixel_values is not None:
                out = body.visual(seq.pixel_values, grid_thw=seq.grid)
                vision = (out.pooler_output, out.deepstack_features)
            got = _predict(body, _prefix(body, seq, vision, image_token, torch.device("cpu")), t, inputs, n)

            text = body.language_model
            ids = seq.ids
            embeds = text.embed_tokens(ids)
            mask = ids[0] == image_token
            if vision is not None:
                embeds[0, mask] = vision[0]
            embeds[0, -1] = body.t_embedder1(t.view(1))[0]
            h = torch.cat([embeds, body.x_embedder(inputs)], dim=1)
            total = h.shape[1]
            allowed = torch.ones(total, total, dtype=torch.bool).tril()
            allowed[seq.prefix :] = True
            bias = torch.zeros(total, total).masked_fill(~allowed, float("-inf"))[None, None]
            visual = torch.cat([mask, torch.zeros(total - ids.shape[1], dtype=torch.bool)])[None]
            out = text(inputs_embeds=h, position_ids=seq.positions[:, None], attention_mask=bias,
                       visual_pos_masks=visual if vision else None, deepstack_visual_embeds=vision[1] if vision else None)
            want = body.final_layer2(out.last_hidden_state[:, ids.shape[1] : ids.shape[1] + n])
        rel = ((got - want).norm() / want.norm()).item()
        print(f"references={len(references)}: cached prefix vs one pass, relative {rel:.2e}")
        assert rel < 1e-5, rel
        target = seq.positions[:, ids.shape[1] : ids.shape[1] + n]
        assert target[0].min() == FIX_POINT and target[1].max() == FIX_POINT + rows - 1
        assert target[2].max() == FIX_POINT + columns - 1


def schedules() -> None:
    assert UniPC(50, 3.0).timesteps == UNIPC_50
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 96, 3), dtype=np.uint8))
    assert torch.equal(unpatch(patches(image)[None], 64, 96), patches(image)[None].view(2, 3, 3, 32, 32)
                       .permute(2, 0, 3, 1, 4).reshape(3, 64, 96))


class Adjust:
    def __init__(self) -> None:
        self.rows: list[tuple[str, object, object]] = []

    def clamp(self, field: str, requested: object, applied: object, reason: str) -> None:
        self.rows.append((field, requested, applied))

    def warn(self, field: str, message: str) -> None:
        self.rows.append((field, "warn", message))


def sizes() -> None:
    """Text-to-image: 1:1 at 4 MP; any W:H; 1-4 MP and 256-4096 px sides, clamped. With references,
    what is not given is the first's; a lone one given neither is upstream's keep_original_aspect."""
    cases = [
        ([], None, None, (2048, 2048), []),
        ([], "16:9", None, (2720, 1536), []),
        ([], "1:1", 1, (1024, 1024), []),
        ([], "1:1", 8, (2048, 2048), ["megapixels"]),
        ([], "1:1", 0.25, (1024, 1024), ["megapixels"]),
        ([], "20:1", None, (4096, 1024), ["aspect_ratio"]),
        ([(3000, 2000)], None, None, (2496, 1664), ["megapixels"]),
        ([(1200, 900)], None, None, (1184, 896), []),
        ([(400, 300)], None, None, (1152, 864), ["megapixels"]),
        ([(3000, 2000)], "1:1", None, (2048, 2048), ["megapixels"]),
        ([(1200, 900), (500, 500)], None, None, (1184, 896), []),
    ]
    for shapes, ratio, megapixels, want, clamped in cases:
        images = [Image.new("RGB", s) for s in shapes]
        adj = Adjust()
        (width, height), prepared = layout(adj, ratio, megapixels, images)  # type: ignore[arg-type]
        assert (width, height) == want, (shapes, ratio, megapixels, width, height)
        assert [row[0] for row in adj.rows] == clamped, (shapes, ratio, megapixels, adj.rows)
        assert width % 32 == height % 32 == 0
        if len(shapes) == 1 and ratio is None and megapixels is None:
            assert prepared[0].size == (width, height)
        else:
            side = reference_size(max(width, height), len(shapes))
            assert all(p.size == area_resize(i, side).size for p, i in zip(prepared, images, strict=True))


def agent_replies() -> None:
    assert caption('{"prompt": "a fox", "reasoning": "x"}', "fox") == "a fox"
    assert caption('```json\n{"prompt": "two\nlines"}\n```', "fox") == "two\nlines"
    assert caption("no json here", "fox") == "fox, highly detailed, masterpiece, best quality, sharp focus"


def functions(root: Path) -> None:
    """`generate` and `generate_turbo` through the real kernel on real PNGs; `prompt_upsampling`
    never reaches the agent for an edit, and is off by default."""
    models = {"model": HiDreamO1.for_test(graph=graph(torch.bfloat16), processor=build_processor(assets())),
              "agent": PromptAgent.for_test()}
    rng = np.random.default_rng(5)
    paths = []
    for index, size in enumerate(((96, 64), (64, 96))):
        path = root / f"reference-{index}.png"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(path)
    cases: list[tuple[str, list[Path], dict[str, object], tuple[int, int], set[str]]] = [
        ("generate", [], {"megapixels": 1, "steps": 2}, (1024, 1024), set()),
        ("generate", paths[:1], {"steps": 2, "prompt_upsampling": True}, (1248, 832), {"megapixels", "prompt_upsampling"}),
        ("generate", paths, {"steps": 1, "guidance": 1.0, "aspect_ratio": "2:1", "megapixels": 1}, (1440, 704), set()),
        ("generate_turbo", paths[:1], {"steps": 4}, (1248, 832), {"megapixels", "steps"}),
    ]
    for index, (function, references, given, expected, adjusted) in enumerate(cases):
        grants = {f"reference_images.{n}": granted_input(path, input_id=f"reference_images.{n}", order=n)
                  for n, path in enumerate(references)}
        wire = {"prompt": "a red fox in snow", "seed": 3, **given,
                "reference_images": [grant.digest for grant in grants.values()]}
        result = invoke_with_fakes(app.get(function), wire, request_id=f"{function}-{index}", spool=root / str(index),
                                   models=models, assets=grants, seconds=3600)
        got = (result.result.width, result.result.height)
        assert got == expected, (function, given, got)
        assert {row.field for row in result.adjustments} == adjusted, (function, given, result.adjustments)
        assert result.result.prompt == "a red fox in snow"
        print(f"{function} references={len(references)} {given}: {got}")


if __name__ == "__main__":
    agent_replies()
    schedules()
    sizes()
    structure()
    with tempfile.TemporaryDirectory() as scratch:
        functions(Path(scratch))
    print("hidream-o1 model proof passed")
