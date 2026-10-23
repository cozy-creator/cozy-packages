#!/usr/bin/env python
"""Real tiny Fibo 2.0 modules on CPU: the staged model makes Diffusers' `BriaFibo2Pipeline`'s image.

Run with the package's own interpreter. Text to image (a short description, and a JSON caption
whose text the text encoder also sees drawn), with and without CFG, and edits of one and two
images; then the functions through the Runtime test harness, for their sizes and adjustments.
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
from diffusers import BriaFibo2Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fibo-2"))

from fibo_2 import NEGATIVE_T2I, app  # noqa: E402
from fibo_2.model import FONTS, Fibo2, FiboGraph, build_processor  # noqa: E402
from fibo_2.upsampler import Captioner  # noqa: E402

CHAT = (
    '{% for message in messages %}{{"<|im_start|>" + message["role"] + "\\n"}}'
    '{% if message["content"] is string %}{{message["content"]}}{% else %}'
    '{% for item in message["content"] %}{{item["text"]}}{% endfor %}{% endif %}'
    '{{"<|im_end|>\\n"}}{% endfor %}{% if add_generation_prompt %}{{"<|im_start|>assistant\\n"}}{% endif %}'
)
SPECIAL = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")
GEOMETRY = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}


def tokenizer() -> PreTrainedTokenizerFast:
    words = 'a red fox in snow at dawn make it night put the from next to " : , { } [ ] user assistant [UNK] [PAD]'
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words.split()))}
    for token in SPECIAL:
        vocab[token] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.add_special_tokens(list(SPECIAL))
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]")


def assets() -> ModelAssets:
    image = {**GEOMETRY, "size": {"shortest_edge": 32 * 32, "longest_edge": 96 * 96},
             "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "image_processor_type": "Qwen2VLImageProcessorFast"}
    files = {
        "tokenizer.json": tokenizer().backend_tokenizer.to_str().encode(),
        "tokenizer_config.json": json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]",
                                             "tokenizer_class": "Qwen2Tokenizer"}).encode(),
        "processor_config.json": json.dumps({"image_processor": image, "video_processor": dict(GEOMETRY),
                                             "processor_class": "Qwen3VLProcessor"}).encode(),
        "chat_template.jinja": CHAT.encode(),
    }
    return ModelAssets({f"processor/{k}": v for k, v in files.items()}, lambda _operation: None)


def graph() -> FiboGraph:
    ids = {token: tokenizer().convert_tokens_to_ids(token) for token in SPECIAL}
    config = {
        "transformer": dict(in_channels=2, dim=32, n_layers=2, n_refiner_layers=1, n_heads=2, cap_feat_dim=48,
                            injection_layer_ids=[1], perceiver_num_layers=1, min_num_gist_tokens=4,
                            max_num_gist_tokens=8, gist_step=4, gist_min_text_len=4, gist_max_text_len=32,
                            axes_dims=[4, 6, 6], axes_lens=[16, 512, 512]),
        "vae": dict(block_out_channels=[8, 8, 8, 8], down_block_types=["DownEncoderBlock2D"] * 4,
                    up_block_types=["UpDecoderBlock2D"] * 4, latent_channels=2, layers_per_block=1,
                    norm_num_groups=4, patch_size=[2, 2]),
        "text_encoder": dict(
            text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=16, num_hidden_layers=36,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                             rope_parameters=dict(rope_type="default", rope_theta=1e6, mrope_section=[1, 1, 2]),
                             tie_word_embeddings=True),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=16,
                               patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                               num_position_embeddings=64, deepstack_visual_indexes=[0]),
            image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
            vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
            tie_word_embeddings=True,
        ),
        "scheduler": dict(FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift=3.0).config),
    }
    torch.manual_seed(7)
    built = FiboGraph(Config(config))
    for root in built.components.values():
        for p in root.parameters():
            with torch.no_grad():
                p.normal_(0, 0.2)
    vae = built.components["vae"]
    vae.bn.running_mean.normal_(0, 0.1)
    vae.bn.running_var.uniform_(0.5, 1.5)
    return built


def staged(model: Fibo2, prompt: str, images: list[Image.Image], **kw: object) -> Image.Image:
    size = {"width": kw["width"], "height": kw["height"]}
    shared = dict(guidance=kw["guidance"], negative=kw["negative"])
    encoded = model.encode(prompt, images, **size, **shared)  # type: ignore[arg-type]
    latents = model.encode_images(images, **size) if images else []  # type: ignore[arg-type]
    tokens = model.denoise(prompt, images, encoded, latents, **size, **shared, steps=3, seed=11,  # type: ignore[arg-type]
                           on_step=lambda _: None, cancel=lambda: None)
    return model.decode(tokens)


def proof(prompt: str, images: list[Image.Image], guidance: float, negative: str | None) -> None:
    built = graph()
    model = Fibo2.for_test(graph=built, processor=build_processor(assets(), {}))
    kw = dict(width=64, height=48, guidance=guidance, negative=negative)
    actual = staged(model, prompt, images, **kw)
    pipe = BriaFibo2Pipeline(**built.components, processor=model.processor,
                             scheduler=FlowMatchEulerDiscreteScheduler.from_config(built.scheduler_config))
    pipe.set_progress_bar_config(disable=True)
    expected = pipe(prompt=prompt, image=images or None, width=64, height=48, num_inference_steps=3,
                    guidance_scale=guidance, negative_prompt=negative, fonts=list(FONTS),
                    generator=torch.Generator().manual_seed(11)).images[0]
    difference = np.abs(np.asarray(actual, np.int16) - np.asarray(expected, np.int16)).max()
    print(f"{prompt[:40]!r} references={len(images)} guidance={guidance}: max |Δ| = {difference}")
    assert actual.size == (64, 48) and difference == 0, difference


def functions(root: Path) -> None:
    """Both functions through the real kernel on real PNGs: sizes, clamps and the upsampling skip."""
    model = Fibo2.for_test(graph=graph(), processor=build_processor(assets(), {}))
    # `prompt_upsampling=False` never calls a VLM: unloaded stand-ins.
    vlms = {"upsampler": Captioner.for_test(), "edit_upsampler": Captioner.for_test()}
    rng = np.random.default_rng(5)
    paths = []
    for index, size in enumerate(((300, 180), (200, 200), (2000, 1000))):
        path = root / f"reference-{index}.png"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(path)
    cases: list[tuple[str, list[Path], dict[str, object], tuple[int, int], set[str]]] = [
        ("generate", [], {}, (1024, 1024), set()),
        ("generate_turbo", [], {"megapixels": 0.25}, (512, 512), set()),
        ("generate", [], {"aspect_ratio": "16:9", "megapixels": 0.25}, (688, 384), set()),
        ("generate", [], {"aspect_ratio": "100:1", "megapixels": 0.25}, (4096, 64), {"aspect_ratio"}),
        # a 300x180 reference is 0.05 MP: held to 0.25, at its 5:3
        ("generate", [paths[0]], {}, (656, 400), {"megapixels"}),
        ("generate_turbo", [paths[0]], {"aspect_ratio": "1:1"}, (512, 512), {"megapixels"}),
        ("generate_turbo", [paths[2]], {"megapixels": 0.3}, (800, 400), set()),
        ("generate_turbo", [paths[2]], {"megapixels": 2}, (1456, 720), {"megapixels"}),
    ]
    for index, (function, references, given, expected, clamped) in enumerate(cases):
        grants = {
            f"reference_images.{n}": granted_input(path, input_id=f"reference_images.{n}", order=n)
            for n, path in enumerate(references)
        }
        wire = {"prompt": "a red fox", "steps": 1, "seed": 3, "prompt_upsampling": False, **given}
        wire["reference_images"] = [grant.digest for grant in grants.values()]
        spool = root / f"run-{index}"
        result = invoke_with_fakes(app.get(function), wire, request_id=f"run-{index}", spool=spool,
                                   models={"model": model, **vlms}, assets=grants)
        assert (result.result.width, result.result.height) == expected, (function, given, result.result)
        assert result.result.prompt == "a red fox" and result.result.seed == 3
        fields = {row.field for row in result.adjustments if row.kind == "clamp"}
        assert fields == clamped, (function, given, fields)
        with Image.open(spool / "image-0001") as png:
            assert png.size == expected, png.size
        print(f"{function} {given} references={len(references)}: {expected}, clamped {sorted(clamped)}")


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    pictures = [Image.fromarray(rng.integers(0, 255, (48 + 16 * i, 64, 3), dtype=np.uint8)) for i in range(2)]
    caption = json.dumps({"short_description": "a red fox in snow", "text_render": [{"text": "a red fox"}]})
    proof("a red fox in snow", [], 1.0, None)
    proof("a red fox in snow", [], 5.0, NEGATIVE_T2I)
    proof(caption, [], 5.0, NEGATIVE_T2I)
    proof("make it night", pictures[:1], 5.0, None)
    proof(json.dumps({"short_description": "a fox at night", "edit_instruction": "make it night"}), pictures[:1], 1.0, None)
    proof("put the fox from <image_2> next to the fox in <image_1>", pictures, 1.0, None)
    with tempfile.TemporaryDirectory() as scratch:
        functions(Path(scratch))
    print("fibo-2 model proof passed")
