#!/usr/bin/env python
"""Real tiny Qwen-Image-Edit modules: the package's staged model agrees exactly with Diffusers'
`QwenImageEditPlusPipeline`, and a Lightning-format LoRA with Diffusers' own reading of it.

Run with the package's own interpreter: qwen-image-edit/.venv/bin/python scripts/qwen-image-edit-model-proof.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cozy_runtime.author import Config, Model, ModelAssets
from cozy_runtime.author.fakes import granted_input, invoke_with_fakes
from cozy_runtime.internal.weights import LoRASource
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers
from PIL import Image
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen-image-edit"))

from qwen_image_edit import app  # noqa: E402
from qwen_image_edit.model import (  # noqa: E402
    LIGHTNING_SCHEDULER,
    LORA_COMPONENT,
    LORA_SITES,
    LoRAGraph,
    QwenImageEdit,
    QwenImageEditGraph,
    QwenImageEditLoRA,
    build_processor,
)

WIDTH, HIDDEN, BLOCKS, RANK = 32, 128, 2, 4
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
            "<|video_pad|>"]


def processor_files() -> dict[str, bytes]:
    """A character-level tokenizer with Qwen's special tokens, and Qwen2-VL's image geometry."""
    vocab = {"[UNK]": 0, "[PAD]": 1, **{t: 2 + i for i, t in enumerate(SPECIALS)}}
    for code in range(32, 127):
        vocab.setdefault(chr(code), len(vocab))
    vocab["\n"] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tokenizer.add_special_tokens(SPECIALS)
    image = {"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2, "min_pixels": 3136,
             "max_pixels": 12845056, "image_processor_type": "Qwen2VLImageProcessorFast"}
    chat = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    return {
        "processor/tokenizer.json": tokenizer.to_str().encode(),
        "processor/tokenizer_config.json": json.dumps(
            {"unk_token": "[UNK]", "pad_token": "[PAD]", "tokenizer_class": "PreTrainedTokenizerFast",
             "chat_template": chat}
        ).encode(),
        "processor/preprocessor_config.json": json.dumps(image).encode(),
        "processor/video_preprocessor_config.json": json.dumps(
            {**{k: v for k, v in image.items() if k != "image_processor_type"},
             "video_processor_type": "Qwen2VLVideoProcessor"}
        ).encode(),
        "processor/chat_template.jinja": chat.encode(),
    }


def graph_and_processor() -> tuple[QwenImageEditGraph, Qwen2VLProcessor, dict[str, object]]:
    files = processor_files()
    tokens = Tokenizer.from_str(files["processor/tokenizer.json"].decode())
    ids = {t: tokens.token_to_id(t) for t in SPECIALS}
    config: dict[str, object] = {
        "transformer": dict(
            patch_size=2, in_channels=16, out_channels=4, num_layers=BLOCKS, attention_head_dim=16,
            num_attention_heads=2, joint_attention_dim=16, axes_dims_rope=[4, 6, 6], zero_cond_t=True,
        ),
        "text_encoder": dict(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
            vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
            text_config=dict(
                vocab_size=tokens.get_vocab_size(), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                rope_parameters=dict(rope_type="default", rope_theta=1e6, mrope_section=[1, 1, 2]),
            ),
            vision_config=dict(
                depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=16,
                patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56,
                fullatt_block_indexes=[0],
            ),
        ),
        "vae": dict(
            base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1, attn_scales=[],
            temperal_downsample=[False, True, True],
            latents_mean=[0.1, -0.2, 0.3, 0.05], latents_std=[1.5, 0.8, 2.0, 1.1],
        ),
        "scheduler": dict(
            FlowMatchEulerDiscreteScheduler(
                base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9,
                shift_terminal=0.02, time_shift_type="exponential", use_dynamic_shifting=True,
            ).config
        ),
    }
    torch.manual_seed(61)
    graph = QwenImageEditGraph(Config(config))
    for root in graph.components.values():
        for p in root.parameters():
            with torch.no_grad():
                p.normal_(0, 0.05)
        root.eval()
    processor = build_processor(ModelAssets(files, lambda _: None), {})
    # The checkpoint's processor folder parsed in memory equals Transformers' own loading of it.
    with tempfile.TemporaryDirectory() as folder:
        for name, raw in files.items():
            (Path(folder) / name.removeprefix("processor/")).write_bytes(raw)
        loaded = Qwen2VLProcessor.from_pretrained(folder, local_files_only=True)
    text = "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>blue<|im_end|>"
    image = Image.linear_gradient("L").resize((90, 60)).convert("RGB")
    ours, theirs = (p(text=[text], images=[image], return_tensors="pt") for p in (processor, loaded))
    for key in theirs:
        assert torch.equal(ours[key], theirs[key]), key
    assert isinstance(processor.image_processor, Qwen2VLImageProcessor)
    assert isinstance(processor.video_processor, Qwen2VLVideoProcessor)
    assert isinstance(processor.tokenizer, PreTrainedTokenizerFast)
    return graph, processor, config


def references() -> list[Image.Image]:
    first = Image.merge("RGB", [Image.linear_gradient("L").resize((96, 64)).rotate(r) for r in (0, 90, 180)])
    second = Image.linear_gradient("L").resize((64, 80)).convert("RGB")
    return [first, second]


def run(
    model: QwenImageEdit, lora: QwenImageEditLoRA | None, prompt: str, images: list[Image.Image],
    *, width: int, height: int, steps: int, guidance: float, negative: str, seed: int,
) -> np.ndarray:
    refs = model.encode_references(images) if images else None
    embeds, mask = model.encode(prompt, images)
    unconditional = model.encode(negative, images) if guidance > 1 else None
    denoise = model.denoise if lora is None else partial(lora.denoise, model)
    latents = denoise(embeds, mask, unconditional, refs, width=width, height=height, steps=steps,
                      guidance=guidance, seed=seed, on_step=lambda _: None, cancel=lambda: None)
    return np.asarray(model.decode(latents, width=width, height=height))


def upstream(
    pipe: QwenImageEditPlusPipeline, prompt: str, images: list[Image.Image], *, seed: int, **kw: object
) -> np.ndarray:
    out = pipe(image=images, prompt=prompt, generator=torch.Generator().manual_seed(seed),
               output_type="pil", **kw).images[0]
    return np.asarray(out)


def lightning_state(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A LoRA in lightx2v's Lightning format: lora_down/lora_up/alpha per site."""
    generator = torch.Generator().manual_seed(7)
    state: dict[str, torch.Tensor] = {}
    for index in range(BLOCKS):
        for site in LORA_SITES:
            linear = transformer.get_submodule(f"transformer_blocks.{index}.{site}")
            key = f"transformer_blocks.{index}.{site}"
            state[f"{key}.lora_down.weight"] = torch.randn(RANK, linear.in_features, generator=generator) * 0.3
            state[f"{key}.lora_up.weight"] = torch.randn(linear.out_features, RANK, generator=generator) * 0.3
            state[f"{key}.alpha"] = torch.tensor(1, dtype=torch.int64)  # scale alpha/rank = 1/4
    return state


def staged() -> None:
    graph, processor, config = graph_and_processor()
    model = QwenImageEdit.for_test(graph=graph, processor=processor)
    transformer = graph.components["transformer"]
    hooks = [h for m in transformer.modules() for h in m._forward_hooks.values()]
    assert len(hooks) == BLOCKS * len(LORA_SITES) and all(isinstance(h, LoRASource) for h in hooks)
    # Upstream runs every module in its prompt's dtype (it casts reference pixels to it), so the
    # exact comparison runs all three in float32.
    for module in graph.components.values():
        module.float()
    pipe = QwenImageEditPlusPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config["scheduler"]),
        vae=graph.components["vae"], text_encoder=graph.components["text_encoder"],
        tokenizer=processor.tokenizer, processor=processor, transformer=transformer,
    )
    images = references()
    # Editing with two ordered references and true CFG, and a lone reference without it.
    for refs, guidance, size in ((images, 4.0, (64, 48)), (images[1:], 1.0, (48, 64))):
        ours = run(model, None, "make the sky red", refs, width=size[0], height=size[1], steps=2,
                   guidance=guidance, negative=" ", seed=5)
        theirs = upstream(pipe, "make the sky red", refs, width=size[0], height=size[1],
                          num_inference_steps=2, true_cfg_scale=guidance,
                          negative_prompt=" " if guidance > 1 else None, seed=5)
        assert ours.shape == (size[1], size[0], 3), ours.shape
        np.testing.assert_array_equal(ours, theirs)
    # Text to image: the instruction with no picture, the target alone in the DiT.
    plain = run(model, None, "a red bicycle", [], width=48, height=32, steps=2, guidance=4.0,
                negative=" ", seed=9)
    assert plain.shape == (32, 48, 3) and plain.std() > 0
    np.testing.assert_array_equal(plain, run(model, None, "a red bicycle", [], width=48, height=32,
                                             steps=2, guidance=4.0, negative=" ", seed=9))

    # Lightning: the LoRA's sites, scale and schedule are Diffusers' reading of the same file.
    state = lightning_state(transformer)
    lora_graph = LoRAGraph(Config({}), width=WIDTH, hidden=HIDDEN, blocks=BLOCKS, rank=RANK)
    lora_root = lora_graph.components[LORA_COMPONENT]
    with torch.no_grad():  # as TensorFS ingests the kohya file: lora_down/lora_up as lora_A/lora_B
        for key, value in lora_root.state_dict().items():
            value.copy_(state[key.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up.")])
    lora = QwenImageEditLoRA.for_test(graph=lora_graph)
    fused = copy.deepcopy(transformer)
    converted = _convert_non_diffusers_qwen_lora_to_diffusers(dict(state))
    with torch.no_grad():
        for key, down in converted.items():
            if key.endswith(".lora_A.weight"):
                site = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
                up = converted[key.replace(".lora_A.", ".lora_B.")]
                fused.get_submodule(site).weight += up @ down  # PEFT's scaling is 1 once folded
    for hook in hooks:
        hook._cozy_baked = False
    reference = QwenImageEditPlusPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER),
        vae=graph.components["vae"], text_encoder=graph.components["text_encoder"],
        tokenizer=processor.tokenizer, processor=processor, transformer=fused,
    )
    theirs = upstream(reference, "put a hat on it", images, width=64, height=48,
                      num_inference_steps=2, true_cfg_scale=1.0, seed=3)
    # Per step (the hook adds the update) equals the fused LoRA to float32 rounding.
    per_step = run(model, lora, "put a hat on it", images, width=64, height=48, steps=2,
                   guidance=1.0, negative="", seed=3)
    assert np.abs(per_step.astype(int) - theirs.astype(int)).max() <= 1
    # Baked, as Runtime does with `lora_factors()`: weight += scale * up @ down, hooks quiet.
    baked = copy.deepcopy(transformer.state_dict())
    model.graph.arming.lora = lora_root
    with torch.no_grad():
        for hook in hooks:
            _, down, up, scale = hook.lora_factors()
            transformer.get_submodule(hook.path).weight += scale * (up @ down)
            hook._cozy_baked = True
    model.graph.arming.lora = None
    bake = run(model, lora, "put a hat on it", images, width=64, height=48, steps=2, guidance=1.0,
               negative="", seed=3)
    np.testing.assert_array_equal(bake, theirs)
    transformer.load_state_dict(baked)
    for hook in hooks:
        hook._cozy_baked = False
    assert model.graph.arming.lora is None
    np.testing.assert_array_equal(per_step, run(model, lora, "put a hat on it", images, width=64,
                                                height=48, steps=2, guidance=1.0, negative="", seed=3))
    kernel(model, lora)


def kernel(model: QwenImageEdit, lora: QwenImageEditLoRA) -> None:
    """Both functions through the real kernel on real files: the output takes the first
    reference's ratio and pixel count, an explicit value only its own, and out of range clamps."""
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)

        def photo(name: str, size: tuple[int, int]) -> Path:
            Image.linear_gradient("L").resize(size).convert("RGB").save(tmp / name)
            return tmp / name

        wide, square = photo("3x2.png", (600, 400)), photo("sq.png", (512, 512))
        cases: list[tuple[str, list[Path], dict[str, object], tuple[int, int], list[tuple[str, str]]]] = [
            ("generate", [wide, square], {}, (608, 400), []),
            ("generate", [square, wide], {}, (512, 512), []),
            ("generate", [photo("big.png", (2000, 1500))], {}, (1680, 1248), [("clamp", "megapixels")]),
            ("generate", [photo("small.png", (200, 150))], {}, (288, 224), [("clamp", "megapixels")]),
            ("generate", [wide], {"aspect_ratio": "1:1"}, (496, 496), []),
            ("generate", [wide], {"megapixels": 1}, (1248, 832), []),
            ("generate", [], {}, (1024, 1024), []),
            ("generate", [], {"aspect_ratio": "16:9"}, (1360, 768), []),
            ("generate", [], {"megapixels": 4}, (1456, 1456), [("clamp", "megapixels")]),
            ("generate", [], {"aspect_ratio": "20:1"}, (4096, 256), [("clamp", "aspect_ratio")]),
            ("generate_turbo", [wide], {"steps": 1}, (608, 400), []),
        ]
        for index, (function, paths, given, expected, clamps) in enumerate(cases):
            grants = {
                f"reference_images.{n}": granted_input(path, input_id=f"reference_images.{n}", order=n)
                for n, path in enumerate(paths)
            }
            wire: dict[str, object] = {"prompt": "a person", "steps": 2, "seed": 7, **given}
            wire["reference_images"] = [grant.digest for grant in grants.values()]
            models: dict[str, Model[Any]] = {"model": model}
            if function == "generate_turbo":
                models["turbo_lora"] = lora
                wire["guidance"] = 3.0  # not a turbo input: warned and dropped
            result = invoke_with_fakes(app.get(function), wire, request_id=f"size-{index}",
                                       spool=tmp / f"size-{index}", models=models, assets=grants)
            image = result.result
            assert (image.width, image.height) == expected, (function, paths, given, image)
            assert image.seed == 7 and image.prompt == "a person"
            with Image.open(tmp / f"size-{index}" / "image-0001") as png:
                assert (png.format, png.size) == ("PNG", expected)
            rows = [(row.kind, row.field) for row in result.adjustments if row.kind == "clamp"]
            assert rows == clamps, (given, result.adjustments)
            if function == "generate_turbo":
                print("turbo adjustments", result.adjustments)
    assert model.graph.arming.lora is None


if __name__ == "__main__":
    with torch.inference_mode(False):
        staged()
    print("qwen-image-edit model: staged execution equals Diffusers' QwenImageEditPlusPipeline")
