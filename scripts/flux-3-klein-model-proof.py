#!/usr/bin/env python
"""Real tiny Klein modules on CPU: the staged model matches Diffusers' Flux3 modular pipeline.

Run with the package's own interpreter. Both checkpoints' samplers (the distilled flow map and the
base model with CFG), text-to-image and two references, through the Runtime test harness.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from cozy_runtime.author import Config, ModelAssets
from diffusers.models.transformers.transformer_flux3 import (
    Flux3ModeTransformerBlock,
    Flux3SingleTransformerBlock,
)
from diffusers.modular_pipelines import Flux3AutoBlocks, Flux3DistilledAutoBlocks
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flux-3-klein"))

from flux_3_klein import _DEMAND, Megapixels, output_size  # noqa: E402
from flux_3_klein.klein import (  # noqa: E402
    PATCH,
    TEXT_LAYERS,
    Klein,
    KleinGraph,
    build_processor,
    prepare_multi_reference,
    prepare_reference,
    reference_canvas,
)

CHAT = (
    '{% for message in messages %}{{"<|im_start|>" + message["role"] + "\\n"}}'
    '{% if message["content"] is string %}{{message["content"]}}{% else %}'
    '{% for item in message["content"] %}{{item["text"]}}{% endfor %}{% endif %}'
    '{{"<|im_end|>\\n"}}{% endfor %}{% if add_generation_prompt %}{{"<|im_start|>assistant\\n"}}{% endif %}'
)


def assets() -> ModelAssets:
    words = "a red fox in snow at dawn the same at night [UNK] [PAD] <|im_start|> <|im_end|> user assistant"
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words.split()))}
    for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"):
        vocab[token] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    geometry = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}
    files = {
        "tokenizer.json": PreTrainedTokenizerFast(tokenizer_object=t).backend_tokenizer.to_str().encode(),
        "tokenizer_config.json": json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]",
                                             "tokenizer_class": "Qwen2Tokenizer"}).encode(),
        "preprocessor_config.json": json.dumps({**geometry, "image_processor_type": "X"}).encode(),
        "video_preprocessor_config.json": json.dumps(geometry).encode(),
        "chat_template.json": json.dumps({"chat_template": CHAT}).encode(),
    }
    return ModelAssets({f"text_encoder/{k}": v for k, v in files.items()}, lambda _operation: None)


def graph(flow_map: bool) -> KleinGraph:
    config = {
        "transformer": dict(in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
                            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=16 * len(TEXT_LAYERS),
                            timestep_channels=16, axes_dims_rope=[4, 4, 4, 4], flow_map=flow_map),
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
            tie_word_embeddings=True,
        ),
    }
    torch.manual_seed(7)
    built = KleinGraph(Config(config))
    for root in built.components.values():
        for p in root.parameters():
            with torch.no_grad():
                p.normal_(0, 0.2)
    vae = built.components["vae"]
    vae.bn.running_mean.normal_(0, 0.1)
    vae.bn.running_var.uniform_(0.5, 1.5)
    assert len(built.components["text_encoder"].model.language_model.layers) == max(TEXT_LAYERS)
    return built


def proof(flow_map: bool, references: list[Image.Image]) -> None:
    built = graph(flow_map)
    model = Klein.for_test(graph=built, processor=build_processor(assets()))
    width = height = 64
    prepared = ([prepare_reference(references[0])[0]] if len(references) == 1
                else [prepare_multi_reference(r) for r in references])
    encoded = model.encode_references(prepared) if references else None
    embeds = model.encode("a red fox in snow")
    negative = None if flow_map else model.encode("")
    tokens = model.denoise(embeds, negative, encoded, width=width, height=height, steps=3,
                           guidance=None if flow_map else 3.5, seed=11,
                           on_step=lambda _: None, cancel=lambda: None)
    image = model.decode(tokens, width=width, height=height)
    assert image.size == (width, height)

    blocks = Flux3DistilledAutoBlocks() if flow_map else Flux3AutoBlocks()
    pipe = blocks.init_pipeline()
    # Diffusers' own blocks, on the same weights: Klein runs every block as one stream.
    reference = copy.deepcopy(built.components["transformer"])
    for block in reference.single_transformer_blocks:
        block.__class__ = Flux3SingleTransformerBlock
    for block in (*reference.txt_mode_blocks, *(b for m in reference.mode_blocks.values() for b in m)):
        block.__class__ = Flux3ModeTransformerBlock
    pipe.update_components(**{**built.components, "transformer": reference}, tokenizer=model.processor)
    rows = width // PATCH
    noise = torch.randn((rows * rows, 8), generator=torch.Generator().manual_seed(11))
    latents = noise.to(torch.bfloat16).float().view(rows, rows, 8).permute(2, 0, 1)[None]
    kwargs = dict(prompt="a red fox in snow", height=height, width=width, num_inference_steps=3,
                  latents=latents, output="images", output_type="np")
    if references:
        kwargs["image"] = prepared
    expected = pipe(**kwargs)[0]
    actual = np.asarray(image).astype(np.float32) / 255
    difference = float(np.abs(actual - expected).mean())
    print(f"flow_map={flow_map} references={len(references)}: mean |Δ| = {difference:.4f}")
    assert difference < 0.02, difference


def sizes() -> None:
    """Any W:H at the tier's area, rounded to 32 px, inside BFL's 64-4096 px sides and the demand
    admission declared; held only past them."""
    assert output_size(Megapixels.MP1, "1:1") == (1024, 1024, False)
    assert output_size(Megapixels.MP1, "3000:2000") == (1248, 832, False)
    assert output_size(Megapixels.MP4, "16:9") == (2720, 1536, False)
    assert output_size(Megapixels.MP4, "8:1") == (4096, 1024, True)
    for tier in Megapixels:
        for step in range(-500, 501):
            w, h = round(10**4 * 10 ** (step / 125)), 10**4
            width, height, held = output_size(tier, f"{w}:{h}")
            assert width % 32 == height % 32 == 0 and 64 <= min(width, height) <= max(width, height) <= 4096
            assert width * height <= _DEMAND[tier][0] * _DEMAND[tier][1], (tier, w, h)
            assert output_size(tier, f"{h}:{w}")[:2] == (height, width)
            if max(w / h, h / w) <= min(4096**2 / (tier << 20), 13):
                assert not held, (tier, w, h)
    fit, canvas = reference_canvas(3000, 2000)
    assert canvas[0] * canvas[1] <= 4 * 1024 * 1024 and canvas[0] % 16 == canvas[1] % 16 == 0
    padded, box = prepare_reference(Image.new("RGB", (1000, 1001)))
    assert padded.size == (1008, 1008) and box is not None
    assert prepare_multi_reference(Image.new("RGB", (2000, 1000))).size == (1440, 720)


if __name__ == "__main__":
    sizes()
    rng = np.random.default_rng(3)
    pictures = [Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)) for _ in range(2)]
    for flow_map in (True, False):
        for count in (0, 1, 2):
            proof(flow_map, pictures[:count])
    print("flux-3-klein model proof passed")
