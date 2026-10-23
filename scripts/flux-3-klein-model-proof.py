#!/usr/bin/env python
"""Real tiny Klein modules on CPU: each function makes the image Diffusers' Flux3 modular pipeline
for its checkpoint makes, `generate` Klein Base's with CFG and `generate_turbo` the distilled flow
map's, text to image and with one or two references, through the Runtime test harness.

Run with the package's own interpreter.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cozy_runtime.author import Config, ModelAssets
from cozy_runtime.author.fakes import granted_input, invoke_with_fakes
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

from flux_3_klein import _BASE, _DEMAND, _DISTILLED, Megapixels, app, output_size  # noqa: E402
from flux_3_klein.upsampler import Drafter, Upsampler  # noqa: E402
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

#: Each function's checkpoint: Klein Base (CFG) or the distilled flow map.
CHECKPOINTS = {"generate": (_BASE, False), "generate_turbo": (_DISTILLED, True)}

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


def run(function: str, flow_map: bool, references: list[Path], root: Path,
        **given: object) -> tuple[Any, Path]:
    """`function` through the real kernel on tiny real modules, the upsampler off."""
    models = {
        "model": Klein.for_test(graph=graph(flow_map), processor=build_processor(assets())),
        "upsampler": Upsampler.for_test(),
        "drafter": Drafter.for_test(),
    }
    grants = {
        f"reference_images.{n}": granted_input(path, input_id=f"reference_images.{n}", order=n)
        for n, path in enumerate(references)
    }
    wire = {"prompt": "a red fox in snow", "seed": 11, "prompt_upsampling": False, **given,
            "reference_images": [grant.digest for grant in grants.values()]}
    spool = root / f"{function}-{len(references)}-{len(list(root.iterdir()))}"
    return invoke_with_fakes(app.get(function), wire, request_id=spool.name, spool=spool, models=models,
                             assets=grants), spool / "image-0001"


def proof(function: str, references: list[Path], root: Path) -> None:
    defaults, flow_map = CHECKPOINTS[function]
    assert app.get(function).model_defaults["model"] == defaults["model"], app.get(function).model_defaults
    invoked, png = run(function, flow_map, references, root, steps=3)
    width, height = invoked.result.width, invoked.result.height
    with Image.open(png) as image:
        actual = np.asarray(image.convert("RGB")).astype(np.float32) / 255

    built = graph(flow_map)
    model = Klein.for_test(graph=built, processor=build_processor(assets()))
    pictures = [Image.open(path) for path in references]
    prepared = ([prepare_reference(pictures[0])[0]] if len(pictures) == 1
                else [prepare_multi_reference(p) for p in pictures])
    pipe = (Flux3DistilledAutoBlocks() if flow_map else Flux3AutoBlocks()).init_pipeline()
    # Diffusers' own blocks, on the same weights: Klein runs every block as one stream.
    reference = copy.deepcopy(built.components["transformer"])
    for block in reference.single_transformer_blocks:
        block.__class__ = Flux3SingleTransformerBlock
    for block in (*reference.txt_mode_blocks, *(b for m in reference.mode_blocks.values() for b in m)):
        block.__class__ = Flux3ModeTransformerBlock
    pipe.update_components(**{**built.components, "transformer": reference}, tokenizer=model.processor)
    # An edit generates on the first reference's 16 px canvas (64 px sides at least) and crops back.
    fit, (width, height) = reference_canvas(*pictures[0].size) if pictures else ((width, height),) * 2
    left, top = (width - fit[0]) // 2, (height - fit[1]) // 2
    rows, columns = height // PATCH, width // PATCH
    noise = torch.randn((rows * columns, 8), generator=torch.Generator().manual_seed(11))
    latents = noise.to(torch.bfloat16).float().view(rows, columns, 8).permute(2, 0, 1)[None]
    kwargs = dict(prompt="a red fox in snow", height=height, width=width, num_inference_steps=3,
                  latents=latents, output="images", output_type="np")
    if prepared:
        kwargs["image"] = prepared
    expected = pipe(**kwargs)[0][top:top + fit[1], left:left + fit[0]]
    difference = float(np.abs(actual - expected).mean())
    print(f"{function} references={len(references)} {fit[0]}x{fit[1]}: mean |Δ| = {difference:.4f}")
    # Matching is 0.002 here; the same base model without CFG is 0.005 off, so this tells them apart.
    assert difference < 0.0035, difference


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


def sizing(root: Path) -> None:
    """With references, what is not given is the first reference's: given neither, its own size
    (generated on the 16 px grid and cropped back); given one, its ratio or its pixel count."""
    rng = np.random.default_rng(5)
    paths = []
    for index, size in enumerate(((100, 60), (90, 50), (200, 200), (40, 40))):
        path = root / f"reference-{index}.png"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(path)
    cases: list[tuple[list[Path], dict[str, object], tuple[int, int], set[str]]] = [
        ([paths[0]], {}, (100, 60), set()),  # canvas 112x64, cropped back to the reference
        ([paths[1], paths[2]], {}, (90, 50), set()),  # several: the first's own size too
        ([paths[0]], {"aspect_ratio": "2:1", "megapixels": 1}, (1440, 736), set()),
        ([paths[0]], {"megapixels": 1}, (1312, 800), set()),  # its 5:3 at the tier given
        ([paths[2]], {"aspect_ratio": "2:1"}, (288, 128), set()),  # its 40,000 px at the ratio given
        ([paths[3]], {"aspect_ratio": "1:1"}, (64, 64), {"megapixels"}),  # raised to 64x64
    ]
    root = root / "sizing"
    root.mkdir()
    for references, given, expected, clamped in cases:
        for function, (_, flow_map) in CHECKPOINTS.items():
            invoked, png = run(function, flow_map, references, root, steps=1, **given)
            assert (invoked.result.width, invoked.result.height) == expected, (function, given, invoked.result)
            with Image.open(png) as image:
                assert image.size == expected, image.size
            assert {row.field for row in invoked.adjustments} == clamped, invoked.adjustments


if __name__ == "__main__":
    sizes()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        sizing(root)
        rng = np.random.default_rng(3)
        pictures = []
        for index in range(2):
            path = root / f"picture-{index}.png"
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(path)
            pictures.append(path)
        for function in CHECKPOINTS:
            for count in (0, 1, 2):
                proof(function, pictures[:count], root)
    print("flux-3-klein model proof passed")
