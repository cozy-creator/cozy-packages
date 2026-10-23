#!/usr/bin/env python
"""Real tiny Klein modules on CPU: each function makes the image Diffusers' Flux3 modular pipeline
for its checkpoint makes, `generate` Klein Base's with CFG and `generate_turbo` the distilled flow
map's, text to image and with one or two references, through the Runtime test harness. BFL's
`flux3` package is read at `BFL_COMMIT`: its sampling constants are the package's, and its DiT and
samplers, on the weights Diffusers' converter maps into the package's modules, denoise to the
package's tokens.

Run with the package's own interpreter.
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import msgspec
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

#: BFL's reference code (black-forest-labs/flux-3-image-klein), its current inference parameters.
BFL_COMMIT = "bfc8889a4497f846416f253a3972a1941ac698b9"
BFL_FILES = ("__init__", "autoencoder", "checkpoint", "dit", "fp8", "nvfp4", "pipeline", "sampling", "text_encoder")


def _fetch_bfl() -> Path:
    root = Path(tempfile.mkdtemp(prefix="bfl-flux3-"))
    (root / "flux3").mkdir()
    for name in BFL_FILES:
        url = f"https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/{BFL_COMMIT}/src/flux3/{name}.py"
        with urllib.request.urlopen(url, timeout=60) as response:
            (root / "flux3" / f"{name}.py").write_bytes(response.read())
    return root


sys.path.insert(0, str(_fetch_bfl()))

import flux_3_klein  # noqa: E402
from diffusers import ClassifierFreeGuidance, FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers  # noqa: E402
from flux3 import autoencoder as bfl_autoencoder  # noqa: E402
from flux3 import dit as bfl_dit  # noqa: E402
from flux3 import pipeline as bfl_pipeline  # noqa: E402
from flux3 import sampling as bfl_sampling  # noqa: E402
from flux3 import text_encoder as bfl_text  # noqa: E402
from flux_3_klein import _BASE, _DEMAND, _DISTILLED, Megapixels, app, output_size  # noqa: E402
from flux_3_klein import klein  # noqa: E402
from flux_3_klein.upsampler import Drafter, Upsampler  # noqa: E402
from flux_3_klein.klein import (  # noqa: E402
    ALPHA,
    PATCH,
    TEXT_LAYERS,
    Klein,
    KleinGraph,
    build_processor,
    image_ids,
    prepare_multi_reference,
    prepare_reference,
    reference_canvas,
    text_ids,
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


def graph(flow_map: bool, transformer: dict[str, object] | None = None) -> KleinGraph:
    config = {
        "transformer": transformer or dict(
            in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16,
            num_attention_heads=2, joint_attention_dim=16 * len(TEXT_LAYERS), timestep_channels=16,
            axes_dims_rope=[4, 4, 4, 4], flow_map=flow_map),
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
    # Diffusers' blocks at BFL's current alpha and CFG (they default to the release's 6.93 and 3.5).
    guidance = {} if flow_map else {"guider": ClassifierFreeGuidance(guidance_scale=bfl_pipeline.MODELS["base"]["guidance"])}
    pipe.update_components(**{**built.components, "transformer": reference}, tokenizer=model.processor,
                           scheduler=FlowMatchEulerDiscreteScheduler(shift=ALPHA), **guidance)
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


def constants() -> None:
    """BFL's sampling, text and reference parameters are the package's, and each function's defaults
    are its checkpoint's (`MODELS`: steps, and CFG for the base model)."""
    assert ALPHA == inspect.signature(bfl_pipeline.Pipeline.generate).parameters["alpha"].default, ALPHA
    assert klein.IMAGE_ROPE_SCALE == bfl_pipeline.IMAGE_ROPE_SCALE
    assert klein.MATCH_MAX_PIXELS == bfl_pipeline.MATCH_MAX_PIXELS
    assert klein.MULTI_REFERENCE_MAX_PIXELS == bfl_pipeline.MULTI_REFERENCE_MAX_PIXELS
    assert (klein.MIN_SIDE, klein.MAX_SIDE) == (bfl_pipeline.MIN_SIDE, bfl_pipeline.MAX_SIDE)
    assert PATCH == bfl_autoencoder.DOWNSAMPLING_RATIO
    assert TEXT_LAYERS == bfl_text.OUTPUT_LAYERS
    assert (klein.PAD_MULTIPLE, klein.MAX_TOKENS) == (bfl_text.PAD_MULTIPLE, bfl_text.MAX_LENGTH)
    inputs = [cls for cls in vars(flux_3_klein).values()
              if isinstance(cls, type) and issubclass(cls, msgspec.Struct) and "steps" in cls.__struct_fields__]
    assert inputs
    for cls in inputs:
        defaults = dict(zip(cls.__struct_fields__[-len(cls.__struct_defaults__):], cls.__struct_defaults__, strict=True))
        expected = bfl_pipeline.MODELS["base" if "guidance" in defaults else "distilled"]
        assert defaults["steps"] == expected["steps"], (cls.__name__, defaults["steps"])
        assert defaults.get("guidance") == expected["guidance"], (cls.__name__, defaults.get("guidance"))


def _bfl_dit(flow_map: bool) -> bfl_dit.DiT:
    """BFL's DiT at a tiny size (its module constants are its architecture)."""
    bfl_dit.HIDDEN, bfl_dit.HEADS, bfl_dit.DEPTH, bfl_dit.DEPTH_SINGLE = 32, 2, 1, 2
    bfl_dit.AXES_DIM = (4, 4, 4, 4)
    bfl_dit.IN_CHANNELS.update(image=8, image_cond=8, text=16)
    torch.manual_seed(13)
    dit = bfl_dit.DiT(flow_map)
    for parameter in dit.parameters():
        with torch.no_grad():
            parameter.normal_(0, 0.3)
    return dit.to(torch.bfloat16).eval()


def bfl(flow_map: bool, refs: int, guidance: float | None = None) -> float:
    """The package's denoise against BFL's sampler and DiT on the same converted weights: the mean
    relative difference of the denoised tokens."""
    reference_dit = _bfl_dit(flow_map)
    built = graph(flow_map, dict(in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=2,
                                 attention_head_dim=16, num_attention_heads=2, joint_attention_dim=16,
                                 timestep_channels=256, axes_dims_rope=[4, 4, 4, 4], mlp_ratio=3.0,
                                 flow_map=flow_map))
    state = convert_flux3_transformer_checkpoint_to_diffusers(dict(reference_dit.state_dict()))
    built.components["transformer"].load_state_dict(state, strict=True)
    model = Klein.for_test(graph=built, processor=build_processor(assets()))

    rows, columns, steps, seed = 3, 4, 4, 11
    generator = torch.Generator().manual_seed(5)
    texts = [torch.randn((8, 16), generator=generator).to(torch.bfloat16) for _ in range(2)]
    ids = bfl_pipeline.scaled_ids(bfl_autoencoder.latent_ids(rows, columns))
    assert torch.equal(ids, image_ids(rows, columns))
    cond, ref_ids = None, None
    if refs:
        ref_ids = torch.cat([image_ids(2, 3, -(1 + n)) for n in range(refs)])
        cond = (torch.randn((len(ref_ids), 8), generator=generator).to(torch.bfloat16), ref_ids)
    noise = torch.randn((rows * columns, 8), generator=torch.Generator().manual_seed(seed))
    image = (noise.to(torch.bfloat16).float(), ids)
    text = [(t, text_ids(len(t))) for t in texts]
    schedule = bfl_sampling.timesteps(steps, inspect.signature(bfl_pipeline.Pipeline.generate).parameters["alpha"].default)
    with torch.autocast("cpu", dtype=torch.bfloat16):  # BFL samples under bf16 autocast (on CUDA)
        if flow_map:
            expected = bfl_sampling.denoise_flow_map(reference_dit, bfl_pipeline.pack([[text[1], cond, image]]), schedule)
        else:
            packed = bfl_pipeline.pack([[text[0], cond, image], [text[1], cond, image]])
            expected = bfl_sampling.denoise(reference_dit, packed, schedule, guidance)
    references = None if cond is None else (cond[0][None], ref_ids)
    actual = model.denoise(texts[1][None], None if flow_map else texts[0][None], references,
                           width=columns * PATCH, height=rows * PATCH, steps=steps, guidance=guidance, seed=seed,
                           on_step=lambda _: None, cancel=lambda: None)[0]
    return float((actual - expected.float()).abs().mean() / expected.float().abs().mean())


def bfl_parity() -> None:
    """Each pass agrees to bf16's rounding; CFG scales that by its guidance (|Δ| grows with it)."""
    cases = [(True, None, 0.02), (False, 0.0, 0.02), (False, 1.0, 0.02),
             (False, bfl_pipeline.MODELS["base"]["guidance"], 0.05)]
    for flow_map, guidance, bound in cases:
        for refs in (0, 2):
            difference = bfl(flow_map, refs, guidance)
            print(f"BFL flow_map={flow_map} guidance={guidance} references={refs}: mean relative |Δ| = "
                  f"{difference:.4f}")
            assert difference < bound, difference
    # A schedule off BFL's alpha is caught.
    klein.ALPHA = 6.93 if ALPHA != 6.93 else 16.0
    try:
        off = bfl(True, 0)
    finally:
        klein.ALPHA = ALPHA
    print(f"BFL flow_map=True at alpha {6.93 if ALPHA != 6.93 else 16.0}: mean relative |Δ| = {off:.4f}")
    assert off > 0.1, off


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
    constants()
    bfl_parity()
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
