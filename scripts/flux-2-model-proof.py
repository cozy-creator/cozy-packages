#!/usr/bin/env python
"""Real tiny FLUX.2 modules on CPU: the staged model matches Diffusers' `Flux2Pipeline` and
`Flux2KleinPipeline` on the same weights.

Run with the package's own interpreter. [dev] (embedded guidance), its Turbo LoRA (fal's sigmas, the
factors merged into a copy of the weights as the reference), [klein] distilled and [klein] Base (CFG),
text-to-image and two references; then `generate` and `generate_turbo` through the Runtime kernel.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from cozy_runtime.author import Config
from cozy_runtime.author.fakes import granted_input, invoke_with_fakes
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Pipeline
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PixtralImageProcessor, PixtralProcessor, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flux-2"))

from flux_2 import app, output_size  # noqa: E402
from flux_2.model import (  # noqa: E402
    TEXT_LAYERS,
    TURBO_SIGMAS,
    Flux2,
    Flux2Graph,
    prepare_references,
    schedule,
    text_encoder,
)
from flux_2.turbo import Flux2TurboLoRA, TurboGraph  # noqa: E402
from flux_2.upsampler import Flux2Upsampler  # noqa: E402

CHAT = (
    "{% for message in messages %}{{'<s>' + message['role'] + ' '}}"
    "{% if message['content'] is string %}{{message['content']}}{% else %}"
    "{% for item in message['content'] %}{{item['text'] if item['type'] == 'text' else '[IMG]'}}{% endfor %}{% endif %}"
    "{{' </s> '}}{% endfor %}{% if add_generation_prompt %}{{'<s>assistant '}}{% endif %}"
)
HIDDEN = 16
GEOMETRY = dict(in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=3 * HIDDEN, timestep_guidance_channels=16, axes_dims_rope=[4, 4, 4, 4])
VAE = dict(block_out_channels=[8, 8, 8, 8], down_block_types=["DownEncoderBlock2D"] * 4,
           up_block_types=["UpDecoderBlock2D"] * 4, latent_channels=2, layers_per_block=1, norm_num_groups=4,
           patch_size=[2, 2])


def tokenizer() -> PreTrainedTokenizerFast:
    words = "<pad> <unk> <s> </s> [IMG] [IMG_BREAK] [IMG_END] system user assistant a red fox in snow the same at night"
    vocab = {word: i for i, word in enumerate(words.split())}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", pad_token="<pad>", bos_token="<s>",
                                   eos_token="</s>", chat_template=CHAT)


def processor(family: str) -> object:
    if family == "qwen3":
        return tokenizer()
    images = PixtralImageProcessor(patch_size=14, size={"longest_edge": 56})
    return PixtralProcessor(image_processor=images, tokenizer=tokenizer(), patch_size=14, spatial_merge_size=2,
                            chat_template=CHAT, image_token="[IMG]", image_break_token="[IMG_BREAK]",
                            image_end_token="[IMG_END]")


def text_config(family: str) -> dict[str, object]:
    text = dict(vocab_size=64, hidden_size=HIDDEN, intermediate_size=16, num_hidden_layers=max(TEXT_LAYERS[family]) + 2,
                num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    if family == "qwen3":
        return dict(model_type="qwen3", **text)
    vision = dict(model_type="pixtral", hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                  num_attention_heads=2, head_dim=8, patch_size=14, image_size=56)
    return dict(model_type="mistral3", text_config=dict(model_type="mistral", **text), vision_config=vision,
                image_token_index=4, spatial_merge_size=2)


def built(family: str, distilled: bool) -> tuple[Flux2, torch.nn.Module]:
    """The staged model on random weights, and the full text encoder Diffusers runs."""
    config = {"transformer": dict(GEOMETRY, guidance_embeds=family == "mistral3"), "vae": VAE,
              "text_encoder": text_config(family), "model_index": {"is_distilled": distilled}}
    torch.manual_seed(7)
    graph = Flux2Graph(Config(config))
    full_encoder = text_encoder(text_config(family))
    for root, std in ((graph.components["transformer"], 0.05), (graph.components["vae"], 0.2), (full_encoder, 0.2)):
        for p in root.parameters():
            with torch.no_grad():
                p.normal_(0, std)
    missing, _ = graph.components["text_encoder"].load_state_dict(full_encoder.state_dict(), strict=False)
    assert not missing, missing
    vae = graph.components["vae"]
    vae.bn.running_mean.normal_(0, 0.1)
    vae.bn.running_var.uniform_(0.5, 1.5)
    return Flux2.for_test(graph=graph, tokenizer=processor(family)), full_encoder.eval()


def turbo_overlay(transformer: torch.nn.Module) -> Flux2TurboLoRA:
    targets = [f"transformer.{p}" for p, m in transformer.named_modules() if isinstance(m, torch.nn.Linear)]
    graph = TurboGraph(Config({"model": {"r": 4, "lora_alpha": 8, "target_modules": targets}}), GEOMETRY)
    torch.manual_seed(3)
    for p in graph.components["model"].parameters():
        with torch.no_grad():
            p.normal_(0, 0.1)
    return Flux2TurboLoRA.for_test(graph=graph)


def merged(transformer: torch.nn.Module, overlay: Flux2TurboLoRA) -> torch.nn.Module:
    """The reference: a copy of the weights with W += scale · B·A, as PEFT's merge does."""
    reference = copy.deepcopy(transformer)
    for path, factors in overlay.graph.sites.items():
        linear = reference.get_submodule(path)
        with torch.no_grad():
            delta = factors.lora_B.weight.float() @ factors.lora_A.weight.float() * factors.scale
            linear.weight += delta.to(linear.weight.dtype)
    return reference


def proof(family: str, distilled: bool, turbo: bool, references: list[Image.Image]) -> None:
    model, encoder = built(family, distilled)
    graph = model.graph
    width, height = 64, 48
    steps = 8 if turbo else 3
    guidance = None if distilled else 2.5 if turbo else 4.0
    cfg = family == "qwen3" and not distilled
    prepared = prepare_references(references) if references else []
    encoded = model.encode_references(prepared) if references else None
    embeds = model.encode("a red fox in snow")
    negative = model.encode("") if cfg else None
    timesteps = schedule(steps, (width // 16) * (height // 16), TURBO_SIGMAS if turbo else None)
    overlay = turbo_overlay(graph.components["transformer"]) if turbo else None

    def denoise() -> torch.Tensor:
        return model.denoise(embeds, negative, encoded, width=width, height=height, timesteps=timesteps,
                             guidance=guidance, seed=11, on_step=lambda _: None, cancel=lambda: None)

    tokens = overlay.applied(graph.components["transformer"], denoise) if overlay else denoise()
    image = model.decode(tokens, width=width, height=height)
    assert image.size == (width, height)

    transformer = merged(graph.components["transformer"], overlay) if overlay else graph.components["transformer"]
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, time_shift_type="exponential")
    parts = dict(scheduler=scheduler, vae=graph.components["vae"], text_encoder=encoder, tokenizer=model.tokenizer,
                 transformer=transformer)
    pipe = Flux2Pipeline(**parts) if family == "mistral3" else Flux2KleinPipeline(**parts, is_distilled=distilled)
    noise = torch.randn((1, 8, height // 16, width // 16), generator=torch.Generator().manual_seed(11))
    kwargs = dict(prompt="a red fox in snow", height=height, width=width, num_inference_steps=steps,
                  latents=noise.to(torch.bfloat16), output_type="np", text_encoder_out_layers=TEXT_LAYERS[family])
    if guidance is not None:
        kwargs["guidance_scale"] = guidance
    if turbo:
        kwargs["sigmas"] = list(TURBO_SIGMAS)
    if references:
        kwargs["image"] = prepared
    final: list[torch.Tensor] = []

    def observe(_pipe: object, _step: int, _t: object, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        final[:] = [values["latents"]]
        return values

    vae = graph.components["vae"]
    vae.decode = lambda z, **options: type(vae).decode(vae, z.float(), **options)  # Diffusers feeds it bf16
    try:
        expected = pipe(**kwargs, callback_on_step_end=observe).images[0]
    finally:
        del vae.decode
    drift = float((tokens.float() - final[0].float()).norm() / final[0].float().norm())
    difference = float(np.abs(np.asarray(image).astype(np.float32) / 255 - expected).mean())
    print(f"{family} distilled={distilled} turbo={turbo} references={len(references)}: "
          f"latents rel L2 {drift:.4f}, pixels mean |Δ| {difference:.4f}")
    assert drift < 0.02 and difference < 0.01, (drift, difference)


def sizes() -> None:
    """The ratio at the area on the 16 px grid; sides 64-4096 and 4 MiP hold only past them."""
    assert output_size(1.0, 1 << 20) == (1024, 1024)
    assert output_size(16 / 9, 4 << 20) == (2736, 1536)
    assert output_size(3000 / 2000, 3000 * 2000) == (2512, 1680)  # clamped to 4 MiP at 3:2
    assert output_size(8.0, 1 << 20) == (2896, 368)
    assert output_size(100.0, 1 << 20) == (4096, 256)
    assert output_size(8.0, 4 << 20) == (4096, 1024)
    for step in range(-400, 401):
        w, h = output_size(10 ** (step / 100), 2 << 20)
        assert w % 16 == h % 16 == 0 and 64 <= min(w, h) <= max(w, h) <= 4096 and w * h <= 1088 * 1024 * 4


def edits(root: Path) -> None:
    """`generate` and `generate_turbo` through the real kernel on real PNGs: given no size, the
    first reference's ratio and pixel count; an explicit input overrides only itself."""
    dev, _ = built("mistral3", False)
    klein, _ = built("qwen3", True)
    # `prompt_upsampling=False` never calls the upsampler: an unloaded stand-in.
    rng = np.random.default_rng(5)
    paths = []
    for index, size in enumerate(((400, 240), (320, 320), (3000, 2000))):
        path = root / f"reference-{index}.png"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(path)
    cases = [
        ("generate", dev, [], {}, (1024, 1024)),
        ("generate", dev, [], {"aspect_ratio": "16:9"}, (1360, 768)),
        ("generate_turbo", klein, [paths[0]], {}, (400, 240)),
        ("generate_turbo", klein, [paths[0], paths[1]], {"megapixels": 1}, (1328, 800)),
        ("generate_turbo", klein, [paths[1]], {"aspect_ratio": "2:1"}, (448, 224)),
        ("generate", dev, [paths[2]], {}, (2512, 1680)),
    ]
    for index, (function, model, references, given, expected) in enumerate(cases):
        grants = {f"reference_images.{n}": granted_input(path, input_id=f"reference_images.{n}", order=n)
                  for n, path in enumerate(references)}
        wire = {"prompt": "a fox", "steps": 1, "seed": 3, "unknown_field": 1, **given}
        wire["reference_images"] = [grant.digest for grant in grants.values()]
        models = {"model": model, "upsampler": Flux2Upsampler.for_test()}
        if function == "generate_turbo":
            models["turbo_lora"] = turbo_overlay(model.graph.components["transformer"]) if model.dev else Flux2TurboLoRA.for_test()
        spool = root / f"edit-{index}"
        result = invoke_with_fakes(app.get(function), wire, request_id=f"edit-{index}", spool=spool, models=models,
                                   assets=grants)
        assert (result.result.width, result.result.height) == expected, (function, given, result.result)
        with Image.open(spool / "image-0001") as png:
            assert png.size == expected, png.size


if __name__ == "__main__":
    sizes()
    rng = np.random.default_rng(3)
    pictures = [Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)),
                Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))]
    for family, distilled, turbo in (("mistral3", False, False), ("mistral3", False, True),
                                     ("qwen3", True, False), ("qwen3", False, False)):
        for count in (0, 1, 2):
            proof(family, distilled, turbo, pictures[:count])
    with tempfile.TemporaryDirectory() as scratch:
        edits(Path(scratch))
    print(json.dumps({"flux-2 model proof": "passed"}))
