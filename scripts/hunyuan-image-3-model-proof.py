"""HunyuanImage 3.0: the package's model code against Tencent's, on CPU, with a small random-weight model.

    python scripts/hunyuan-image-3-model-proof.py <HunyuanImage-3.0 checkout> <checkpoint files dir>

The checkout is `Tencent-Hunyuan/HunyuanImage-3.0` at 6e9113a6; the files dir holds the Instruct
checkpoint's `config.json`, `generation_config.json`, `tokenizer.json` and `tokenizer_config.json`.
Run it in Tencent's environment (torch, transformers 4.57.1, diffusers 0.35.2, einops). Both models are
built from one shrunken config and share weights; each case compares Tencent's `generate_image` with
the package's `generate_cot` / `denoise` / `decode_latents`: the chain of thought must be identical
(same seeds) and the images equal within float32 noise. It also checks that the package's full-size
module tree names exactly the checkpoint's tensors.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

UPSTREAM, FILES = Path(sys.argv[1]), Path(sys.argv[2])
PORT = Path(__file__).resolve().parent.parent / "hunyuan-image-3" / "hunyuan_image_3" / "tencent"


def load_package(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Tencent's eager MoE pins the CUDA device and marks NVTX ranges; its VAE allocates an empty CUDA tensor.
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
_empty = torch.empty
torch.empty = lambda *a, **k: _empty(*a, **{**k, "device": "cpu"} if k.get("device") == "cuda" else k)

sys.path.insert(0, str(UPSTREAM))
load_package("tencent_upstream", UPSTREAM / "hunyuan_image_3")
from tencent_upstream.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM as Reference  # noqa: E402
from tencent_upstream.configuration_hunyuan_image_3 import HunyuanImage3Config  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

load_package("hunyuan_port", PORT)
from hunyuan_port.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM as Port  # noqa: E402
from hunyuan_port.configuration_hunyuan_image_3 import HunyuanImage3Config as PortConfig  # noqa: E402
from hunyuan_port.system_prompt import get_system_prompt  # noqa: E402
from hunyuan_port.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast as PortTokenizer  # noqa: E402

FULL = json.loads((FILES / "config.json").read_text())
GENERATION = json.loads((FILES / "generation_config.json").read_text())
TINY = {
    "hidden_size": 64, "num_attention_heads": 4, "attention_head_dim": 16, "head_dim": 16,
    "num_key_value_heads": 2, "num_hidden_layers": 2, "intermediate_size": 48,
    "moe_intermediate_size": [32, 32], "moe_topk": [2, 2], "num_experts": 4, "num_shared_expert": [1, 1],
    "image_base_size": 256, "patch_embed_hidden_dim": 32,
    "vae": {**FULL["vae"], "block_out_channels": [32] * 5, "latent_channels": 16, "layers_per_block": 1},
    "vit": {**FULL["vit"], "hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 2,
            "num_hidden_layers": 1},
    "vit_aligner": {**FULL["vit_aligner"], "input_dim": 32, "n_embed": 64},
}


def tiny(distilled: bool) -> dict:
    return {**FULL, **TINY, "cfg_distilled": distilled, "use_meanflow": distilled}


def build(distilled: bool) -> tuple[Reference, Port]:
    torch.manual_seed(0)
    config = HunyuanImage3Config(**tiny(distilled))
    config._attn_implementation = "sdpa"
    reference = Reference(config).eval()
    reference.load_tokenizer(str(FILES))
    reference.generation_config = GenerationConfig(**GENERATION)
    port = Port(PortConfig(**tiny(distilled))).eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    tokenizer_config = json.loads((FILES / "tokenizer_config.json").read_text())
    tokenizer_config.pop("tokenizer_class")
    port._tokenizer = PortTokenizer(
        tokenizer_object=Tokenizer.from_str((FILES / "tokenizer.json").read_text()), **tokenizer_config)
    port.generation_config = SimpleNamespace(**GENERATION)
    return reference, port


def full_tree_keys() -> None:
    """The package's full-size construction names exactly the checkpoint's tensors."""
    index = json.loads((FILES / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.device("meta"):
        port = Port(PortConfig(**FULL))
    built = set(port.state_dict())
    missing, extra = sorted(set(index) - built), sorted(built - set(index))
    assert not missing and not extra, (missing[:5], extra[:5])
    print(f"full tree: {len(built)} tensors, identical to the checkpoint index")


def picture(seed: int, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def case(name: str, reference: Reference, port: Port, *, images: list[Image.Image], bot_task: str,
         image_size: str, steps: int, seed: int) -> None:
    started = time.monotonic()
    reference.generation_config.diff_infer_steps = steps
    port.generation_config.max_new_tokens = 48
    random.seed(seed)
    torch.manual_seed(seed + 1)
    # Tencent's postprocess resizes an edit's output to its reference's exact ratio; the package keeps
    # the model's own size (the lead's ruling), so the comparison is before that resize.
    reference.image_processor.postprocess_outputs = lambda outputs, **_: outputs
    cot_ref, (image_ref,) = reference.generate_image(
        prompt="A red fox in fresh snow", image=images or None, seed=seed, image_size=image_size,
        use_system_prompt="en_unified", bot_task=bot_task, infer_align_image_size=True, max_new_tokens=48,
    )
    random.seed(seed)
    text_seed = random.randint(0, 10_000_000) if bot_task != "image" else None
    system_prompt = get_system_prompt("en_unified", bot_task).strip()
    cond = [port.image_processor.build_cond_images(image_list=images, infer_align_image_size=True)] if images else None
    cot = None
    if bot_task != "image":
        generator = torch.Generator().manual_seed(seed + 1)
        cot = port.generate_cot("A red fox in fresh snow", cond, system_prompt, bot_task,
                                seed=text_seed, generator=generator)
        assert [cot] == cot_ref, (cot[:200], cot_ref[0][:200] if cot_ref else None)
    latents = port.denoise("A red fox in fresh snow", cond, system_prompt, cot, image_size=image_size,
                           steps=steps, guidance_scale=GENERATION["diff_guidance_scale"], seed=seed)
    image = port.decode_latents(latents)
    assert image.size == image_ref.size, (image.size, image_ref.size)
    diff = np.abs(np.asarray(image, dtype=np.int16) - np.asarray(image_ref, dtype=np.int16)).max()
    assert diff <= 1, f"{name}: max pixel difference {diff}"
    print(f"{name}: cot {len(cot or '')} chars identical, image {image.size} max |Δ| {diff} "
          f"[{time.monotonic() - started:.0f}s]")


def main() -> None:
    torch.set_num_threads(1)
    if (FILES / "model.safetensors.index.json").exists():
        full_tree_keys()
    for distilled in (False, True):
        reference, port = build(distilled)
        label = "distil" if distilled else "instruct"
        case(f"{label} t2i", reference, port, images=[], bot_task="image", image_size="256x256", steps=3, seed=7)
        case(f"{label} t2i 2:1", reference, port, images=[], bot_task="image", image_size="352x176", steps=2, seed=8)
        case(f"{label} edit", reference, port, images=[picture(1, (300, 200))], bot_task="image",
             image_size="304x208", steps=2, seed=9)
        case(f"{label} fusion", reference, port, images=[picture(2, (256, 256)), picture(3, (200, 320))],
             bot_task="image", image_size="256x256", steps=2, seed=10)
        case(f"{label} think t2i", reference, port, images=[], bot_task="think_recaption",
             image_size="256x256", steps=2, seed=11)
        case(f"{label} think edit", reference, port, images=[picture(4, (256, 256))], bot_task="think_recaption",
             image_size="256x256", steps=2, seed=12)
    print("model proof passed")


if __name__ == "__main__":
    main()
