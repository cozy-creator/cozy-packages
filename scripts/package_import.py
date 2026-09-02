#!/usr/bin/env python
"""Import one package's real stack inside its own locked environment.

The compiled extensions are the ABI proof. `torchvision.ops` and `torch` resolve C++
symbols at import, so a lock that pins the wrong torch fails HERE with an undefined-symbol
error — a fact no static check and no lock comparison can reach. Run it with the package's
own interpreter, never the checking venv:

    sdxl/.venv/bin/python scripts/package_import.py sdxl
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: package directory -> (module to import, compiled peers that must load beside it)
PACKAGES = {
    "anima": (
        "anima",
        (
            "torch",
            "torchvision",
            "torchvision.transforms",
            "triton",
            "transformers",
            "diffusers",
            "av",
        ),
    ),
    "sdxl": ("sdxl", ("torch", "triton", "transformers", "diffusers", "av")),
    "quality-judge": ("quality_judge", ("torch", "triton", "transformers", "tokenizers",
                                        "numpy", "PIL.Image", "tensorfs", "av")),
    "minimax-h3": ("h3", ("torch", "torchvision", "torchvision.ops", "triton", "transformers",
                  "diffusers", "tensorfs", "av")),
    "h3-quality-gate": ("h3_quality_gate", ("cozy_eval", "numpy", "msgspec")),
    "minimax-h3-tools": ("h3_tables.job", ("torch", "triton", "numpy", "msgspec")),
}


def check_anima_cosmos_padding_mask(torch: Any) -> None:
    """Exercise the exact Diffusers branch used by Anima's first denoise step on CPU."""
    cosmos = importlib.import_module("diffusers.models.transformers.transformer_cosmos")
    model = cosmos.CosmosTransformer3DModel(
        in_channels=1,
        out_channels=1,
        num_attention_heads=1,
        attention_head_dim=12,
        num_layers=0,
        mlp_ratio=1.0,
        text_embed_dim=12,
        adaln_lora_dim=12,
        max_size=(1, 2, 2),
        patch_size=(1, 1, 1),
        rope_scale=(1.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    )
    result = model(
        hidden_states=torch.zeros((1, 1, 1, 2, 2)),
        timestep=torch.zeros((1,)),
        encoder_hidden_states=torch.zeros((1, 1, 12)),
        padding_mask=torch.zeros((1, 1, 1, 1)),
    )
    assert result.sample.shape == (1, 1, 1, 2, 2)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in PACKAGES:
        print(f"usage: {argv[0]} <{'|'.join(PACKAGES)}>", file=sys.stderr)
        return 2
    package = argv[1]
    module_name, peers = PACKAGES[package]
    package_root = ROOT / package
    source_root = package_root / "src"
    sys.path.insert(0, str(source_root if source_root.is_dir() else package_root))
    for peer in peers:
        importlib.import_module(peer)
    if "torch" in peers:
        torch = importlib.import_module("torch")
        print(f"{package}: torch {torch.__version__}, {len(peers)} peers imported")
        if package == "anima":
            check_anima_cosmos_padding_mask(torch)
            print("anima: Cosmos padding-mask forward passed on CPU")
    else:
        print(f"{package}: torch-free, {len(peers)} peers imported")
    module = importlib.import_module(module_name)
    app = module.app
    print(f"{package}: {module_name}:app registers {sorted(app._registry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
