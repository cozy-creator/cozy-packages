#!/usr/bin/env python
"""Import one package's real stack inside its own locked environment.

The compiled extensions are the ABI proof. `torchvision.ops` and `torch` resolve C++
symbols at import, so a lock that pins the wrong torch fails HERE with an undefined-symbol
error — a fact no static check and no lock comparison can reach. Run it with the package's
own interpreter, never the checking venv:

    sdxl/.venv/bin/python scripts/package_import.py sdxl

`--installed` imports the package from the wheel installed in that environment instead of
the source tree (run it under `python -P`, so nothing but site-packages can answer): the
proof that the built wheel carries every module and asset the application needs.
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
    "minimax-h3": ("h3", ("torch", "torchvision", "torchvision.ops", "triton", "transformers",
                  "diffusers", "tensorfs", "av")),
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
    installed = "--installed" in argv
    args = [arg for arg in argv[1:] if arg != "--installed"]
    if len(args) != 1 or args[0] not in PACKAGES:
        print(f"usage: {argv[0]} <{'|'.join(PACKAGES)}> [--installed]", file=sys.stderr)
        return 2
    package = args[0]
    module_name, peers = PACKAGES[package]
    package_root = ROOT / package
    source_root = package_root / "src"
    if not installed:
        sys.path.insert(0, str(source_root if source_root.is_dir() else package_root))
    for peer in peers:
        importlib.import_module(peer)
    if "torch" in peers:
        torch = importlib.import_module("torch")
        print(f"{package}: torch {torch.__version__}, {len(peers)} peers imported")
        if package == "anima" and not installed:
            check_anima_cosmos_padding_mask(torch)
            print("anima: Cosmos padding-mask forward passed on CPU")
    else:
        print(f"{package}: torch-free, {len(peers)} peers imported")
    module = importlib.import_module(module_name)
    if installed:
        location = pathlib.Path(str(module.__file__))
        assert "site-packages" in location.parts, f"{module_name} answered from {location}"
        if package == "minimax-h3":
            builtin = importlib.import_module("cozy_runtime.models.minimax_h3.model")
            runtime_root = pathlib.Path(str(builtin.__file__)).parent
            assert "site-packages" in runtime_root.parts
            for name in ("H3Model", "H3TurboBase", "H3TurboLoRA"):
                assert getattr(module, name) is getattr(builtin, name)
            for asset in ("processor", "tokenizer"):
                assert (runtime_root / asset).is_dir(), f"Runtime carries no {asset}/"
                assert not (location.parent / asset).exists(), f"workflow duplicates {asset}/"
            for task in ("fl2va", "ref2va", "fl2va_turbo", "ref2va_turbo"):
                relative = pathlib.Path("timestep-plans") / f"{task}.json"
                assert (location.parent / relative).read_bytes() == (
                    runtime_root / relative
                ).read_bytes()
        print(f"{package}: {module_name} imported from the installed wheel")
    app = module.app
    print(f"{package}: {module_name}:app registers {sorted(app._registry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
