"""CPU proof of actual latent/pixel boundary code; no real model or GPU decode."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.anima.decoders import (
    AnimaProcessImagesOutputStep,
    AnimaVaeDecoderStep,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState


def method(source, owner, name, scope):
    cls = next(node for node in ast.parse(source).body if getattr(node, "name", None) == owner)
    value = next(node for node in cls.body if getattr(node, "name", None) == name)
    value.decorator_list = []
    exec(compile(ast.Module(body=[value], type_ignores=[]), "<production-method>", "exec"), scope)
    return scope[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    sys.argv = ["normalization-cpu-proof", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    from comfy import latent_formats

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: (root / name).read_text()
        for name in ("sdxl/sdxl/__init__.py", "anima/anima/__init__.py")
    }
    metadata = json.loads(args.metadata.read_text())
    scope = {"Any": Any, "torch": torch, "_UNTILED_DECODE_PIXELS": 2048 * 2048}
    decode = method(sources["sdxl/sdxl/__init__.py"], "SdxlModel", "decode", scope)
    generator = torch.Generator().manual_seed(6001)

    class VaeProbe(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.config = config
            self.seen = None

        @property
        def dtype(self):
            return self.weight.dtype

        def decode(self, latents, **kwargs):
            self.seen = latents.detach().clone()
            image = torch.zeros((1, 3, 1, 1024, 1024), dtype=self.dtype)
            return (
                (image,)
                if kwargs.get("return_dict") is False
                else SimpleNamespace(sample=image[:, :, 0])
            )

    latent = torch.randn((1, 4, 128, 128), generator=generator)
    scale = metadata["sdxl"]["configs"]["vae"]["scaling_factor"]
    assert scale == latent_formats.SDXL.scale_factor
    vae = VaeProbe(SimpleNamespace(force_upcast=False))
    owner = SimpleNamespace(pipe=SimpleNamespace(components={"vae": vae}, vae_scale=scale))
    decode(owner, latent)
    stock = latent_formats.SDXL().process_out(latent).bfloat16()
    assert torch.equal(vae.seen, stock)
    old = latent.bfloat16() / scale
    checks = [
        {
            "stage": "sdxl_inverse_scale",
            "shape": list(latent.shape),
            "exact_vae_input": True,
            "legacy_bf16_max_difference": float((old - stock).abs().max()),
        }
    ]
    tree = ast.parse(sources["anima/anima/__init__.py"])
    cls = next(node for node in tree.body if getattr(node, "name", None) == "_Fp32VaeDecoder")
    scope.update({"AnimaVaeDecoderStep": AnimaVaeDecoderStep})
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "<production-anima-decoder>", "exec"),
        scope,
    )
    config = SimpleNamespace(**metadata["anima"]["configs"]["vae"])
    latent = torch.randn((1, 16, 1, 128, 128), generator=generator)
    stock_format = latent_formats.Wan21()
    assert torch.equal(torch.tensor(config.latents_mean), stock_format.latents_mean.flatten())
    # Canonical decimal configs round once to FP32 just like stock constants.
    assert torch.equal(torch.tensor(config.latents_std), stock_format.latents_std.flatten())
    vae = VaeProbe(config)
    state = PipelineState()
    state.set("latents", latent)
    scope["_Fp32VaeDecoder"]()(SimpleNamespace(vae=vae), state)
    stock = stock_format.process_out(latent).bfloat16()
    assert torch.equal(vae.seen, stock) and state.get("images").dtype == torch.float32
    original = VaeProbe(config)
    old_state = PipelineState()
    old_state.set("latents", latent)
    AnimaVaeDecoderStep()(SimpleNamespace(vae=original), old_state)
    checks.append(
        {
            "stage": "anima_inverse_scale",
            "shape": list(latent.shape),
            "exact_vae_input": True,
            "legacy_bf16_max_difference": float((original.seen - stock).abs().max()),
        }
    )
    image = torch.linspace(-1.1, 1.1, 3 * 1024 * 1024).view(1, 3, 1024, 1024).bfloat16()
    generate = next(
        node
        for node in ast.parse(sources["sdxl/sdxl/__init__.py"]).body
        if getattr(node, "name", None) == "generate"
    )
    assignment = next(
        node
        for node in generate.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "pixels" for target in node.targets)
    )
    scope["image"] = image
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), "<production-pixels>", "exec"),
        scope,
    )
    actual = scope["pixels"].numpy()
    normalized = image.float().add_(1).div_(2).clamp_(0, 1)[0].permute(1, 2, 0)
    stock = np.clip(255.0 * normalized.numpy(), 0, 255).astype(np.uint8)
    assert np.array_equal(actual, stock)
    legacy = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    checks.append(
        {
            "stage": "sdxl_final_pixels",
            "shape": list(actual.shape),
            "exact_stock_rgb": True,
            "legacy_differing_bytes": int(np.count_nonzero(legacy != stock)),
        }
    )
    state = PipelineState()
    state.set("images", image.float())
    state.set("output_type", "pt")
    AnimaProcessImagesOutputStep()(
        SimpleNamespace(image_processor=VaeImageProcessor(vae_scale_factor=8)), state
    )
    normalized = state.get("images")
    assert normalized.dtype == torch.float32
    generate = next(
        node
        for node in ast.parse(sources["anima/anima/__init__.py"]).body
        if getattr(node, "name", None) == "generate"
    )
    assignment = next(
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "pixels" for target in node.targets)
    )
    scope["image"] = normalized[0]
    exec(
        compile(
            ast.Module(body=[assignment], type_ignores=[]), "<production-anima-pixels>", "exec"
        ),
        scope,
    )
    assert np.array_equal(scope["pixels"].permute(1, 2, 0).numpy(), stock)
    checks.append(
        {"stage": "anima_final_pixels", "shape": list(stock.shape), "exact_stock_rgb": True}
    )
    result = {
        "scope": "actual CPU production boundary/formula proof at full authored geometry with synthetic values; no real model/GPU/image-quality claim",
        "checks": checks,
        "source_sha256": {
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()
        },
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": len(checks), "checks": checks, "scope": result["scope"]}))


if __name__ == "__main__":
    main()
