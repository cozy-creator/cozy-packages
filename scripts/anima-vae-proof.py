#!/usr/bin/env python
"""CPU proof that Anima's VAE decodes what Diffusers' Qwen-Image VAE decodes.

The real config with random weights, so it needs no download. A lone frame (untiled and
tiled) and a five-frame video decode within float32 rounding of stock Diffusers; a lone
frame runs no 3D convolution and keeps no frame cache; the norms and nearest upsampling
are bit-exact in every activation dtype.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    QwenImageRMS_norm,
    QwenImageUpsample,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anima"))

from anima.vae import AnimaVAE, _Norm, _Upsample, anima_vae  # noqa: E402

TOLERANCE = 1e-4  # float32 rounding of a different summation order, on a [-1, 1] output
_failures = 0


def check(name: str, ok: bool, detail: str) -> None:
    global _failures
    _failures += not ok
    print(f"{'  ok   ' if ok else '  FAIL '}{name}: {detail}")


def decode(vae: Any, z: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return vae.decode(z).sample


def same(name: str, stock: Any, package: Any, z: torch.Tensor) -> None:
    want, got = decode(stock, z), decode(package, z)
    error = float((want - got).abs().max())
    spread = float(want.std())
    clipped = float((want.abs() >= 1).float().mean())
    check(
        name,
        error <= TOLERANCE and spread > 0.1 and clipped < 0.1,
        f"max |diff| {error:.2e}, output std {spread:.2f}, clipped {clipped:.1%}",
    )


def main() -> None:
    torch.manual_seed(0)
    config = json.loads((ROOT / "anima/configs/vae.json").read_text())
    stock = AutoencoderKLQwenImage.from_config(config).eval()
    package = anima_vae(config).eval()
    package.load_state_dict(stock.state_dict())
    check("class", isinstance(package, AnimaVAE), type(package).__name__)

    lone = torch.randn(1, 16, 1, 20, 28)
    same("lone frame, untiled", stock, package, lone)

    caches: list[object] = []
    hook = package.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: caches.append(kwargs.get("feat_cache")), with_kwargs=True
    )
    with torch.profiler.profile() as profile:
        decode(package, lone)
    hook.remove()
    check("lone frame keeps no cache", caches == [None], f"feat_cache {caches!r}")
    with torch.profiler.profile() as control:
        decode(stock, lone)
    ran = ["aten::conv3d" in {event.name for event in p.events()} for p in (control, profile)]
    check("lone frame runs no 3D convolution", ran == [True, False], f"stock, package: {ran}")

    for vae in (stock, package):
        vae.enable_tiling(96, 96, 64, 64)
    same("lone frame, tiled", stock, package, lone)
    for vae in (stock, package):
        vae.disable_tiling()
    same("five frames", stock, package, torch.randn(1, 16, 2, 12, 16))

    for images, shape in ((False, (1, 96, 1, 24, 40)), (True, (1, 384, 12, 12))):
        norm = QwenImageRMS_norm(shape[1], images=images)
        norm.gamma.data.normal_(1, 0.3)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            want = norm.to(dtype)
            got = copy.deepcopy(want)
            got.__class__ = _Norm
            x = (torch.randn(shape) * 3).to(dtype)
            check(f"norm {dtype} {len(shape)}d", torch.equal(want(x), got(x)), "bit-exact")
    upsample = QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
    nearest = copy.deepcopy(upsample)
    nearest.__class__ = _Upsample
    x = torch.randn(1, 8, 9, 13, dtype=torch.bfloat16)
    check("nearest upsample bfloat16", torch.equal(upsample(x), nearest(x)), "bit-exact")

    if _failures:
        raise SystemExit(f"{_failures} arm(s) failed")


if __name__ == "__main__":
    main()
