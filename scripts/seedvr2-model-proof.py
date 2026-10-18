#!/usr/bin/env python
"""Real tiny SeedVR2 modules on the CPU, through the Runtime's attention selection.

1. Every DiT block's window attention is a Runtime attention site in bfloat16, and with the kernel
   the Runtime installs, each batched run of equal-length windows gives exactly what attention
   over each window alone gives.
2. The VAE takes frames as host bytes and scales them where it runs, exactly as the float path.

Run with the package's own interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from cozy_runtime.internal import attention
from torch.nn.functional import scaled_dot_product_attention as sdpa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "seedvr2"))

from seedvr2.model import DIT, VAE  # noqa: E402
from seedvr2.upstream.models.dit.attention import Runs, WindowAttention  # noqa: E402
from seedvr2.upstream.models.dit.nadit import NaDiT  # noqa: E402
from seedvr2.upstream.models.video_vae_v3.modules.attn_video_vae import (  # noqa: E402
    VideoAutoencoderKLWrapper,
)

LAYERS = 4


class PerWindow:
    """The reference: plain attention over one window at a time."""

    def __call__(self, attn: Any, q: Any, k: Any, v: Any, runs: Runs) -> Any:
        out = []
        for start, count, length in runs:
            for first in range(start, start + count * length, length):
                window = (x[first : first + length].transpose(0, 1)[None] for x in (q, k, v))
                out.append(sdpa(*window)[0].transpose(0, 1))
        return torch.cat(out)


def windows() -> None:
    torch.manual_seed(0)
    dit = NaDiT(
        **{
            **DIT,
            "vid_dim": 64,
            "txt_in_dim": 40,
            "txt_dim": 64,
            "emb_dim": 384,
            "heads": 2,
            "head_dim": 32,
            "num_layers": LAYERS,
            "block_type": DIT["block_type"][:LAYERS],
            "window": DIT["window"][:LAYERS],
            "window_method": DIT["window_method"][:LAYERS],
        }
    ).eval()
    roots = {"dit": dit}
    sites = [module for _, module, _ in attention.sites(roots)]
    assert len(sites) == LAYERS and all(isinstance(m, WindowAttention) for m in sites)
    assert {(attention._dtype(m), attention._head_dim(m)) for m in sites} == {("bfloat16", 32)}
    sdpa = attention.BY_NAME["sdpa"]
    applied = attention.install([attention.Ready(sdpa, attention._member(sdpa))], roots)
    assert applied.totals() == {"sdpa": LAYERS}, applied.totals()
    t, h, w = 5, 34, 60  # 1080p's window layout at a quarter of its rows and columns
    args = {
        "vid": torch.randn(t * h * w, 33),
        "txt": torch.randn(58, 40),
        "vid_shape": torch.tensor([[t, h, w]]),
        "txt_shape": torch.tensor([[58]]),
        "timestep": torch.tensor([1000.0]),
    }
    with torch.inference_mode():
        batched = dit(**args).vid_sample
        for module in sites:
            module.set_processor(PerWindow())
        assert torch.equal(batched, dit(**args).vid_sample)


def host_bytes() -> None:
    torch.manual_seed(1)
    vae = VideoAutoencoderKLWrapper(**{**VAE, "block_out_channels": (32, 32, 64, 64)}).eval()
    vae.set_causal_slicing(split_size=4, memory_device="same")
    vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
    frames = torch.randint(0, 256, (1, 3, 9, 64, 96), dtype=torch.uint8)
    with torch.inference_mode():
        scaled = vae.encode(frames).posterior.mode()
        floats = vae.encode(frames.float().div(127.5).sub(1)).posterior.mode()
    assert scaled.shape == (1, 16, 3, 8, 12) and torch.equal(scaled, floats)


if __name__ == "__main__":
    windows()
    host_bytes()
    print("seedvr2 model proof: window attention sites and host-byte VAE input agree")
