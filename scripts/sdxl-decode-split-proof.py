#!/usr/bin/env python
"""SDXL's VAE decode through the Runtime's `split`, on the CPU with a tiny VAE of SDXL's shape
(three 2x upsamplers: 8 image pixels a latent pixel). At level 0, which the Runtime runs while
the device holds the whole decode, it is today's decode bit for bit. Levels 1-3 cut the latents
into tiles (each level about halves the largest), cross-faded over `DECODE_OVERLAP`; each prints
its PSNR against the whole decode and the largest tile's operands, the bytes the split declares.

Run with the package's own interpreter.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdxl"))

from cozy_runtime.author import _split  # noqa: E402
from sdxl import DECODE_OVERLAP, DECODE_TILES, _Decoded  # noqa: E402

#: SDXL's VAE layout (4 blocks, 3 upsamplers, 4 latent channels) at a few channels
CONFIG = {
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (8, 16, 16, 16),
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "latent_channels": 4,
}


class Level:
    """The memory's side of `split` at one level, as the Runtime runs it: windows of at most
    1/2^level of the elements, never finer than the finest window."""

    def __init__(self, level: int) -> None:
        self.level, self.windows = level, list[Any]()

    def split(self, name: str, key: str, total: int, finest: int, run: Any) -> Any:
        most = max(-(-total // (1 << self.level)), finest)
        self.windows = _split._windows(self.shape, DECODE_TILES, most)
        return run(most)

    def piece(self) -> None:
        pass


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(4 / mse) if mse else math.inf


def main() -> None:
    torch.manual_seed(0)
    vae = AutoencoderKL(**CONFIG).eval()
    z = torch.randn(1, 4, 96, 128, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        whole = vae.decode(z).sample
        for level in range(4):
            memory = Level(level)
            memory.shape = tuple(z.shape)  # type: ignore[attr-defined]
            _split.install(memory)
            try:
                got = _split.split(_Decoded(vae), z, *DECODE_TILES)
            finally:
                _split.install(None)
            assert got.shape == whole.shape, (level, got.shape)
            if level == 0:
                assert len(memory.windows) == 1 and torch.equal(got, whole), "level 0 is not whole"
            tile = max(math.prod(b - a for a, b in w) for w in memory.windows)
            print(
                f"level {level}: {len(memory.windows)} tiles, the largest {tile} latent px "
                f"({tile / (z.shape[2] * z.shape[3]):.2f} of whole: its input "
                f"{tile * z.shape[1] * z.element_size()} B, its output "
                f"{tile * 64 * whole.shape[1] * whole.element_size()} B), "
                f"PSNR {psnr(got, whole):.1f} dB against whole"
            )
    print(f"sdxl decode split proof: level 0 is the whole decode; tiles overlap {DECODE_OVERLAP}")


if __name__ == "__main__":
    main()
