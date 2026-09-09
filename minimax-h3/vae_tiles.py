"""The official H3 video VAE with one temporal chunk's spatial tiles decoded as ONE batch.

Diffusers' `_decode_clip` lays a grid of equal 256 px tiles (64 px minimum overlap) over
the chunk and runs the ViT decoder once per tile: at 1344x768 that is a 4x7 grid, 28
batch-1 forwards per chunk and ~590 per 362-frame clip, each too small to fill the card
(19.4 s on an H100). Every tile is the same size — `_split_tiles` pushes the slack into
the overlaps, never into a shorter edge tile — and the decoder is batch-independent: a
ViT over `(B, S, C)` tokens whose norms reduce over the last dimension only, whose
register/cls tokens are per-batch replicas and whose RoPE comes from the tile geometry.
(`MiniMaxH3VideoGroupNorm`, the one batch-mixing module, is encoder-side.) So the tiles
of one chunk go through `post_quant_conv` and the decoder as one `(tiles, C, T, h, w)`
batch and are stitched by the unchanged `_stitch_tiles` in the unchanged order.

The arithmetic per tile is the same; the BYTES are a property of the kernel a larger GEMM
selects, so they are measured, never assumed. Measured 2026-09-07 at the release tile
geometry: the batched decode differs from the sequential one by at most ONE ULP of the
compute dtype at the output's peak magnitude — fp32 on CPU (1.5e-8, 157 dB) and fp16
autocast on an RTX 4070 (1.2e-4, 85-91 dB at every batch from 2 to 28); the H100 verdict
is banked on the tracker issue (h3a-017). `scripts/h3-conform.py` (`vae-tiles` arm) holds
that bound on CPU with a red arm. Tile geometry, autocast and precision are untouched —
h3a-006 owns the lossy decode variants.
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLMiniMaxH3

#: Tiles per decoder forward. The release geometry (1344x768 at 256 px tiles, 64 px
#: overlap) is a 4x7 grid, so one chunk is exactly one forward; a larger frame decodes in
#: groups of this many, which bounds the batch's activation footprint. The batch size is
#: part of the numerical identity of the decode (a different GEMM may reduce differently),
#: so it is a declared plan fact and changes only with a banked before/after digest.
TILE_BATCH = 28


class TileBatchedVideoVAE(AutoencoderKLMiniMaxH3):  # type: ignore[misc]
    """`AutoencoderKLMiniMaxH3` whose tiled clip decode batches the tiles."""

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        ratio = self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            z.shape[-2] * ratio, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            z.shape[-1] * ratio, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        tiles = [
            z[..., y // ratio : (y + y_len) // ratio, x // ratio : (x + x_len) // ratio]
            for y, y_len in zip(y_indices, y_lengths, strict=True)
            for x, x_len in zip(x_indices, x_lengths, strict=True)
        ]
        decoded: list[torch.Tensor] = []
        for start in range(0, len(tiles), TILE_BATCH):
            group = tiles[start : start + TILE_BATCH]
            batch = group[0] if len(group) == 1 else torch.cat(group, dim=0)
            decoded.extend(self.decoder(self.post_quant_conv(batch)).split(z.shape[0], dim=0))
        columns = len(x_indices)
        rows = [decoded[row * columns : (row + 1) * columns] for row in range(len(y_indices))]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)
