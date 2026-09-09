#!/usr/bin/env python
"""CUDA proof that the video VAE's fp16 decode operands are bit-exact.

`official.py:decode_video_chunks` decodes under `torch.autocast(float16)`, so every decoder
conv/linear weight is rounded f32->f16 per op whatever it is stored as.
`_apply_video_vae_dtype` stores that rounding, which must therefore change nothing at all.
The arms pull `vae_tiles.decode_chunks` under that autocast, which is the SERVED path since
h3a-017 rather than the equivalent `decode()`.

The proof runs the RELEASE decode geometry at full clip length (345 pixel frames -> 102
latent frames, 1344x768, the 4x7 release tile grid) on the real `TileBatchedVideoVAE`
class with a narrowed channel/layer geometry and random weights, so it fits any card. It
also runs the unautocast encode path `encode_vae_condition` uses, which must be untouched.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from cozy_runtime.author import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minimax-h3"))

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # noqa: E402
    video_latent_num_frames,
)

from official import (  # noqa: E402
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    _apply_video_vae_dtype,
    frames_for,
)
from vae_tiles import TILE_BATCH, TileBatchedVideoVAE  # noqa: E402


def _half_decode_config(module: torch.nn.Module) -> Config:
    return Config(
        {},
        tensor_dtypes={
            f"video_vae.{name}": "f16"
            if (
                name == "post_quant_conv.weight"
                or (name.startswith("decoder.") and name.endswith(".weight") and value.dim() >= 2)
            )
            else "f32"
            for name, value in module.state_dict().items()
        },
    )


# Full length, both cells the release has shipped or is shipping: `frames_for` snaps whole
# seconds onto the VAE's own 17n+5 grid and `video_latent_num_frames` maps those to 5n+2,
# so nothing here hard-codes 102 or 107. 1344x768 divides by the 16-pixel spatial ratio.
CELLS = (14, 15)
CANVAS = (768, 1344)


def _census(module: torch.nn.Module) -> dict[str, object]:
    state = module.state_dict()
    return {
        "dtypes": dict(Counter(str(value.dtype) for value in state.values())),
        "bytes": sum(value.numel() * value.element_size() for value in state.values()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this proof is the CUDA autocast path; no CUDA device is present")
    device = torch.device("cuda")
    torch.manual_seed(0)
    baseline = TileBatchedVideoVAE(
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=8,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
    ).eval()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(0, 0.02)
    cast = _apply_video_vae_dtype(copy.deepcopy(baseline), _half_decode_config(baseline))
    uniform = copy.deepcopy(baseline).half().eval()

    ratio = baseline.spatial_compression_ratio
    rows = baseline._split_tiles(
        CANVAS[0], baseline.tile_sample_min_height, baseline.tile_sample_min_overlap_height
    )
    columns = baseline._split_tiles(
        CANVAS[1], baseline.tile_sample_min_width, baseline.tile_sample_min_overlap_width
    )
    assert len(rows[0]) * len(columns[0]) == TILE_BATCH, "not the release tile grid"

    cells: dict[int, dict[str, object]] = {}
    for seconds in CELLS:
        pixel_frames = frames_for(seconds)
        latent_frames = video_latent_num_frames(pixel_frames, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK)
        latents = torch.randn(
            1,
            baseline.config.latent_channels,
            latent_frames,
            CANVAS[0] // ratio,
            CANVAS[1] // ratio,
            generator=torch.Generator().manual_seed(1),
        ).to(device)
        decoded: dict[str, torch.Tensor] = {}
        for name, module in (("baseline", baseline), ("cast", cast), ("uniform", uniform)):
            module.to(device)
            with torch.no_grad():
                # The SERVED path since h3a-017: `decode_video_chunks` pulls
                # `vae_tiles.decode_chunks` under this autocast, one temporal chunk at a
                # time. Testing `decode()` would test an equivalent path, not the shipped
                # one, so the chunks are pulled and concatenated here exactly as it does.
                pieces = []
                chunks = module.decode_chunks(latents)
                while True:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                        piece = next(chunks, None)
                    if piece is None:
                        break
                    pieces.append(piece)
                frames = torch.cat(pieces, dim=2)
                del pieces
                # To HOST before widening. `.float()` on device is a second full-size copy
                # -- 3.98 GiB at the 15 s cell -- and it OOMs an 8 GB card on the
                # INSTRUMENTATION after the decode itself has already succeeded (se-053).
                decoded[name] = frames.cpu().float()
            del frames
            module.to("cpu")
            torch.cuda.empty_cache()
        cells[seconds] = {
            "latent_frames": latent_frames,
            "pixel_frames": pixel_frames,
            "decoded_shape": list(decoded["baseline"].shape),
            "decode_bit_exact": torch.equal(decoded["baseline"], decoded["cast"]),
            "red_uniform_half_bit_exact": torch.equal(decoded["baseline"], decoded["uniform"]),
            "red_uniform_half_max_abs": float(
                (decoded["baseline"] - decoded["uniform"]).abs().max()
            ),
        }
        del decoded
        torch.cuda.empty_cache()
    latents = None

    # `encode_vae_condition` has no autocast, so the encode side must not move at all.
    pixels = torch.randn(1, 3, 5, 128, 128, generator=torch.Generator().manual_seed(2)).to(device)
    encoded: dict[str, torch.Tensor] = {}
    for name, module in (("baseline", baseline), ("cast", cast)):
        module.to(device)
        with torch.no_grad():
            encoded[name] = module.encode(pixels, return_dict=False)[0].mean.float().cpu()
        module.to("cpu")
        torch.cuda.empty_cache()

    # The narrowed geometry proves the PATH; this sweep proves every operand SHAPE the
    # release VAE actually stores, by running each distinct (module, shape) under the same
    # autocast with the weight stored both ways.
    release = TileBatchedVideoVAE()
    shapes = sorted(
        {
            (type(owner).__name__, tuple(parameter.shape))
            for owner_name, owner in release.decoder.named_modules()
            for name, parameter in owner.named_parameters(recurse=False)
            if name == "weight" and parameter.dim() >= 2
        }
        | {("Conv3d", tuple(release.post_quant_conv.weight.shape))}
    )
    del release
    swept = 0
    mismatched: list[str] = []
    for kind, shape in shapes:
        if len(shape) == 2:
            out_features, in_features = shape
            module = torch.nn.Linear(in_features, out_features).to(device)
            sample = torch.randn(4, 37, in_features, device=device) * 3.0
        else:
            out_channels, in_channels, *kernel = shape
            module = torch.nn.Conv3d(in_channels, out_channels, tuple(kernel)).to(device)
            sample = torch.randn(2, in_channels, *(k + 3 for k in kernel), device=device) * 3.0
        with torch.no_grad():
            module.weight.normal_(0, 0.05)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                wide = module(sample)
                module.weight.data = module.weight.data.to(torch.float16)
                narrow = module(sample)
        if wide.dtype is not narrow.dtype or not torch.equal(wide, narrow):
            mismatched.append(f"{kind}{shape}")
        swept += 1
        del module, sample, wide, narrow
    torch.cuda.empty_cache()

    bit_exact = all(bool(cell["decode_bit_exact"]) for cell in cells.values())
    uniform_exact = any(bool(cell["red_uniform_half_bit_exact"]) for cell in cells.values())
    encode_exact = torch.equal(encoded["baseline"], encoded["cast"])
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cells": cells,
                "tile_grid": [len(rows[0]), len(columns[0])],
                "census": {
                    name: _census(m) for name, m in (("baseline", baseline), ("cast", cast))
                },
                "release_operand_shapes_swept": swept,
                "release_operand_shapes_mismatched": mismatched,
                "decode_bit_exact": bit_exact,
                "encode_bit_exact": encode_exact,
                "red_uniform_half_bit_exact": uniform_exact,
            },
            indent=2,
        )
    )
    if not (bit_exact and encode_exact) or uniform_exact or mismatched:
        raise SystemExit("VAE precision proof FAILED")


if __name__ == "__main__":
    main()
