# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime[media]>=0.7.0,<1", "minimax-h3", "cozy-eval"]
# [tool.uv.sources]
# minimax-h3 = {path = "../../minimax-h3", editable = true}
# ///
"""Observe H3 VAE reconstruction and residency with an ordinary private script.

cozy run ./examples/client-scripts/h3_vae_roundtrip.py --rental-only \
    model.model=paul/minimax-h3@sha256:<reviewed-checkpoint>

The deterministic reference is generated here; no reference upload is necessary.
Set CYCLE=False to keep VAE resident instead of staging DiT between encode/decode.
This reports observations, not a general model-quality gate.
"""

import numpy as np
import torch
from cozy_eval.integrity import output_integrity
from cozy_runtime.author import (
    ImageAsset,
    ImageFrame,
    Outputs,
    ScriptContext,
    Telemetry,
    VideoAsset,
)

from h3 import _rgb8
from h3_diagnostics import VaeModel, roundtrip

WIDTH, HEIGHT = 1344, 768
FRAMES = 22
CYCLE = True


def main(
    ctx: ScriptContext, *, model: VaeModel, out: Outputs, tel: Telemetry
) -> list[ImageAsset | VideoAsset]:
    # Fixed gradients and colored blocks expose grid/reconstruction artifacts.
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.uint32)
    reference = np.stack(
        ((255 * x // (WIDTH - 1)), (255 * y // (HEIGHT - 1)), ((x // 96 + y // 96) % 2) * 255),
        axis=-1,
    ).astype(np.uint8)
    pixels = torch.from_numpy(reference)
    video, hashes = roundtrip(model, pixels, frames=FRAMES, cycle=CYCLE, ctx=ctx, tel=tel)
    if tuple(video.shape) != (1, FRAMES, 3, HEIGHT, WIDTH) or not torch.isfinite(video).all():
        raise ValueError("H3 VAE returned invalid reconstruction geometry or values")
    rgb, _ = _rgb8(torch, video)
    facts = output_integrity(rgb.numpy())
    tel.metric("vae.grid_peak_ratio", float(facts.grid_peak_ratio))
    for name, value in hashes.items():
        tel.log("resident tensor fingerprint", tensor=name, sha256=value)
    ctx.raise_if_cancelled()
    return [
        out.save_video(rgb, fps=24),
        out.save_image(ImageFrame(WIDTH, HEIGHT, reference.tobytes()), format="png"),
        out.save_image(ImageFrame(WIDTH, HEIGHT, rgb[-1].numpy().tobytes()), format="png"),
    ]
