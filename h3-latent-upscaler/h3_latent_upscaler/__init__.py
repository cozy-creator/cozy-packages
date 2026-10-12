"""Learned spatial upscaling of MiniMax H3 video latents."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    InvalidRequest,
    Outputs,
    Telemetry,
    WeightsOutput,
)
from cozy_runtime.models.minimax_h3.refinement import decode_latents, encode_latents

from .model import H3LatentUpscaler
from .normalize import normalize

__all__ = ["H3LatentUpscaler", "app"]
app = App()
app.job(normalize, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)

LatentAsset = Annotated[
    FileAsset,
    AssetBound(max_bytes=2 << 30, media_types=("application/octet-stream",)),
]


class UpscaleInput(msgspec.Struct):
    latents: LatentAsset
    height: Annotated[int, msgspec.Meta(ge=32, le=8192, multiple_of=32)]
    width: Annotated[int, msgspec.Meta(ge=32, le=8192, multiple_of=32)]
    temporal_chunking: bool = False


class UpscaleOutput(msgspec.Struct):
    latents: LatentAsset


@app.entrypoint(
    defaults={"model": [{"gpu": "*", "gpus": 1, "lane": "h3-latent-upscaler@0.1.0/bf16"}]},
    memoize=True,
)
def upscale(
    ctx: Context, payload: UpscaleInput, model: H3LatentUpscaler, out: Outputs, tel: Telemetry
) -> UpscaleOutput:
    """Upscale retained H3 video latents; preserve clean audio and refinement context."""
    ctx.raise_if_cancelled()
    if payload.height * payload.width > 16_777_216:
        raise InvalidRequest(
            "target exceeds the H3 latent carrier canvas limit", code="h3_refinement"
        )
    with tel.stage("reading H3 latents", overall_range=(0.0, 0.1)):
        value = decode_latents(payload.latents.read_bytes())
    height, width = payload.height // 16, payload.width // 16
    with tel.stage("upscaling H3 latents", overall_range=(0.1, 0.9)):
        video = model.upscale(
            value.video, height=height, width=width, temporal_chunking=payload.temporal_chunking
        )
        conditions = value.conditions
        if value.task == "fl2va":
            conditions = tuple(
                model.upscale(
                    condition,
                    height=height,
                    width=width,
                    temporal_chunking=payload.temporal_chunking,
                )
                for condition in conditions
            )
    ctx.raise_if_cancelled()
    with tel.stage("saving H3 latents", overall_range=(0.9, 1.0)):
        result = replace(value, video=video, conditions=conditions)
        return UpscaleOutput(
            out.save_bytes(encode_latents(result), media_type="application/octet-stream")
        )
