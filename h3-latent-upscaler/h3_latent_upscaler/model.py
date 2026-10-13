"""Runtime-owned LBH-123-AI 3D Conv v1, over normalized H3 video latents."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

import torch
from cozy_runtime.author import Config, Loader, Model, uses_components

from .upstream.architecture import LatentResizer3D

ARCHITECTURE = "minimax-h3-latent-upscaler-3d-conv-v1"
# The network was trained on H3's normalized latents normalized again by these statistics,
# as the ComfyUI node does to the already-normalized latents it receives.
_STATS = json.loads(files("h3_latent_upscaler.upstream").joinpath("config.json").read_text())
MEAN = torch.tensor(_STATS["latent_normalization"]["mean"]).view(1, -1, 1, 1, 1)
STD = torch.tensor(_STATS["latent_normalization"]["std"]).view(1, -1, 1, 1, 1)


class UpscalerGraph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError("the latent upscaler requires its normalized 3D Conv v1 checkpoint")
        self.components = {"model": LatentResizer3D().eval().bfloat16()}  # type: ignore[no-untyped-call]


def build_upscaler(config: Config) -> UpscalerGraph:
    return UpscalerGraph(config)


class H3LatentUpscaler(Model[UpscalerGraph]):
    graph: UpscalerGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpscalerGraph, factory=build_upscaler)

    @uses_components("model")
    def upscale(
        self, video: torch.Tensor, *, height: int, width: int, temporal_chunking: bool = False
    ) -> torch.Tensor:
        """Return CPU float32 normalized Bx24xTxHxW; audio and time are unchanged.

        Target sizes are exact latent dimensions, selected by the H3 caller.
        """
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
            raise ValueError("expected batch-one, 24-channel H3 video latents")
        if min(video.shape[2:]) < 1 or not video.is_floating_point():
            raise ValueError("H3 video latents need nonempty floating-point dimensions")
        in_height, in_width = video.shape[-2:]
        if not (in_height <= height <= 4 * in_height and in_width <= width <= 4 * in_width):
            raise ValueError("learned latent upscaling supports spatial factors from 1 to 4")
        if (height, width) == (in_height, in_width):
            return video.float().cpu()
        model = self.graph.components["model"]
        device = next(model.parameters()).device
        scale = (height / in_height + width / in_width) / 2.0
        with torch.inference_mode():
            result = model(
                ((video.float() - MEAN) / STD).to(device=device, dtype=torch.bfloat16),
                scale=scale,
                target_size=(video.shape[2], height, width),
                enable_chunking=temporal_chunking,
            )
            return cast(torch.Tensor, result.float().cpu() * STD + MEAN)
