"""Learned spatial upscaling of MiniMax H3 video latents."""

from cozy_runtime.author import App, WeightsOutput

from .model import H3LatentUpscaler
from .normalize import normalize

__all__ = ["H3LatentUpscaler", "app"]
app = App()
app.job(normalize, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)
