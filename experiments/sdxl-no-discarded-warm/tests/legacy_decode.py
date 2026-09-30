"""Frozen R19 decoder body, retained verbatim as the numerical reference."""
from typing import Any
import torch
from sdxl import _vae_decode_mode, _vae_spatial_scale

def _decode_vae(self, latents: Any, vae_tile_size: int) -> Any:
    vae = self.pipe.components["vae"]
    scale = _vae_spatial_scale(vae)
    pixels = int(latents.shape[-2]) * int(latents.shape[-1]) * scale**2
    with _vae_decode_mode(vae, pixels, vae_tile_size):
        original_dtype = next(vae.parameters()).dtype
        upcast = bool(getattr(vae.config, "force_upcast", False))
        try:
            if upcast:
                native_bf16 = latents.is_cuda and torch.cuda.is_bf16_supported(including_emulation=False)
                wide = torch.bfloat16 if native_bf16 else torch.float32
                vae.to(dtype=wide)
                latents = latents.to(dtype=wide)
            with torch.inference_mode():
                return vae.decode(latents / self.pipe.vae_scale).sample
        finally:
            if upcast:
                vae.to(dtype=original_dtype)
