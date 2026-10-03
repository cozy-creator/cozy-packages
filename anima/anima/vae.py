"""Anima's VAE: Diffusers' Qwen-Image VAE, decoding a lone frame as 2D work.

Diffusers decodes an image as a one-frame video. Every causal 3D convolution pads two zero
frames in front of the frame and convolves all three, and the frame cache clones each
activation for a next frame that never comes. With one frame only the kernel's last temporal
tap meets data, so that tap runs as a 2D convolution: the same sum without its zero terms. A
lone frame keeps no cache. Norms keep Diffusers' float32 arithmetic and nearest upsampling
copies values, both without a float32 copy of the activation.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    QwenImageCausalConv3d,
    QwenImageRMS_norm,
    QwenImageUpsample,
)


class _CausalConv(QwenImageCausalConv3d):
    """A causal convolution whose temporal padding is all in front: a lone uncached frame
    meets only the last temporal tap."""

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        if cache_x is not None or x.shape[2] != 1:
            return super().forward(x, cache_x)  # type: ignore[no-any-return]
        left, _, top, _, _, _ = self._padding
        return F.conv2d(
            x[:, :, 0],
            self.weight[:, :, -1],
            self.bias,
            self.stride[1:],
            (top, left),
            self.dilation[1:],
            self.groups,
        ).unsqueeze(2)


class _Norm(QwenImageRMS_norm):
    """Diffusers' norm: the float32 quotient rounded to the activation dtype, then the affine."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return super().forward(x)  # type: ignore[no-any-return]
        dim = 1 if self.channel_first else -1
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
        out = torch.div(x, norm.clamp_min_(1e-12), out=torch.empty_like(x))
        return out.mul_(self.scale).mul_(self.gamma).add_(self.bias)


class _Upsample(QwenImageUpsample):
    """Nearest upsampling copies values, so it is exact in the activation dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.Upsample.forward(self, x)


class AnimaVAE(AutoencoderKLQwenImage):
    """Diffusers' decode, tiled or not, with no frame cache for a lone frame."""

    _lone_frame = False

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Any:
        self._lone_frame = z.shape[2] == 1
        try:
            return super()._decode(z, return_dict)
        finally:
            self._lone_frame = False

    def clear_cache(self) -> None:
        super().clear_cache()
        if self._lone_frame:  # only a next frame reads the cache
            self._feat_map = None


def anima_vae(config: dict[str, Any]) -> AnimaVAE:
    """The Qwen-Image VAE for `config`, its modules upgraded in place to the subclasses above."""
    vae = AutoencoderKLQwenImage.from_config(config)
    for module in vae.modules():
        if type(module) is QwenImageCausalConv3d:
            if module.stride[0] == module.dilation[0] == 1 and (
                module._padding[4] == module.kernel_size[0] - 1
            ):
                module.__class__ = _CausalConv
        elif type(module) is QwenImageRMS_norm:
            module.__class__ = _Norm
        elif type(module) is QwenImageUpsample and module.mode in ("nearest", "nearest-exact"):
            module.__class__ = _Upsample
    vae.__class__ = AnimaVAE
    return cast(AnimaVAE, vae)
