"""Colour correction of a restored frame toward its source frame.

After numz/ComfyUI-SeedVR2_VideoUpscaler `src/utils/color_fix.py` (Apache-2.0, see
upstream/NOTICE). Frames are (N, 3, H, W) in [-1, 1].
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

Correction = Literal["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]

_BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
_RGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = torch.linalg.inv(_RGB_TO_XYZ)
_D65 = torch.tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
_DELTA = 6 / 29
_KAPPA = (29 / 3) ** 3


def _low(image: torch.Tensor) -> torch.Tensor:
    """Five dilated 3x3 blurs (radius 1..16): the colour and light the wavelet keeps."""
    kernel = _BLUR.to(image).expand(image.shape[1], 1, 3, 3)
    for level in range(5):
        radius = min(2**level, max(1, min(image.shape[-2:]) // 8))
        padded = F.pad(image, (radius,) * 4, mode="replicate")
        image = F.conv2d(padded, kernel, groups=image.shape[1], dilation=radius)
    return image


def wavelet(restored: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """The restoration's high frequencies over the source's low frequencies."""
    return (restored - _low(restored) + _low(source)).clamp(-1, 1)


def _to_lab(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.add(1).div(2).clamp(0, 1)
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = torch.einsum("ij,njhw->nihw", _RGB_TO_XYZ.to(rgb), linear) / _D65.to(rgb)
    f = torch.where(xyz > _DELTA**3, xyz.clamp(min=0) ** (1 / 3), (xyz * _KAPPA + 16) / 116)
    return torch.stack(
        [116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], dim=1
    )


def _from_lab(lab: torch.Tensor) -> torch.Tensor:
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], dim=1)
    xyz = torch.where(f > _DELTA, f**3, (116 * f - 16) / _KAPPA) * _D65.to(lab)
    linear = torch.einsum("ij,njhw->nihw", _XYZ_TO_RGB.to(lab), xyz)
    rgb = torch.where(
        linear > 0.0031308, 1.055 * linear.clamp(min=0) ** (1 / 2.4) - 0.055, 12.92 * linear
    )
    return rgb.clamp(0, 1).mul(2).sub(1)


def _match(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Quantile mapping: each value takes the reference value of the same rank."""
    matched = torch.empty_like(values)
    matched.view(-1)[values.flatten().argsort()] = reference.flatten().sort().values
    return matched


def _resample(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """`_match` for unequal counts: the reference quantile nearest each value's rank."""
    ranks = torch.arange(values.numel(), device=values.device)
    quantiles = reference.sort().values[
        ranks * (reference.numel() - 1) // max(values.numel() - 1, 1)
    ]
    matched = torch.empty_like(values)
    matched[values.argsort()] = quantiles
    return matched


def lab(restored: torch.Tensor, source: torch.Tensor, luminance: float = 0.8) -> torch.Tensor:
    """Wavelet, then CIELAB histogram matching per frame: a*/b* fully, L* by 1 - luminance."""
    content, style = _to_lab(wavelet(restored, source)), _to_lab(source)
    for frame in range(content.shape[0]):
        a, b = (_match(content[frame, c], style[frame, c]) for c in (1, 2))
        lightness = content[frame, 0]
        lightness = luminance * lightness + (1 - luminance) * _match(lightness, style[frame, 0])
        content[frame] = torch.stack([lightness, a, b])
    return _from_lab(content)


def adain(restored: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Per-frame, per-channel mean and standard deviation of the source."""
    mean, variance = restored.mean((2, 3), keepdim=True), restored.var((2, 3), keepdim=True)
    target, spread = source.mean((2, 3), keepdim=True), source.var((2, 3), keepdim=True)
    return (restored - mean) / (variance + 1e-5).sqrt() * (spread + 1e-5).sqrt() + target


def _to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.add(1).div(2).clamp(0, 1)
    value, top = rgb.max(1)
    spread = value - rgb.min(1).values
    safe = torch.where(spread > 1e-10, spread, torch.ones_like(spread))
    r, g, b = rgb.unbind(1)
    hue = torch.stack([((g - b) / safe) % 6, (b - r) / safe + 2, (r - g) / safe + 4], 1)
    hue = hue.gather(1, top[:, None])[:, 0].where(spread > 1e-10, torch.zeros_like(spread)) / 6
    saturation = torch.where(
        value > 1e-10, spread / value.clamp(min=1e-10), torch.zeros_like(value)
    )
    return torch.stack([hue, saturation, value], 1)


def _from_hsv(hsv: torch.Tensor) -> torch.Tensor:
    hue, saturation, value = hsv.unbind(1)
    k = torch.stack([(5 + hue * 6) % 6, (3 + hue * 6) % 6, (1 + hue * 6) % 6], 1)
    rgb = value[:, None] - value[:, None] * saturation[:, None] * (
        torch.minimum(k, 4 - k).clamp(0, 1)
    )
    return rgb.clamp(0, 1).mul(2).sub(1)


def hsv(restored: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Saturation matched to the source within each of twelve 30-degree hue bins."""
    content, style = _to_hsv(restored), _to_hsv(source)
    for frame in range(content.shape[0]):
        hue, saturation = content[frame, 0], content[frame, 1]
        reference_hue, reference = style[frame, 0], style[frame, 1]
        matched = saturation.clone()
        for index in range(12):
            low, high = index / 12, (index + 1) / 12
            mine = (hue >= low) & (hue < high) | ((hue >= 11 / 12) if index == 0 else False)
            theirs = (reference_hue >= low) & (reference_hue < high) | (
                (reference_hue >= 11 / 12) if index == 0 else False
            )
            if int(mine.sum()) > 100 and int(theirs.sum()) > 100:
                matched[mine] = _resample(saturation[mine], reference[theirs])
        content[frame, 1] = matched
    return _from_hsv(content)


def _saturation(rgb: torch.Tensor) -> torch.Tensor:
    return _to_hsv(rgb)[:, 1:2]


def wavelet_adaptive(restored: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Wavelet, with hsv blended in where the restoration over-saturates."""
    base, saturated = wavelet(restored, source), hsv(restored, source)
    reference = _saturation(source)
    weight = torch.sigmoid(5 * (_saturation(restored) - reference - 0.15))
    weight = weight * ((_saturation(base) - reference) > 0.075)
    return base * (1 - weight) + saturated * weight


METHODS = {
    "lab": lab,
    "wavelet": wavelet,
    "wavelet_adaptive": wavelet_adaptive,
    "hsv": hsv,
    "adain": adain,
}


def correct(restored: torch.Tensor, source: torch.Tensor, method: Correction) -> torch.Tensor:
    return restored if method == "none" else METHODS[method](restored, source)
