"""Private benchmark copy of the LBH H3 learned spatial latent upscaler.

Copyright (c) 2026 LBH-123-AI (MIT original); Director adaptation Apache-2.0.
Extracted without numerical changes from AIMixer/ComfyUI_MiniMaxH3_Director
63834b9f8561aa14a6e5289ac3d20dee78bb7d23/director/h3_latent_upscale.py.
Only the architecture is retained; Runtime owns checkpoint loading and residency.
See benchmark-licenses/ for both upstream licenses.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Per-channel mean/std from LBH-123-AI H3 latent-upscaler training.
LATENTS_MEAN = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053832,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450743,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
]


def _tensor(value: object) -> torch.Tensor:
    # Torch's generic Module.__call__ is untyped; these layers return tensors.
    if not isinstance(value, torch.Tensor):
        raise TypeError("latent upscaler layer returned a non-tensor")
    return value


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


def _zero_module[M: nn.Module](module: M) -> M:
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb3D(nn.Module):
    def __init__(
        self, channels: int, emb_channels: int, dropout: float = 0, out_channels: int | None = None
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return _tensor(self.skip(x) + h)


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        assert self.pwconv.bias is not None
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return _tensor(x + h)


class LatentResizer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _temporal_kernel(self) -> int:
        """Temporal dwconv kernel (fallback 5). Used as chunk overlap."""
        for block in list(self.in_blocks) + list(self.out_blocks):
            if isinstance(block, TemporalConv):
                return int(block.dwconv.weight.shape[2])
        return 5

    def forward(
        self,
        x: torch.Tensor,
        scale: float,
        target_size: tuple[int, int, int],
        enable_chunking: bool = False,
    ) -> torch.Tensor:
        if tuple(target_size) == tuple(x.shape[-3:]):
            return x

        b, c, t = x.shape[0], x.shape[1], x.shape[2]
        chunk = 24
        overlap = self._temporal_kernel()

        if not enable_chunking or t <= chunk:
            return self._forward_seg(x, scale, target_size)

        log.info(
            "H3 latent upscaler temporal chunking: T=%d chunk=%d overlap=%d",
            t,
            chunk,
            overlap,
        )
        size = (int(target_size[0]), int(target_size[1]), int(target_size[2]))
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out_full = torch.zeros(b, c, t, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, t, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < t:
            seg_start = start
            seg_end = min(t, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(t, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(t + 2 * overlap, out_end + overlap)
            seg = x_padded[:, :, lo:hi]
            seg_out = self._forward_seg(seg, scale, (hi - lo, size[-2], size[-1]))
            s0 = (out_start + overlap) - lo
            n_valid = out_end - out_start
            valid_out = seg_out[:, :, s0 : s0 + n_valid]

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(
                    1, blend_len + 1, device=x.device, dtype=x.dtype
                ) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(
                    blend_len, 0, -1, device=x.device, dtype=x.dtype
                ) / (blend_len + 1)
            w = weight.view(1, 1, n_valid, 1, 1)
            out_full[:, :, out_start:out_end] += valid_out * w
            weight_full[:, :, out_start:out_end] += w
            start += chunk

        return out_full / weight_full.clamp(min=1e-8)

    def _forward_seg(
        self, x: torch.Tensor, scale: float, target_size: tuple[int, int, int]
    ) -> torch.Tensor:
        scale_emb = torch.tensor([float(scale) - 1.0], dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for block in self.in_blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        x = self.norm_out(x)
        x = F.silu(x)
        return _tensor(self.conv_out(x))
