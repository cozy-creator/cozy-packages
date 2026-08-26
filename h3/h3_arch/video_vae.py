"""The H3 VIDEO VAE — a 3D causal CNN encoder and a ViT3D decoder, 16x spatial, 4x temporal.

562 destinations, fp16, ported from the loader proto-001 benchmarked the selected candidate
under (ComfyUI v0.33.0, `comfy/ldm/minimax/vae.py`). The endpoint owns this architecture, so
the endpoint is what has to be right about it: `scripts/h3-keys.py` builds this class on
`meta` and diffs every (key, shape, dtype) against the release's own safetensors header.

WHAT WAS STRIPPED, each because it is not architecture:

  * `comfy.ops` / injectable `operations.X` -> plain `torch.nn`. The indirection exists so a
    loader can decide dtype and casting per weight; that decision is the fill plane's.
  * `cast_to` / `cast_to_input` / `intermediate_device()` -> deleted. A cast at use time is
    device management, and there is no device in this file.
  * the preallocated CPU buffer `decode` streamed finished chunks into -> deleted. It exists
    to fit a card, and a concatenation holds the same values.
  * `optimized_attention` -> `F.scaled_dot_product_attention`.
  * `quant_ops.ck.apply_rope_split_half` -> `_apply_rope_split_half`, which is the same
    rotation the upstream reference spelling computes (`ldm/ideogram4/model.py`) written out.
  * `Conv3d(autopad="causal_zero")` -> the equal weight slice, inline in `CausalConv3d`.

WHAT SURVIVED, because it is SEMANTIC and not memory choreography: THE DECODE WINDOW, on all
three axes. The encoder is causal and was trained on 17-frame clips (1 + 4x4), so it produces
5 tokens per clip and encoding a long video in one shot is a different function, not a cheaper
one. The decoder's mirror is stronger still — its 3D RoPE coordinates are NORMALIZED BY THE
GRID EXTENT (`_token_ids` divides by `dim_size`), so a 7-token window and a 200-token sequence
put the same frame at different angles. Decoding in windows is the model, not an optimization.

THE GRID ARTEFACT (#557), and the reason that paragraph is now three axes and not one. The
first version of this file drew the line between "semantic" and "memory choreography" on the
TIME axis only: it kept the 17-frame clip grid and DELETED the reference's 256 px spatial
tiling as a card-fitting trick, on the stated grounds that "a tile seam is an artefact the
reference does not have". Both halves were wrong. The reference blends its tiles over a >=64 px
overlap, so it has no seam; and the extent-normalization argument this file makes for time
applies unchanged to height and width. Decoding 48x84 latent cells in ONE window instead of the
reference's 16x16 puts every token at a normalized coordinate 5x finer than the decoder ever
sees, the attention loses the neighbour relation, and each token reconstructs its own 16x16
pixel block with an independent offset — a lattice at EXACTLY the 16 px cell period, locked to
the frame. Measured on the oracle renders: token-boundary gradient excess 150% of baseline
through this decoder against 10% through the reference's, ON THE SAME `minimax_h3_video_vae_fp16`
FILE (the `C-comfy-curve-bf16` bank render is that control). The window is now built from the
release constants and `scripts/h3-conform.py` arms it against the reference expression with the
one-window decode kept as the red control.

STILL DIVERGENT, stated rather than fixed: `encode` does NOT window spatially, and the
reference's `tiled_encode` does. The encoder's GroupNorm reduces over the spatial extent, so
that is the same class of defect on the same argument. It is not fixed here because the only
caller is the vision seam, which refuses (`vision_seam_unbuilt`), and an unprovable change to
an unreachable path is a claim. It becomes real work the moment that seam is built.

MEASURED ROUND-TRIP FACT, not a bug in this port: `encode` drops the last `vae_token_drop`
tokens of the whole clip sequence, so `decode(encode(x))` is 12 frames shorter than `x`.
Upstream does exactly this; it is reproduced rather than corrected.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import VideoVaeConfig
from .layout import split_windows

#: The release's `pixel_norm_type: "imagenet"`, which is a normalization and not a weight.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: The seed the CONDITIONING posterior is sampled under, independently of the request's own
#: generator, so the same keyframe or reference always encodes to the same anchor. Fixed at
#: 42 in the reference implementation; a released-model fact, not a preference.
CONDITION_ENCODE_SEED = 42

#: The carrier stores fp16 AND fp16 is the destination's compute dtype, so construction names
#: it here rather than inheriting torch's float32 default: the fill plane matches on (key,
#: shape, DTYPE), and a float32 graph refuses all 562 of these.
DTYPE = torch.float16

#: Read off the header (`decoder.register_tokens` is [1, 4, 2048]); the release's own config
#: does not state it.
REGISTER_TOKENS = 4

#: Upstream's GroupNorm constants. The release's config states neither, so these are the
#: benchmarked loader's values and nothing stronger.
GROUP_NORM_GROUPS = 32
GROUP_NORM_EPS = 1e-6


# ------------------------------------------------------------------ 3D causal CNN encoder


class CausalConv3d(nn.Conv3d):
    """Reflect spatial padding, causal (front-only, zeros) temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, dtype=DTYPE
        )
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, x: Tensor) -> Tensor:
        if sum(self.causal_padding) == 0:
            return super().forward(x)
        pad_t, pad_h, pad_w = self.causal_padding
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode=self.spatial_padding_mode)
        if x.shape[2] == 1:
            # A single frame sees nothing but the zero front padding through every temporal
            # tap but the last, so the last tap IS the convolution. Exactly equal, not an
            # approximation — it is upstream's `autopad="causal_zero"`.
            return F.conv3d(
                x, self.weight[:, :, -1:], self.bias, self.stride, self.padding, self.dilation
            )
        return super().forward(F.pad(x, (0, 0, 0, 0, pad_t * 2, 0), mode="constant"))


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    """GroupNorm whose statistics are computed PER FRAME: time is folded into the batch."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            return super().forward(x)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
        x = super().forward(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _group_norm_3d(num_channels: int, isolated: bool) -> nn.Module:
    cls = TemporalIsolatedGroupNorm if isolated else nn.GroupNorm
    return cls(
        num_groups=GROUP_NORM_GROUPS,
        num_channels=num_channels,
        eps=GROUP_NORM_EPS,
        affine=True,
        dtype=DTYPE,
    )


class Downsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_stride: int,
        space_stride: int,
        padding_mode: str,
    ) -> None:
        super().__init__()
        self.space_stride = space_stride
        self.padding_mode = padding_mode
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(time_stride, space_stride, space_stride),
            padding=(1, 0, 0),
            spatial_padding_mode=padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.padding_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, isolated_gn: bool, padding_mode: str
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = _group_norm_3d(in_channels, isolated_gn)
        self.norm2 = _group_norm_3d(out_channels, isolated_gn)
        self.conv1 = CausalConv3d(
            in_channels, out_channels, 3, padding=1, spatial_padding_mode=padding_mode
        )
        self.conv2 = CausalConv3d(
            out_channels, out_channels, 3, padding=1, spatial_padding_mode=padding_mode
        )
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class DownLevel(nn.Module):
    """One resolution level: `down.N.block.M`, plus `down.N.downsample` where it strides."""

    def __init__(self, blocks: list[nn.Module], downsample: nn.Module | None) -> None:
        super().__init__()
        self.block = nn.ModuleList(blocks)
        self.downsample = downsample


class EncoderFcn3d(nn.Module):
    def __init__(self, cfg: VideoVaeConfig) -> None:
        super().__init__()
        block_mid = [cfg.ch * mult for mult in cfg.ch_mult]
        block_in = [block_mid[0], *block_mid[:-1]]

        self.conv_in = CausalConv3d(
            cfg.in_channels, block_in[0], 3, padding=1, spatial_padding_mode=cfg.padding_mode
        )
        self.down = nn.ModuleList()
        for level in range(len(cfg.ch_mult)):
            blocks: list[nn.Module] = [
                ResnetBlock3D(
                    block_in[level] if i == 0 else block_mid[level],
                    block_mid[level],
                    cfg.use_t_isolated_gn,
                    cfg.padding_mode,
                )
                for i in range(cfg.num_res_blocks)
            ]
            downsample: nn.Module | None = None
            if cfg.space_down[level] * cfg.time_down[level] > 1:
                downsample = Downsample3D(
                    block_mid[level],
                    block_mid[level],
                    cfg.time_down[level],
                    cfg.space_down[level],
                    cfg.padding_mode,
                )
            self.down.append(DownLevel(blocks, downsample))

        self.norm_out = _group_norm_3d(block_mid[-1], cfg.use_t_isolated_gn)
        # `double_z`: the encoder emits mean and logvar packed on the channel axis.
        self.conv_out = CausalConv3d(
            block_mid[-1],
            2 * cfg.z_channels,
            3,
            padding=1,
            spatial_padding_mode=cfg.padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if level.downsample is not None:
                h = level.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ------------------------------------------------------------------ ViT3D decoder


def _token_ids(dims: tuple[int, ...], device: object, dtype: object) -> Tensor:
    """Grid coordinates in [-1, 1], one row per patch. The extent normalization is what makes
    the decode window size part of the model."""
    coords = [
        (torch.arange(0.5, size, device=device, dtype=dtype) / size) * 2.0 - 1.0 for size in dims
    ]
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
    return grid.flatten(0, len(dims) - 1).unsqueeze(0)


def _rope_cos_sin(
    dims: tuple[int, ...],
    num_suffix: int,
    rope_dim: int,
    theta: float,
    device: object,
    dtype: object,
) -> tuple[Tensor, Tensor]:
    """The ND rotary table as its two halves, [1, S, 1, pairs], broadcast over heads.

    Not a registered buffer: a non-persistent buffer built on `meta` stays `meta` after the
    fill plane has filled every destination, and this table is a closed form of the sequence
    shape anyway.
    """
    ids = _token_ids(dims, device, dtype)
    suffix = torch.zeros((1, num_suffix, len(dims)), device=device, dtype=ids.dtype)
    ids = torch.cat([ids, suffix], dim=1)
    inv_freq = 1.0 / theta ** torch.arange(
        0, 1, 2 * len(dims) / rope_dim, device=device, dtype=torch.float32
    )
    angles = (2.0 * math.pi) * ids[:, :, :, None].float() * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3)
    return torch.cos(angles).to(dtype).unsqueeze(2), torch.sin(angles).to(dtype).unsqueeze(2)


def _apply_rope_split_half(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Split-half rotation over the leading `2 * pairs` channels; the tail is left alone."""
    rot = cos.shape[-1] * 2
    half = rot // 2
    first, second = x[..., :half], x[..., half:rot]
    return torch.cat([first * cos - second * sin, first * sin + second * cos, x[..., rot:]], dim=-1)


class Attention(nn.Module):
    def __init__(self, heads: int, dim_head: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        inner = heads * dim_head
        self.to_qkv = nn.Linear(inner, inner * 3, bias=True, dtype=DTYPE)
        self.to_out = nn.Linear(inner, inner, bias=True, dtype=DTYPE)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        # PER-HEAD interleave: the packed row is [head][q|k|v], never [q_all|k_all|v_all].
        qkv = self.to_qkv(x).view(batch, seq, self.heads, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # `qk_norm_affine: false` — the two QK norms carry no weight, so they are not modules
        # here and contribute no destination.
        query = _apply_rope_split_half(F.rms_norm(query, (self.dim_head,), eps=self.eps), cos, sin)
        key = _apply_rope_split_half(F.rms_norm(key, (self.dim_head,), eps=self.eps), cos, sin)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return self.to_out(out.transpose(1, 2).reshape(batch, seq, -1).nan_to_num(0.0))


class FeedForward(nn.Module):
    """Gated SiLU: `w1` emits gate and value packed together."""

    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=True, dtype=DTYPE)
        self.w2 = nn.Linear(inner_dim, dim, bias=True, dtype=DTYPE)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ffn_dim: int, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True, dtype=DTYPE)
        self.attn = Attention(heads, dim_head, eps)
        # LayerScale, per channel and per branch.
        self.scale1 = nn.Parameter(torch.empty(dim, dtype=DTYPE))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True, dtype=DTYPE)
        self.ff = FeedForward(dim, ffn_dim)
        self.scale2 = nn.Parameter(torch.empty(dim, dtype=DTYPE))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin) * self.scale1
        return x + self.ff(self.norm2(x)) * self.scale2


class ViT3dDecoder(nn.Module):
    def __init__(self, cfg: VideoVaeConfig) -> None:
        super().__init__()
        dim = cfg.decoder_dim
        self.patch_size = cfg.vae_ratio
        self.patch_size_t = cfg.vae_ratio_t
        self.out_channels = cfg.out_ch
        self.rope_dim = int(cfg.decoder_dim_head * cfg.decoder_rope_dim_ratio)
        self.rope_theta = cfg.decoder_rope_theta

        self.x_embedder = nn.Linear(cfg.z_channels, dim, dtype=DTYPE)
        self.register_tokens = nn.Parameter(torch.empty(1, REGISTER_TOKENS, dim, dtype=DTYPE))
        # Never read at inference. It is in the key set, which makes it a destination, and a
        # PERSISTENT buffer is what makes a destination out of a non-parameter.
        self.register_buffer("mask_token", torch.empty(1, 1, dim, dtype=DTYPE), persistent=True)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    cfg.decoder_heads,
                    cfg.decoder_dim_head,
                    cfg.decoder_ffn_dim,
                    cfg.decoder_norm_eps,
                )
                for _ in range(cfg.decoder_num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(
            dim, eps=cfg.decoder_norm_eps, elementwise_affine=True, dtype=DTYPE
        )
        self.proj_out = nn.Linear(
            dim, cfg.out_ch * cfg.vae_ratio_t * cfg.vae_ratio * cfg.vae_ratio, dtype=DTYPE
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, _, lat_t, lat_h, lat_w = x.shape
        tokens = self.x_embedder(x.flatten(2).transpose(1, 2))
        num_patches = tokens.shape[1]

        # Four register tokens and one zero token, all at coordinate 0.
        num_suffix = 1 + REGISTER_TOKENS
        seq = torch.cat(
            [
                tokens,
                self.register_tokens.expand(batch, -1, -1),
                torch.zeros_like(tokens[:, 0:1, :]),
            ],
            dim=1,
        )
        cos, sin = _rope_cos_sin(
            (lat_t, lat_h, lat_w), num_suffix, self.rope_dim, self.rope_theta, x.device, x.dtype
        )

        for block in self.transformer_blocks:
            seq = block(seq, cos, sin)

        out = self.proj_out(self.norm_out(seq))[:, :num_patches, :]
        out = out.view(
            batch,
            lat_t,
            lat_h,
            lat_w,
            self.out_channels,
            self.patch_size_t,
            self.patch_size,
            self.patch_size,
        )
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return out.reshape(
            batch,
            self.out_channels,
            lat_t * self.patch_size_t,
            lat_h * self.patch_size,
            lat_w * self.patch_size,
        )


def _blend(head: Tensor, tail: Tensor, extent: int, dim: int) -> Tensor:
    """Linear crossfade of `head`'s last `extent` slices along `dim` into `tail`'s first
    `extent`, returning `tail` with its leading overlap replaced. One function for all three
    axes, because the reference blends all three with this ramp."""
    extent = min(head.shape[dim], tail.shape[dim], extent)
    shape = [1] * tail.ndim
    shape[dim] = extent
    weight = torch.arange(extent, device=tail.device, dtype=tail.dtype).view(shape) / extent

    lead = [slice(None)] * tail.ndim
    lead[dim] = slice(0, extent)
    back = [slice(None)] * head.ndim
    back[dim] = slice(-extent, None)
    blended = head[tuple(back)] * (1.0 - weight) + tail[tuple(lead)] * weight

    if extent < tail.shape[dim]:
        rest = [slice(None)] * tail.ndim
        rest[dim] = slice(extent, None)
        return torch.cat([blended, tail[tuple(rest)]], dim=dim)
    return blended




# ------------------------------------------------------------------ the component root


class VideoVae(nn.Module):
    """The whole video VAE: `encoder` -> `quant_conv` and `post_quant_conv` -> `decoder`.

        encoder      116 destinations   3D causal CNN, 6 levels, 12 residual blocks
        decoder      440 destinations   36-block ViT3D, 4 register tokens
        quant/post     4 destinations   1x1x1 moment and latent projections
        latents_*      2 destinations   the normalization statistics, shipped as weights
                     ───
                     562 destinations, fp16

    `encode` and `decode` are pure functions of their arguments and this module's own
    parameters: no device call, no offload, no autocast, no global state.
    """

    def __init__(self, config: VideoVaeConfig) -> None:
        super().__init__()
        self.vae_ratio = config.vae_ratio
        self.vae_ratio_t = config.vae_ratio_t
        self.clip_length = config.vae_clip_length
        self.token_drop = config.vae_token_drop
        self.tile_size = config.vae_tile_size
        self.tile_overlap_min = config.vae_tile_overlap_min
        # The 17-frame clip grid, in tokens and frames. Every one of these is integer
        # arithmetic over the two release constants and none of them is a memory budget.
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)

        self.encoder = EncoderFcn3d(config)
        self.quant_conv = nn.Conv3d(2 * config.z_channels, 2 * config.embed_dim, 1, dtype=DTYPE)
        self.post_quant_conv = nn.Conv3d(config.embed_dim, config.z_channels, 1, dtype=DTYPE)
        self.decoder = ViT3dDecoder(config)

        # Persistent, because they are keys in the checkpoint: a normalization constant that
        # ships as a weight is a weight, and `torch.empty` is what leaves it to the fill.
        self.register_buffer(
            "latents_mean", torch.empty(config.z_channels, dtype=DTYPE), persistent=True
        )
        self.register_buffer(
            "latents_std", torch.empty(config.z_channels, dtype=DTYPE), persistent=True
        )

    # -------------------------------------------------------------- pixel conventions

    def _normalize_pixels(self, x: Tensor) -> Tensor:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        return ((x + 1.0) * 0.5 - mean) / std

    def _finalize_pixels(self, x: Tensor) -> Tensor:
        """Raw decoder output -> float32 pixels in [0, 1]. The asymmetry with `encode`'s
        [-1, 1] input is upstream's convention, kept so a benchmark comparison is legible."""
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=torch.float32)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=torch.float32)
        return (x * std.view(1, 3, 1, 1, 1) + mean.view(1, 3, 1, 1, 1)).clamp(0.0, 1.0)

    def _encode_moments(self, pixels: Tensor) -> Tensor:
        return self.quant_conv(self.encoder(pixels))

    def _decode_window(self, latents: Tensor) -> Tensor:
        """ONE call of the ViT decoder, on the latent extent it is handed. Every token's RoPE
        coordinate is normalized by THIS tensor's `latent_h`/`latent_w`, which is why the
        caller may not simply hand it the whole frame."""
        return self.decoder(self.post_quant_conv(latents))

    def _decode_pixels(self, latents: Tensor) -> Tensor:
        """The spatial window plan, blended. `_decode_window` on the 16x16 cell grid the
        decoder's normalized coordinates were fitted at, overlapped and crossfaded, so the
        result carries neither the one-window lattice (#557) nor a window seam."""
        pixels = latents.shape[-2] * self.vae_ratio, latents.shape[-1] * self.vae_ratio
        y_start, y_len, y_over = split_windows(
            pixels[0], self.tile_size, self.tile_overlap_min, self.vae_ratio
        )
        x_start, x_len, x_over = split_windows(
            pixels[1], self.tile_size, self.tile_overlap_min, self.vae_ratio
        )
        if len(y_start) == 1 and len(x_start) == 1:
            return self._decode_window(latents)

        rows: list[Tensor] = []
        # The tail of each window is taken from the RAW decode, before either blend, so a
        # window is crossfaded against what its neighbour actually produced there.
        below: list[Tensor] = []
        for i, (top, height) in enumerate(zip(y_start, y_len, strict=True)):
            zi, zh = top // self.vae_ratio, height // self.vae_ratio
            row: list[Tensor] = []
            next_below: list[Tensor] = []
            right: Tensor | None = None
            for j, (left, width) in enumerate(zip(x_start, x_len, strict=True)):
                zj, zw = left // self.vae_ratio, width // self.vae_ratio
                window = self._decode_window(latents[..., zi : zi + zh, zj : zj + zw])
                if i < len(y_start) - 1:
                    next_below.append(window[..., -y_over[i] :, :].clone())
                next_right = window[..., -x_over[j] :].clone() if j < len(x_start) - 1 else None
                if i > 0:
                    window = _blend(below[j], window, y_over[i - 1], -2)
                if right is not None:
                    window = _blend(right, window, x_over[j - 1], -1)
                right = next_right
                if i < len(y_start) - 1:
                    window = window[..., : -y_over[i], :]
                if j < len(x_start) - 1:
                    window = window[..., : -x_over[j]]
                row.append(window)
            below = next_below
            rows.append(torch.cat(row, dim=-1))
        return torch.cat(rows, dim=-2)

    # -------------------------------------------------------------- the clip grid

    def _decode_chunks(self, num_tokens: int) -> tuple[int, int]:
        """(tokens to pad, number of decode windows) for a latent sequence of `num_tokens`."""
        pseudo = num_tokens + self.token_drop
        pad = (-pseudo) % self.tokens_chunk_size
        pseudo += pad
        chunks = pseudo // self.tokens_chunk_size - int(self.token_drop > 0)
        if chunks < 1:
            # Too few tokens for one window (T_lat == 2, the 17-frame case): pad one more.
            pad += self.tokens_chunk_size
            chunks += 1
        return pad, chunks

    def _pad_frames(self, padded_tokens: int, pad_tokens: int) -> int:
        """Frames the padded tail contributed, which the frame plan then takes back."""
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t
        before = padded_tokens - pad_tokens
        return sum(
            intra_tail if (before + k) % self.tokens_chunk_size == 0 else self.vae_ratio_t
            for k in range(pad_tokens)
        )

    def _decode_frame_count(self, num_tokens: int) -> int:
        """How many frames `_decode_temporal` will keep. Pure integer arithmetic — upstream
        needed it to size a buffer; here it is what the concatenated result is trimmed to."""
        pad_tokens, num_chunks = self._decode_chunks(num_tokens)
        padded = num_tokens + pad_tokens
        chunk_frames_max = self.tokens_chunk_size * self.vae_ratio_t
        splits = int(self.token_drop > 0) + 1

        total = 0
        final_overlap = 0
        for i in range(num_chunks):
            start = i * self.tokens_chunk_size
            end = start + self.tokens_chunk_size + self.token_overlap
            window = max(0, min(end, padded) - min(start, padded)) * self.vae_ratio_t
            for j in range(splits):
                first = j * chunk_frames_max
                last = min(first + chunk_frames_max, window)
                kept = max(0, last - first - self.frame_pre_padding)
                if j == 0:
                    total += kept
                else:
                    final_overlap = kept
        return total + final_overlap - self._pad_frames(padded, pad_tokens)

    def _decode_temporal(self, latents: Tensor) -> Tensor:
        chunk_frames_max = self.tokens_chunk_size * self.vae_ratio_t
        splits = int(self.token_drop > 0) + 1
        frames = self._decode_frame_count(latents.shape[2])

        pad_tokens, num_chunks = self._decode_chunks(latents.shape[2])
        if pad_tokens > 0:
            tail = latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)
            latents = torch.cat([latents, tail], dim=2)

        parts: list[Tensor] = []
        overlap: Tensor | None = None
        for i in range(num_chunks):
            start = i * self.tokens_chunk_size
            end = start + self.tokens_chunk_size + self.token_overlap
            window = self._decode_pixels(latents[:, :, start:end])
            for j in range(splits):
                first = j * chunk_frames_max
                last = min(first + chunk_frames_max, window.shape[2])
                part = window[:, :, first:last][:, :, self.frame_pre_padding :]
                if j != 0:
                    overlap = part.contiguous()
                    continue
                if overlap is not None:
                    part = _blend(overlap, part, self.frame_overlap, 2)
                    overlap = None
                parts.append(part)
            if i == num_chunks - 1 and overlap is not None:
                parts.append(overlap)
                overlap = None

        pixels = torch.cat([part for part in parts if part.shape[2] > 0], dim=2)
        return self._finalize_pixels(pixels[:, :, :frames])

    # -------------------------------------------------------------- the two operations

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype the FILL PLANE gave these weights. Read from a parameter rather than
        stored, because the endpoint does not choose it — the artifact's encoding and the
        runtime's delivery do, and this component is handed the result."""
        return self.post_quant_conv.weight.dtype

    def encode(self, pixels: Tensor) -> Tensor:
        """[B, 3, T, H, W] in [-1, 1] -> normalized latents [B, 24, t, H/16, W/16], float32.

        Clip by clip, because the encoder is causal: a 17-frame clip is 5 tokens (1 + 4x4) and
        the last partial clip is filled by repeating its final frame.

        ACTIVATIONS FOLLOW THE WEIGHTS, and this line is a defect the numerics oracle found
        (se-001, on a rented 4090). The reference loader tolerates an activation whose dtype
        differs from the weight's, by casting the WEIGHT at use time — which is exactly the
        `cast_to` this port deleted, because moving a weight is the runtime's. So the cast
        has to happen on the side an author owns: the activation. Without it a caller who
        hands fp32 pixels to an fp16-filled component gets
        `Input type (float) and bias type (c10::Half) should be the same`, and the serve
        path DOES hand it fp32 — the DiT's video head is the checkpoint's fp32 island.
        """
        pixels = pixels.to(self.compute_dtype)
        if pixels.shape[2] == 1:
            moments = self._encode_moments(self._normalize_pixels(pixels))[:, :, -1:]
        else:
            clips: list[Tensor] = []
            for i in range(math.ceil(pixels.shape[2] / self.clip_length)):
                clip = pixels[:, :, i * self.clip_length : (i + 1) * self.clip_length]
                if clip.shape[2] < self.clip_length:
                    missing = self.clip_length - clip.shape[2]
                    clip = torch.cat([clip, clip[:, :, -1:].repeat(1, 1, missing, 1, 1)], dim=2)
                clips.append(self._encode_moments(self._normalize_pixels(clip)))
            moments = torch.cat(clips, dim=2)
            if self.token_drop > 0:
                moments = moments[:, :, : -self.token_drop]

        # `double_z`: the first half is the mean, and the mean IS the latent — this VAE is
        # sampled by the DiT's noise schedule, never by its own logvar.
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).float()
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).float()
        return (mean - latents_mean) / latents_std

    def encode_condition(self, pixels: Tensor, *, seed: int = CONDITION_ENCODE_SEED) -> Tensor:
        """[B, 3, T, H, W] in [-1, 1] -> the CONDITIONING latents of a keyframe or reference.

        NOT `encode`, and the difference is load-bearing rather than stylistic. `encode`
        takes the posterior's MEAN, which is right for a target the DiT's own schedule will
        sample. A conditioning anchor is not that: the released model SAMPLES the posterior
        under a generator seeded independently of the request, and then ROUNDS the sample to
        float16 — keeping about 11 bits of every conditioning latent — before normalizing.
        Both steps are part of the recipe that reproduces its conditioning, and taking the
        mean here instead would be a quiet, permanent difference from the reference on every
        keyframe and every reference image.

        The seed is FIXED, so the same reference always encodes to the same anchor, and it
        is drawn on the HOST so two cards agree. Reference:
        `modular_pipelines/minimax_h3/encoders.py::encode_vae_condition`.
        """
        pixels = pixels.to(self.compute_dtype)
        if pixels.shape[2] == 1:
            moments = self._encode_moments(self._normalize_pixels(pixels))[:, :, -1:]
        else:
            clips: list[Tensor] = []
            for i in range(math.ceil(pixels.shape[2] / self.clip_length)):
                clip = pixels[:, :, i * self.clip_length : (i + 1) * self.clip_length]
                if clip.shape[2] < self.clip_length:
                    missing = self.clip_length - clip.shape[2]
                    clip = torch.cat([clip, clip[:, :, -1:].repeat(1, 1, missing, 1, 1)], dim=2)
                clips.append(self._encode_moments(self._normalize_pixels(clip)))
            moments = torch.cat(clips, dim=2)
            if self.token_drop > 0:
                moments = moments[:, :, : -self.token_drop]

        mean, logvar = torch.chunk(moments.float(), 2, dim=1)
        # The same clamp upstream's `DiagonalGaussianDistribution` applies. Without it an
        # unclamped logvar can overflow `exp` on a checkpoint that never trained it.
        std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
        noise = torch.randn(
            mean.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32
        ).to(mean.device)
        sampled = (mean + std * noise).to(torch.float16).float()
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).float()
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).float()
        return (sampled - latents_mean) / latents_std

    def decode(self, latents: Tensor) -> Tensor:
        """Normalized latents [B, 24, t, h, w] -> float32 pixels [B, 3, T, h*16, w*16] in
        [0, 1], decoded on the clip grid the ViT's normalized RoPE coordinates require."""
        latents = latents.to(self.compute_dtype)
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents = latents * latents_std + latents_mean
        if latents.shape[2] == 1:
            return self._finalize_pixels(self._decode_pixels(latents)[:, :, -1:])
        return self._decode_temporal(latents)
