"""The H3 audio VAE — 32 kHz stereo, 917 destinations, F32 throughout.

A DAC-lineage waveform encoder (Snake activations, five strided residual stages), one
attention `pre_block` that pools the 2048-wide trunk down to the 32-channel latent, and a
BigVGAN decoder that runs the seven upsample stages back out to samples. The encoder's
strides multiply to 800, so the latent rate is 40 Hz and a latent frame is 800 audio
samples; the two stereo channels are folded into the batch and processed independently.

Weight-norm parametrizations are already folded into plain conv weights in the released
checkpoint, so every module here is an ordinary `torch.nn` one and every key is a plain
`*.weight` / `*.bias`.

WHAT THIS FILE DOES NOT DO: no device call, no dtype cast, no chunked decode. `encode` and
`decode` are pure functions of their arguments and this module's own parameters; where the
artifact's weights live and what precision they are filled at is the runtime's decision,
made before either method is entered.

Lineage of the reference implementation: descript-audio-codec (MIT) for the encoder, NVIDIA
BigVGAN (MIT, from hifi-gan) for the decoder, alias-free-torch (Apache-2.0) for the
anti-aliased activations.
"""

#  The checking venv holds no torch, so `nn.Module` is `Any` and strict mode reports every
#  class here. Same true-about-CI/false-about-the-code pair pyproject.toml already loosens
#  for the two model modules; nothing else is relaxed.
# mypy: disallow-subclassing-any=false

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import AudioVaeConfig

# ------------------------------------------------------------------ Snake activations


def snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """x + 1/beta * sin^2(alpha * x)."""
    t = torch.sin(alpha * x)
    return t.mul_(t).mul_((beta + 1e-9).reciprocal()).add_(x)


class Snake1d(nn.Module):
    """Encoder-side Snake: one per-channel `alpha` stored [1, C, 1], used as beta too."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha, self.alpha)


class SnakeBeta(nn.Module):
    """Decoder-side Snake with an independent beta. Both are stored in log scale
    (`snake_logscale`), so they are exponentiated before use."""

    def __init__(self, in_features: int, logscale: bool = True) -> None:
        super().__init__()
        self.logscale = logscale
        self.alpha = nn.Parameter(torch.empty(in_features))
        self.beta = nn.Parameter(torch.empty(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        if self.logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake(x, alpha, beta)


# ------------------------------------------------------ alias-free (anti-aliased) resampling

#: The anti-alias FIR taps. Upstream BigVGAN COMPUTES these as Kaiser-windowed sincs at
#: construction; this release SHIPS them, [1, 1, 12] F32, as 254 of the 917 destinations. So
#: they are persistent buffers with no initializer — recomputing them here would put author
#: constants where the artifact's own bytes belong, and the two are not guaranteed equal.
_FILTER_KERNEL = 12


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = _FILTER_KERNEL) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", torch.empty(1, 1, kernel_size), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        ).mul_(self.ratio)
        return x[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, stride: int = 1, kernel_size: int = _FILTER_KERNEL) -> None:
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", torch.empty(1, 1, kernel_size), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = _FILTER_KERNEL) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.lowpass = LowPassFilter1d(stride=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Upsample x2, apply the pointwise activation, downsample x2 — the alias-free wrapper
    that keeps Snake's harmonics below Nyquist."""

    def __init__(self, activation: nn.Module, up_ratio: int = 2, down_ratio: int = 2) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio)
        self.downsample = DownSample1d(down_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


# ------------------------------------------------------------------ DAC encoder


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y.add_(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """One flat `nn.Sequential` — the checkpoint's keys are its bare integer indices, so the
    nesting is load-bearing and not a style choice."""

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int) -> None:
        super().__init__()
        stages: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            stages.append(EncoderBlock(d_model, stride=stride))
        stages.append(Snake1d(d_model))
        stages.append(nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1))
        self.block = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------- attention posterior head (pre_block)


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)).mul_(self.w1(x)))


class CausalAttention(nn.Module):
    """A fused biasless qkv plus THREE separate bias tensors. `zero_k_bias` is a constant
    zero that still ships as a destination, so it is a persistent buffer, not a constant
    this file invents — the artifact carries it and the census must demand it."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim), persistent=True)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=bias)
        q, k, v = (
            qkv.reshape(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Mean over heads, then average-pool head_dim (256) down to the latent width (32).
        pooled = F.adaptive_avg_pool1d(torch.mean(attended, dim=1), self.out_dim)
        return self.proj(pooled)


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return x.add_(self.mlp(self.norm2(x)))


# ------------------------------------------------------------------ BigVGAN decoder


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def upsample_kernel(rate: int) -> int:
    """BigVGAN's kernel-per-rate rule. Checked against the pinned header: decoder rates
    (5, 5, 2, 2, 2, 2, 2) give transpose kernels (9, 9, 4, 4, 4, 4, 4)."""
    return 2 * rate if rate % 2 == 0 else 2 * rate - 1


class AMPBlock1(nn.Module):
    """Three dilated conv pairs, each pair sandwiched between two anti-aliased Snakes:
    3 + 3 convs and 6 activations per block, 21 blocks in this decoder."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, ...],
        logscale: bool = True,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
                for _ in dilation
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [
                Activation1d(SnakeBeta(channels, logscale=logscale))
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            xt = c2(a2(c1(a1(x))))
            x = xt.add_(x)
        return x


class BigVGAN(nn.Module):
    """The H3 32 kHz vocoder: seven upsample stages, three resblock kernels each.

    `use_bias_at_final=False` and `use_tanh_at_final=False` — `conv_post` carries no bias
    and the waveform is clamped to [-1, 1] instead of squashed.
    """

    def __init__(self, config: AudioVaeConfig) -> None:
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.decoder_rates)
        initial = config.decoder_dim

        self.conv_pre = nn.Conv1d(config.latent_dim, initial, 7, 1, padding=3)

        self.ups = nn.ModuleList(
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        initial // (2**i),
                        initial // (2 ** (i + 1)),
                        upsample_kernel(rate),
                        rate,
                        padding=(upsample_kernel(rate) - rate) // 2,
                    )
                ]
            )
            for i, rate in enumerate(config.decoder_rates)
        )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = initial // (2 ** (i + 1))
            for kernel, dilation in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    AMPBlock1(channels, kernel, dilation, logscale=config.snake_logscale)
                )

        final = initial // (2**self.num_upsamples)
        self.activation_post = Activation1d(SnakeBeta(final, logscale=config.snake_logscale))
        self.conv_post = nn.Conv1d(final, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            # The three kernel branches are AVERAGED, not summed — the /num_kernels is part
            # of the trained scale. Accumulated in index order, which is a float fact.
            base = i * self.num_kernels
            branch = self.resblocks[base](x)
            for j in range(1, self.num_kernels):
                branch = branch + self.resblocks[base + j](x)
            x = branch.div_(self.num_kernels)
        return self.conv_post(self.activation_post(x)).clamp_(-1.0, 1.0)


# ------------------------------------------------------------------ the component root


class AutoencoderKLMiniMaxH3Audio(nn.Module):
    """The H3 audio VAE component root — 917 destinations, all F32.

        encoder        110 destinations   DAC trunk, total stride 800
        pre_block       22 destinations   causal attention + GeGLU, 2048 -> 32
        mean/logs        4 destinations   posterior head (`logs_proj` is unused at inference)
        dec_in_proj      2 destinations   32 -> 2048
        decoder        777 destinations   BigVGAN, 7 stages x 3 kernels
        latents_*        2 destinations   the per-channel normalization statistics

    The stereo pair is folded into the batch: both channels run the same mono graph, which
    is why `output_channel` never appears in a shape.

    THE NAME IS UPSTREAM'S, VERBATIM (#580) — diffusers'
    `autoencoder_kl_minimax_h3_audio.AutoencoderKLMiniMaxH3Audio`, the class this one
    mirrors. The module path says which is which: `h3_arch.audio_vae` is the port,
    `diffusers` is the original.
    """

    def __init__(self, config: AudioVaeConfig) -> None:
        super().__init__()
        if not config.attn_proj:
            raise ValueError(
                "attn_proj=False names a topology with no `pre_block` at all, which the "
                "pinned release does not carry and this file will not guess at"
            )
        self.config = config
        self.sample_rate = config.sample_rate
        #: 2*4*4*5*5 = 800 audio samples per latent frame, so 40 latent frames per second.
        self.hop_length = math.prod(config.encoder_rates)
        self.latents_per_second = config.sample_rate // self.hop_length

        self.encoder = Encoder(config.encoder_dim, config.encoder_rates, config.latent_dim)
        self.pre_block = AttnProjection(config.latent_dim, config.latent_channels, num_heads=8)

        self.mean_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)
        #: Present in the checkpoint, unused at inference: `encode` returns the posterior
        #: mean and never samples, so the log-variance head is filled and never called.
        self.logs_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)

        self.dec_in_proj = nn.Conv1d(config.latent_channels, config.latent_dim, 1)
        self.decoder = BigVGAN(config)

        self.register_buffer("latents_mean", torch.empty(config.latent_channels), persistent=True)
        self.register_buffer("latents_std", torch.empty(config.latent_channels), persistent=True)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype the FILL PLANE gave these weights, read from a parameter. The endpoint
        does not choose it: the artifact's encoding and the runtime's delivery do."""
        return self.dec_in_proj.weight.dtype

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, ch, L] samples in [-1, 1] at 32 kHz -> normalized latents [B, 32, ch, T].

        L is right-padded with zeros to a whole number of latent frames.
        """
        waveform = waveform.to(self.compute_dtype)
        batch, channels, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        waveform = F.pad(waveform, (0, right_pad))
        x = self.encoder(waveform.reshape(batch * channels, 1, -1))
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        z = (z - self.latents_mean.view(1, -1, 1)) / self.latents_std.view(1, -1, 1)
        return z.reshape(batch, channels, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents [B, 32, ch, T] -> [B, ch, L] samples clamped to [-1, 1].

        ACTIVATIONS FOLLOW THE WEIGHTS. See `video_vae.encode` for the defect this line
        answers: the reference loader casts the WEIGHT at use time and this port cannot,
        because moving a weight is the runtime's job and not an author's."""
        latents = latents.to(self.compute_dtype)
        batch, latent_channels, channels, frames = latents.shape
        z = latents.permute(0, 2, 1, 3).reshape(batch * channels, latent_channels, frames)
        z = z * self.latents_std.view(1, -1, 1) + self.latents_mean.view(1, -1, 1)
        waveform = self.decoder(self.dec_in_proj(z))
        return waveform.reshape(batch, channels, -1)
