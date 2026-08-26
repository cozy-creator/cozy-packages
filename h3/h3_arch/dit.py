"""The MiniMax H3 audio-video DiT — one single-stream packed-token transformer denoising
24-channel video latents and 32-channel stereo audio latents JOINTLY.

The packed sequence is `[text | conditions | target audio | target video]`, uniform per
segment in (modality tag, timestep class), which is what makes the whole modulation plane
computable from a handful of distinct timesteps instead of per row. Geometry, packing and
the timestep plan are pure CPU work and live in `layout.py`; this file is the weights.

Ported from ComfyUI v0.33.0 `comfy/ldm/minimax/model.py` — the loader proto-001 actually
benchmarked the selected candidate under — with the infrastructure removed (see the package
docstring). Two ports of one kernel are worth naming because they are where a silent
numeric drift would hide:

  * `comfy_kitchen`'s fused `rms_rope_split_half_` writes per-head RMSNorm and a PARTIAL
    split-half rope in place on the fused qkv buffer. It is reproduced here eagerly, from
    that package's own eager reference, at the same rot_dim: 96 of each head's 128 dims
    rotate and 32 pass through.
  * `comfy.ops.linear_input_act(fc2, fc1(x), "swiglu")` folds SwiGLU into an INT8 input
    quantizer when the weight is INT8 and is `fc2(silu(gate) * up)` otherwise. The fold is
    a quantization-kernel optimization for an encoding this endpoint does not serve, so
    only the eager form survives.

DTYPES ARE THE CHECKPOINT'S, not comfy's. Upstream constructs the AdaLN projections in
fp32 and upcasts on load; the released curve carrier STORES them fp16, and a destination
whose declared dtype is not the stored one is not a `plain` fill. So every destination here
is declared at the dtype the artifact carries and `scripts/h3-keys.py` checks it. The
consequence is real and named: the AdaLN matmul runs at fp16 input width rather than fp32,
which is a numeric difference from the reference loader and is UNVERIFIED until it runs
beside that loader on real weights.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import AdaLnStructure, DitConfig

#: The condition rows' pinned timestep classes. A visual condition sits at 0.999 rather
#: than 1.0 so it is near-clean but not identically the clean-latent class; audio pins
#: exactly at 1.0. Both are the checkpoint's training convention, not a tunable.
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def time_shift_sigma(sigma: Tensor | float, from_shift: float, to_shift: float) -> Tensor:
    """Map one flow-match schedule's sigma onto another's shift, in closed form.

    H3 runs video at shift 12 and audio at shift 3 off the SAME sampler clock, so the audio
    stream's own sigma is always derivable and never separately sampled."""
    s = torch.as_tensor(sigma, dtype=torch.float32)
    base = s / (from_shift + s * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(latent: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw] rows in (t, h, w) order."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(
    rows: Tensor, t: int, h: int, w: int, c: int, patch_size: tuple[int, int, int]
) -> Tensor:
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent: Tensor) -> Tensor:
    """[B, C=32, ch=2, T] -> [ch*T, 32], channel-major: all of ch0's frames, then ch1's."""
    _, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows: Tensor, ch: int = 2) -> Tensor:
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def rope_rotation_table(angles: Tensor, dtype: torch.dtype) -> Tensor:
    """[S, rot_dim] pair angles -> [1, S, 1, rot_dim/2, 2, 2] rotation matrices."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]  # the halves are duplicates by construction
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


def _apply_rope_split_half(x: Tensor, freqs: Tensor) -> Tensor:
    """`comfy_kitchen.backends.eager.apply_rope_split_half1`, verbatim in behaviour.

    Split-half means the pair partner of dim j is dim j + rot/2, not j + 1."""
    t = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs.dtype)
    out = freqs[..., 0] * t[..., 0] + freqs[..., 1] * t[..., 1]
    return out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def _rms_rope_split_half(x: Tensor, freqs: Tensor, scale: Tensor, eps: float) -> Tensor:
    """Per-head RMSNorm then PARTIAL split-half rope — the first `2*freqs.shape[-3]` dims
    rotate and the tail passes through untouched."""
    normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=scale, eps=eps)
    rot = freqs.shape[-3] * 2
    if rot and rot != x.shape[-1]:
        return torch.cat(
            (_apply_rope_split_half(normed[..., :rot], freqs), normed[..., rot:]), dim=-1
        )
    return _apply_rope_split_half(normed, freqs)


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """[1, heads, S, head_dim] x3 -> [S, heads*head_dim].

    KERNEL CHOICE IS AUTHOR TERRITORY and placement is not (§3.2): this is the one line an
    expert kernel (SageAttention-class) composes AROUND, by calling the leaf and never
    reading its weights. SDPA is the launch choice because it needs no vendored build."""
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(1, out.shape[2], -1).squeeze(0)


class TimeEmbedder(nn.Module):
    """The FULL structure's sinusoidal timestep embedder. A curve artifact does not carry
    it — its `adaln_t_table` is the sampled curve this would have produced."""

    def __init__(self, freq_dim: int, hidden: int, out: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True, dtype=dtype)
        self.proj_out = nn.Linear(hidden, out, bias=True, dtype=dtype)

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # cos BEFORE sin
        return self.proj_out(nn.functional.silu(self.proj_in(emb)))


class Attention(nn.Module):
    def __init__(
        self, hidden: int, heads: int, head_dim: int, eps: float, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.eps = eps
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False, dtype=dtype)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype)

    def forward(self, x: Tensor, rope_freqs: Tensor | None = None) -> Tensor:
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = _rms_rope_split_half(q, rope_freqs, self.q_norm.weight, self.eps)
            k = _rms_rope_split_half(k, rope_freqs, self.k_norm.weight, self.eps)
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return self.out_proj(
            attention(
                q[0].transpose(0, 1).unsqueeze(0),
                k[0].transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
            )
        )


class MLP(nn.Module):
    """SwiGLU: `fc1` produces gate and up in one projection, `fc2` brings it back."""

    def __init__(self, hidden: int, ffn: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(gate) * up)


class AdalnProj(nn.Module):
    """One block's modulation projection: `[M, t_dim] -> expand tensors of [M*3, hidden]`.

    `t_dim` is the whole difference between the two live structures. FULL feeds it the 2688
    -wide timestep embedding through a SiLU; CURVE feeds it 8 interpolated coordinates of a
    shared basis and applies no SiLU, because the curve already absorbed it. That is the
    roughly 26 GB the pruned topology does not carry, and it is one integer."""

    def __init__(
        self,
        t_dim: int,
        hidden: int,
        expand: int,
        modalities: int,
        *,
        apply_silu: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype)

    def forward(self, t_emb: Tensor) -> tuple[Tensor, ...]:
        source = t_emb.to(self.linear.weight.dtype)
        x = self.linear(nn.functional.silu(source) if self.apply_silu else source)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        chunks: tuple[Tensor, ...] = x.chunk(self.expand, dim=-1)
        return chunks


Segments = list[tuple[int, int, int]]


def _mod_scale_shift(h: Tensor, shift: Tensor, scale: Tensor, segments: Segments) -> Tensor:
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x: Tensor, gate: Tensor, other: Tensor, segments: Segments) -> Tensor:
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


class RefinerBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        eps: float,
        qk_eps: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype)
        self.mlp = MLP(hidden, ffn, dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(self.norm1(x)).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    """Two unmodulated blocks over the projected Qwen states. No AdaLN: the text stream
    carries no timestep of its own before it enters the packed sequence."""

    def __init__(
        self,
        num_layers: int,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        eps: float,
        qk_eps: float,
        final_eps: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        t_dim: int,
        eps: float,
        qk_eps: float,
        *,
        apply_silu: bool,
        adaln_dtype: torch.dtype,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype)
        self.mlp = MLP(hidden, ffn, dtype)
        self.adaln_proj = AdalnProj(
            t_dim, hidden, 6, 3, apply_silu=apply_silu, dtype=adaln_dtype
        )

    def forward(
        self, x: Tensor, t_emb: Tensor, mod_segments: Segments, rope_freqs: Tensor
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class FinalLayer(nn.Module):
    """Two heads over one shared norm — the video and audio streams leave together.

    `video_out` and `audio_out` are the checkpoint's fp32 island: the last projection
    before a latent is handed to a VAE runs at full width in the released weights."""

    def __init__(
        self,
        hidden: int,
        t_dim: int,
        video_dim: int,
        audio_dim: int,
        eps: float,
        *,
        apply_silu: bool,
        adaln_dtype: torch.dtype,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.adaln_proj = AdalnProj(
            t_dim, hidden, 2, 1, apply_silu=apply_silu, dtype=adaln_dtype
        )
        self.video_out = nn.Linear(hidden, video_dim, bias=True, dtype=torch.float32)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32)

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        video_seg: tuple[int, int, int],
        audio_seg: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


class MiniMaxH3Dit(nn.Module):
    """The component root. One of these per TASK SLOT — two instances of this same class
    with two different sets of weights, never one graph with a partition selector."""

    def __init__(self, config: DitConfig) -> None:
        super().__init__()
        self.config = config
        dtype = torch.bfloat16
        curve = config.structure is AdaLnStructure.CURVE
        # AdaLN weights ship fp16 in the released curve carrier; declaring anything else
        # would make a `plain` fill a conversion. See the module docstring.
        adaln_dtype = torch.float16
        hidden = config.hidden_size

        self.video_patch_proj = nn.Linear(
            config.video_patch_dim, hidden, bias=True, dtype=torch.float32
        )
        self.audio_patch_proj = nn.Linear(
            config.audio_latents_dim, hidden, bias=True, dtype=torch.float32
        )
        self.condition_proj = nn.Linear(config.text_dim, hidden, bias=True, dtype=dtype)
        if curve:
            self.register_buffer(
                "adaln_t_table",
                torch.empty(config.adaln_curve_grid, config.time_embed_dim, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.time_embedder = TimeEmbedder(
                config.timestep_input_dim,
                config.time_embed_hidden_size,
                config.time_embed_dim,
                dtype=torch.float32,
            )
        self.rope = nn.Module()
        self.rope.register_buffer(
            "inv_freq", torch.empty(config.rope_inv_freq_len, dtype=torch.float32), persistent=True
        )
        self.token_refiner = TokenRefiner(
            config.token_refiner_num_layers,
            hidden,
            config.num_attention_heads,
            config.attention_head_dim,
            config.ffn_hidden_size,
            config.norm_eps,
            config.qk_norm_eps,
            config.final_norm_eps,
            dtype,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.ffn_hidden_size,
                    config.time_embed_dim,
                    config.norm_eps,
                    config.qk_norm_eps,
                    apply_silu=not curve,
                    adaln_dtype=adaln_dtype,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = FinalLayer(
            hidden,
            config.time_embed_dim,
            config.video_patch_dim,
            config.audio_latents_dim,
            config.final_norm_eps,
            apply_silu=not curve,
            adaln_dtype=adaln_dtype,
            dtype=dtype,
        )

    @property
    def uses_curve(self) -> bool:
        return self.config.structure is AdaLnStructure.CURVE

    def refine_text(self, text_states: Tensor) -> Tensor:
        """[L, text_dim] Qwen states -> [L, hidden]. Already-hidden-width states pass."""
        if text_states.shape[-1] == self.config.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states))

    def rope_freqs(self, position_ids: Tensor, dtype: torch.dtype) -> Tensor:
        """[S, 3] (t, h, w) coordinates -> the [1, S, 1, 48, 2, 2] rotation table."""
        pos = position_ids.to(torch.float32).to(self.rope.inv_freq.device)
        per_axis = pos.unsqueeze(-1) * self.rope.inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return rope_rotation_table(torch.cat((half, half), dim=-1), dtype)

    def timestep_embedding(self, t_values: Tensor) -> Tensor:
        """The distinct timesteps of one step, embedded once and reused by every block."""
        if not self.uses_curve:
            return self.time_embedder(t_values)
        table = self.adaln_t_table
        pos = t_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        # max-clamp keeps t=1.0 on the last interval instead of reading past the table
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    def forward(
        self,
        packed: Tensor,
        t_values: Tensor,
        mod_segments: Segments,
        position_ids: Tensor,
        video_seg: tuple[int, int, int],
        audio_seg: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        """ONE denoise evaluation over an ALREADY PACKED sequence.

        The packing, the modulation segment table and the timestep plan are the endpoint's
        (`layout.py`) and arrive resolved: this signature is what makes `denoise` a single
        declared component scope over the transformer and nothing else.

        THE OUTPUT CONVENTION, because half of se-002's first render was this sentence not
        being written down (#522a): these heads are DATA-WARD VELOCITY, pointing from
        noise toward data, in ROW shape. The solver consumes them as
        `x + (sigma - sigma_next) * v` and `layout.H3Solver` says so on its own side.

        ComfyUI returns `[-video_out, -audio_out]` from the same computation because its
        GENERIC flow sampler assumes the opposite sign and then applies the opposite
        delta; the two flips compose to this same update. Diffusers' dedicated
        `MiniMaxH3Scheduler` keeps the heads raw exactly as here and documents the
        convention in the scheduler. Taking ComfyUI's negation WITHOUT its delta — or, as
        happened, its delta without its negation — is 30 evaluations of anti-denoising,
        and there is no numeric symptom short of the finished video.

        NO PROGRESS CALLBACK. A denoise evaluation is one unit of progress; emitting one
        per BLOCK reported 50 advances for a single step and made the 30-step bar
        meaningless. The solver owns the count because the solver owns the loop (#522e).
        """
        t_emb = self.timestep_embedding(t_values)
        rope_freqs = self.rope_freqs(position_ids, packed.dtype)
        h = packed
        for block in self.blocks:
            h = block(h, t_emb, mod_segments, rope_freqs)
        out: tuple[Tensor, Tensor] = self.final_layer(h, t_emb, video_seg, audio_seg)
        return out
