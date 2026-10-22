"""The video VAE decoder's fused glue (h3a-096): the elementwise chains between its GEMMs.

The 36-block ViT decoder runs under float16 autocast with a float32 residual stream. Eager,
each block spends a third of its GPU time outside GEMMs and attention (2026-10-18, RTX 4090,
one 28-tile chunk: 0.79 s of 2.37 s). That time goes on float32 norms, the casts around them,
the per-head q/k norms with their casts, the rotary `cat`s, the float32 scale-adds and SwiGLU.
Here each chain is one pass over rows:

- `norm`: `rmsnorm(h) * w`, cast to float16 (norm1);
- `add_norm`: `h + branch * scale` in float32, then norm2 from the same registers;
- `add`: the closing `h + branch * scale`;
- `qk`: the per-head RMSNorm, its float16 cast and the partial rotary, for q and for k;
- `swiglu`: `value * silu(gate)`.

Numerics are eager's. Every float16 rounding eager makes is made here too (`_rh`, inline PTX
the compiler cannot fold), the scale-add is a separate IEEE multiply and add, and SiLU is
torch's `x / (1 + exp(-x))`. The norms reduce in another order than torch's kernel and may
differ by one ulp. Installed by `H3Model.execution_plan` when the decoder has the release
shape and float16 or int8.convrot linears. Otherwise, or while its kernels build, eager serves.

Measured 2026-10-18 on an RTX 4090 with the release weights and a real 124-frame latent,
weights resident: 1.93-2.10 s a chunk against 2.45-2.54 s (5 s decode 13.8 s against 17.3 s).
Non-GEMM time fell from 0.79 s to 0.24 s a chunk. Pixels sit at 71.7 dB from eager (worst
frame 70.3 dB); 99.6 % are identical and none is more than one 8-bit level off.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import triton
import triton.language as tl
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoAttnProcessor,
    MiniMaxH3VideoTransformerBlock,
)
from triton.language.extra import libdevice

from cozy_runtime.author.kernels import INT8_CONVROT, KernelsUnavailable, Launch, Plan, TritonEntry, TritonKernels

from .glue import record

#: The release decoder's widths: the only ones compiled.
HIDDEN, HEAD_DIM, ROTARY, FFN = 2048, 64, 48, 8192
_ROW_WARPS = 8
_QK_HEADS = 8
_SWIGLU_BLOCK = 1024


@triton.jit
def _rh(x):  # type: ignore[no-untyped-def]
    # fp32 -> fp16 -> fp32, round to nearest even, as an opaque instruction pair: the compiler
    # would fold `x.to(fp16).to(fp32)` and keep precision eager does not have.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _norm_kernel(x_ptr, out_ptr, weight_ptr, n_cols, eps, X16: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, (x * inv * weight).to(tl.float16), mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _add_norm_kernel(  # type: ignore[no-untyped-def]
    h_ptr,
    branch_ptr,
    scale_ptr,
    weight_ptr,
    hidden_ptr,
    out_ptr,
    n_cols,
    eps,
    H16: tl.constexpr,
    NORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offset = row * n_cols + cols
    h = tl.load(h_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
    # Eager: `h + branch * scale`, a float32 multiply then a float32 add, never an FMA.
    h = libdevice.add_rn(h, libdevice.mul_rn(branch, scale))
    tl.store(hidden_ptr + offset, h, mask=mask)
    if NORM:
        inv = tl.math.rsqrt(tl.sum(h * h, axis=0) / n_cols + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        tl.store(out_ptr + offset, (h * inv * weight).to(tl.float16), mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _qk_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    seq,
    heads,
    eps,
    HEAD_DIM: tl.constexpr,
    ROTARY: tl.constexpr,
    HEADS: tl.constexpr,
):
    # One program per (row, HEADS heads). Eager: `norm(x.float()).to(fp16)`, then on the first
    # ROTARY channels `x * cos + cat([-x2, x1]) * sin`, every product and the sum in fp16.
    row = tl.program_id(0).to(tl.int64)
    token = row % seq
    head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)[:, None]
    chan = tl.arange(0, HEAD_DIM)[None, :]
    mask = head < heads
    base = (row * heads + head) * HEAD_DIM
    x = tl.load(x_ptr + base + chan, mask=mask, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1)[:, None] / HEAD_DIM + eps)
    normed = _rh(x * inv)
    half = ROTARY // 2
    rotating = chan < ROTARY
    low = chan < half
    partner = tl.where(low, chan + half, chan - half)
    paired = tl.load(x_ptr + base + partner, mask=mask & rotating, other=0.0).to(tl.float32)
    rotated = _rh(paired * inv)
    rotated = tl.where(low, -rotated, rotated)
    cos = _rh(tl.load(cos_ptr + token * ROTARY + chan, mask=rotating, other=1.0))
    sin = _rh(tl.load(sin_ptr + token * ROTARY + chan, mask=rotating, other=0.0))
    turned = _rh(_rh(normed * cos) + _rh(rotated * sin))
    out = tl.where(rotating, turned, normed)
    tl.store(out_ptr + base + chan, out.to(tl.float16), mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _swiglu_kernel(x_ptr, out_ptr, half, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    for start in range(0, half, BLOCK):
        cols = start + offsets
        mask = cols < half
        value = tl.load(x_ptr + row * 2 * half + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * 2 * half + half + cols, mask=mask, other=0.0).to(tl.float32)
        # torch's fp16 SiLU: computed in fp32 with an exact exp and IEEE divide, stored fp16.
        activated = _rh(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)))
        tl.store(out_ptr + row * half + cols, (value * activated).to(tl.float16), mask=mask)


def _entries() -> list[TritonEntry]:
    block = triton.next_power_of_2(HIDDEN)
    rows = [
        TritonEntry(
            _norm_kernel,
            {"x_ptr": "*fp16" if x16 else "*fp32", "out_ptr": "*fp16", "weight_ptr": "*fp32",
             "n_cols": "i32", "eps": "fp32", "X16": "constexpr", "BLOCK": "constexpr"},
            {"X16": x16, "BLOCK": block},
            ("x_ptr", "out_ptr", "weight_ptr", "n_cols"),
            _ROW_WARPS,
        )
        for x16 in (0, 1)
    ]
    rows += [
        TritonEntry(
            _add_norm_kernel,
            {"h_ptr": "*fp16" if h16 else "*fp32", "branch_ptr": "*fp16", "scale_ptr": "*fp32",
             "weight_ptr": "*fp32", "hidden_ptr": "*fp32", "out_ptr": "*fp16" if norm else "*fp32",
             "n_cols": "i32",
             "eps": "fp32", "H16": "constexpr", "NORM": "constexpr", "BLOCK": "constexpr"},
            {"H16": h16, "NORM": norm, "BLOCK": block},
            ("h_ptr", "branch_ptr", "scale_ptr", "weight_ptr", "hidden_ptr", "out_ptr", "n_cols"),
            _ROW_WARPS,
        )
        for h16, norm in ((1, 1), (0, 1), (0, 0))
    ]
    rows.append(
        TritonEntry(
            _qk_kernel,
            {"x_ptr": "*fp16", "out_ptr": "*fp16", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
             "seq": "i32", "heads": "i32", "eps": "fp32", "HEAD_DIM": "constexpr",
             "ROTARY": "constexpr", "HEADS": "constexpr"},
            {"HEAD_DIM": HEAD_DIM, "ROTARY": ROTARY, "HEADS": _QK_HEADS},
            ("x_ptr", "out_ptr", "cos_ptr", "sin_ptr"),
            4,
        )
    )
    rows.append(
        TritonEntry(
            _swiglu_kernel,
            {"x_ptr": "*fp16", "out_ptr": "*fp16", "half": "i32", "BLOCK": "constexpr"},
            {"BLOCK": _SWIGLU_BLOCK},
            ("x_ptr", "out_ptr", "half"),
            8,
        )
    )
    return rows


GLUE = TritonKernels("minimax-h3-vae-glue", _entries())


def _rows(x: Any, width: int) -> Any:
    return x.reshape(-1, width).contiguous()


def norm(launch: Launch, h: Any, weight: Any, eps: float) -> Any:
    rows = _rows(h, HIDDEN)
    out = torch.empty(rows.shape, dtype=torch.float16, device=h.device)
    x16 = int(rows.dtype is torch.float16)
    launch(_norm_kernel, (rows.shape[0], 1, 1), rows, out, weight, HIDDEN, float(eps), x16,
           triton.next_power_of_2(HIDDEN))
    return out.view(*h.shape)


def add(launch: Launch, h: Any, branch: Any, scale: Any, weight: Any = None, eps: float = 0.0) -> tuple[Any, Any]:
    """`h + branch * scale` in float32 and, given `weight`, its float16 RMSNorm."""
    rows, branch_rows = _rows(h, HIDDEN), _rows(branch, HIDDEN)
    hidden = torch.empty(rows.shape, dtype=torch.float32, device=h.device)
    out = torch.empty(rows.shape, dtype=torch.float16, device=h.device) if weight is not None else hidden
    launch(_add_norm_kernel, (rows.shape[0], 1, 1), rows, branch_rows, scale,
           weight if weight is not None else scale, hidden, out, HIDDEN, float(eps),
           int(rows.dtype is torch.float16), int(weight is not None), triton.next_power_of_2(HIDDEN))
    return hidden.view(*h.shape), out.view(*h.shape) if weight is not None else None


def qk(launch: Launch, x: Any, cos: Any, sin: Any, eps: float) -> Any:
    """Per-head RMSNorm, float16 cast and partial rotary on `(batch, seq, heads, head_dim)`."""
    batch, seq, heads, _ = (int(d) for d in x.shape)
    rows = x.reshape(batch * seq, heads * HEAD_DIM).contiguous()
    out = torch.empty_like(rows)
    launch(_qk_kernel, (batch * seq, -(-heads // _QK_HEADS), 1), rows, out,
           cos.reshape(-1, ROTARY).contiguous(), sin.reshape(-1, ROTARY).contiguous(), seq, heads,
           float(eps), HEAD_DIM, ROTARY, _QK_HEADS)
    return out.view(*x.shape)


def swiglu(launch: Launch, x: Any) -> Any:
    rows = _rows(x, 2 * FFN)
    out = torch.empty(rows.shape[0], FFN, dtype=torch.float16, device=x.device)
    launch(_swiglu_kernel, (rows.shape[0], 1, 1), rows, out, FFN, _SWIGLU_BLOCK)
    return out.view(*x.shape[:-1], FFN)


class FusedVideoSwiGLU(SwiGLU):  # type: ignore[misc]
    _glue: Launch

    def forward(self, hidden_states: Any) -> Any:
        return swiglu(self._glue, self.proj(hidden_states))


class FusedVideoAttnProcessor(MiniMaxH3VideoAttnProcessor):  # type: ignore[misc]
    _glue: Launch

    def __call__(self, attn: Any, hidden_states: Any, rotary_emb: Any = None) -> Any:
        if rotary_emb is None:
            return super().__call__(attn, hidden_states, rotary_emb)
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        cos, sin = rotary_emb
        query = qk(self._glue, query, cos, sin, attn.norm_q.eps)
        key = qk(self._glue, key, cos, sin, attn.norm_k.eps)
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=None, backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.to_out[0](hidden_states.flatten(2, 3))


class FusedVideoTransformerBlock(MiniMaxH3VideoTransformerBlock):  # type: ignore[misc]
    _glue: Launch

    def forward(self, hidden_states: Any, rotary_emb: Any = None) -> Any:
        if not _fusable(hidden_states, rotary_emb):
            return MiniMaxH3VideoTransformerBlock.forward(self, hidden_states, rotary_emb)
        normed = norm(self._glue, hidden_states, self.norm1.weight, self.norm1.eps)
        branch = self.attn(normed, rotary_emb)
        hidden_states, normed = add(self._glue, hidden_states, branch, self.scale1, self.norm2.weight, self.norm2.eps)
        hidden_states, _ = add(self._glue, hidden_states, self.ff(normed), self.scale2)
        return hidden_states


def _fusable(hidden_states: Any, rotary_emb: Any) -> bool:
    """The kernels' operands: the served decode's float16 autocast on CUDA."""
    return (
        rotary_emb is not None
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.float32)
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") is torch.float16
    )


def _linear(module: Any) -> bool:
    """A float16 `nn.Linear`, or the int8 lane's encoded leaf (it casts from and to the autocast dtype)."""
    if getattr(module, "provider", None) == INT8_CONVROT:
        return True
    weight = getattr(module, "weight", None)
    return weight is not None and weight.dtype is torch.float16


def _matches(block: Any) -> bool:
    attn, ff = block.attn, block.ff
    return (
        type(block) is MiniMaxH3VideoTransformerBlock
        and type(attn.processor) is MiniMaxH3VideoAttnProcessor
        and type(ff) is FeedForward
        and type(ff.net[0]) is SwiGLU
        and type(ff.net[0].activation) is torch.nn.SiLU
        and attn.heads * attn.dim_head == HIDDEN
        and attn.dim_head == HEAD_DIM
        and attn.norm_q.weight is None
        and attn.norm_k.weight is None
        and block.norm1.weight is not None
        and block.norm1.weight.dtype is torch.float32
        and block.norm2.weight.dtype is torch.float32
        and block.scale1.dtype is torch.float32
        and block.scale2.dtype is torch.float32
        and ff.net[0].proj.out_features == 2 * FFN
        and all(_linear(m) for m in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0], ff.net[0].proj, ff.net[2]))
        and not any(
            m._forward_hooks or m._forward_pre_hooks
            for m in (attn, ff, ff.net[0], block.norm1, block.norm2, attn.norm_q, attn.norm_k)
        )
    )


def identity() -> str:
    return "sha256:" + hashlib.sha256(b"minimax-h3.vae-glue\0" + GLUE.identity.encode()).hexdigest()


def _install(blocks: Any, launch: Launch) -> None:
    for block in blocks:
        processor = FusedVideoAttnProcessor()
        processor._attention_backend = block.attn.processor._attention_backend
        processor._parallel_config = block.attn.processor._parallel_config
        processor._glue = launch
        block.attn.set_processor(processor)
        block.ff.net[0].__class__ = FusedVideoSwiGLU
        block.ff.net[0]._glue = launch
        block.__class__ = FusedVideoTransformerBlock
        block._glue = launch


def install(model: object, vae: Any, plan: Plan) -> str:
    """The fused glue on `vae`'s decoder blocks once its kernels are built (now, or at the next
    request's start); "" when the decoder is not the release shape or the device cannot. The
    outcome is reported as `vae_glue` (`glue.record`): eager with its code, or deferred until
    the install lands and then fused."""
    device = torch.device(plan.device)
    decoder = getattr(vae, "decoder", None)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        reason = f"decoder glue needs CUDA sm80+, not {device}"
        return _report(model, plan, "fusion_device_incapable", reason)
    if decoder is None:
        return _report(model, plan, "fusion_no_transformer", "the VAE has no transformer decoder")
    blocks = list(decoder.transformer_blocks)
    if not blocks or not all(_matches(b) for b in blocks) or decoder.rope.inv_freq.numel() * 6 != ROTARY:
        return _report(model, plan, "fusion_block_shape", "the decoder is not the release shape")
    return _fuse(model, plan, blocks)


def _fuse(model: object, plan: Plan, blocks: list[Any]) -> str:
    try:
        if plan.when_ready is None:
            _landed(model, plan, blocks, plan.load(GLUE))
        else:
            # Reported first, so an install that lands at once reports over it.
            reason = "fuses at a request boundary once its kernels are built"
            _report(model, plan, "deferred", reason, len(blocks))
            plan.when_ready(GLUE, lambda launch: _landed(model, plan, blocks, launch))
    except KernelsUnavailable as exc:
        return _report(model, plan, f"fusion_{exc.code}", str(exc))
    return identity()


def _landed(model: object, plan: Plan, blocks: list[Any], launch: Launch) -> None:
    try:
        _install(blocks, launch)
    except Exception as exc:
        _report(model, plan, "fusion_install_failed", f"{type(exc).__name__}: {exc}", len(blocks))
        raise
    document = {"applied": True, "code": "", "reason": "", "sites": len(blocks)}
    record(model, plan, "vae_glue", {**document, "identity": identity()})


def _report(model: object, plan: Plan, code: str, reason: str, sites: int = 0) -> str:
    document = {"applied": False, "code": code, "reason": reason, "sites": sites, "identity": ""}
    record(model, plan, "vae_glue", document)
    return ""
