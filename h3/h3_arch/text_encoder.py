"""The H3 text encoder — Qwen3-VL-32B truncated to 50 layers, 902 destinations, BF16.

A Qwen3 decoder stack (64 query heads over 8 KV heads, per-head q/k RMSNorm, interleaved
MRoPE) beside the 27-block Qwen3-VL vision tower, and nothing after them: the released
checkpoint carries no final norm and no LM head, and its own header metadata says what it
is for — `{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}`.
Layer 50's raw output IS the conditioning the DiT cross-attends to.

Vision enters twice. Once as embeddings: a block of patches runs the tower, the merger
folds each 2x2 spatial group into one 5120-wide token, and those tokens are SPLICED into
the token-embedding sequence at their pad positions. Once as deepstack: three intermediate
tower activations get their own mergers and are ADDED into the language stream at the
vision positions, one per decoder layer, over the first three layers.

WHAT THIS FILE DOES NOT DO: no device call, no dtype cast to anywhere but this module's own
parameter dtype, no KV cache, no tokenizer. `forward` is one encoder pass and a pure
function of its arguments and this module's parameters; where the weights live and when
they arrived is the runtime's decision, made before entry.

Two facts that are NOT in the checkpoint and so are stated here rather than inferred: the
rotary inverse-frequency tables (computed per call — a registered buffer would be a 903rd
destination the fill plane has nothing to fill) and `_ROPE_SECTIONS`.

Lineage of the reference implementation: ComfyUI v0.33.0 `comfy/text_encoders/{minimax,
qwen3vl,qwen35,llama}.py` (GPL-3.0), the loader family the selected checkpoint was
benchmarked under, with the injectable-ops indirection, the use-time casts, the prefetch
queue and the generation path removed.
"""

#  The checking venv holds no torch, so `nn.Module` is `Any` and strict mode reports every
#  class here. Same true-about-CI/false-about-the-code pair pyproject.toml already loosens
#  for the two model modules; nothing else is relaxed.
# mypy: disallow-subclassing-any=false

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import TextEncoderConfig

#: Every one of the 902 destinations is BF16 in the release header. A stored dtype is a
#: fact about the artifact; what a destination is FILLED at is the fill plane's decision.
_DTYPE = torch.bfloat16

#: Qwen3-VL's interleaved-MRoPE section widths (T, H, W) over the 64 rotary pairs of a
#: 128-wide head. NOT derivable from the header — this is the upstream Qwen3-VL family
#: constant, and it changes numerics without changing a single key.
_ROPE_SECTIONS = (24, 20, 20)

#: The vision tower's own rotary base, independent of the language stack's `rope_theta`.
_VISION_ROPE_THETA = 10000.0


# ------------------------------------------------------------------ rotary


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate in fp32 and come back — upstream's split/addcmul form, same arithmetic."""
    dtype = q.dtype
    qf, kf = q.float(), k.float()
    return (
        (qf * cos + _rotate_half(qf) * sin).to(dtype),
        (kf * cos + _rotate_half(kf) * sin).to(dtype),
    )


def _language_rope(
    position_ids: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) broadcastable over [batch, heads, seq, head_dim], in fp32.

    A 3-row `position_ids` is the MRoPE table (T/H/W); Qwen3-VL interleaves rather than
    sectioning it — T frequencies everywhere, H and W replacing every third slot inside
    their own section width.
    """
    if position_ids.dim() == 1:
        position_ids = position_ids[None]
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32)
            / head_dim
        )
    )
    freqs = (inv_freq[None, :, None] * position_ids[:, None, :].float()).transpose(1, 2)
    if freqs.shape[0] == 3:
        interleaved = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            slot = slice(offset, _ROPE_SECTIONS[axis] * 3, 3)
            interleaved[..., slot] = freqs[axis][..., slot]
        emb = torch.cat((interleaved, interleaved), dim=-1)[None, None]
    else:
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    return emb.cos(), emb.sin()


def _vision_rope(
    grid: list[tuple[int, int, int]], head_dim: int, merge_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) of shape [tokens, 1, head_dim] for the tower's 2-D rotary.

    Row/column ids are emitted in MERGE ORDER — the four patches of a 2x2 group are
    adjacent — which is the same order `fast_pos_embed_interpolate` and the merger use.
    """
    dim = head_dim // 2
    inv_freq = 1.0 / (
        _VISION_ROPE_THETA ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    span = max(max(h, w) for _, h, w in grid)
    table = torch.outer(torch.arange(span, device=device, dtype=torch.float32), inv_freq)
    ids = []
    for frames, height, width in grid:
        blocks_h = torch.arange(height // merge_size, device=device)
        blocks_w = torch.arange(width // merge_size, device=device)
        intra = torch.arange(merge_size, device=device)
        rows = blocks_h[:, None, None, None] * merge_size + intra[None, None, :, None]
        cols = blocks_w[None, :, None, None] * merge_size + intra[None, None, None, :]
        shape = (height // merge_size, width // merge_size, merge_size, merge_size)
        coords = torch.stack((rows.expand(shape).reshape(-1), cols.expand(shape).reshape(-1)), -1)
        ids.append(coords.repeat(frames, 1) if frames > 1 else coords)
    freqs = table[torch.cat(ids)].flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(-2), emb.sin().unsqueeze(-2)


# ------------------------------------------------------------------ the vision tower


class _VisionPatchEmbed(nn.Module):
    """Flattened patches -> tower tokens. The temporal patch is 2, so one token spans two
    frames; a still image is presented as the same frame twice by the caller."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.in_channels = config.vision_in_channels
        self.embed_dim = config.vision_hidden_size
        self.patch = (
            config.vision_temporal_patch_size,
            config.vision_patch_size,
            config.vision_patch_size,
        )
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, self.patch, stride=self.patch, dtype=_DTYPE
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = patches.view(-1, self.in_channels, *self.patch)
        return self.proj(x).view(-1, self.embed_dim)


class _VisionMlp(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        self.linear_fc1 = nn.Linear(hidden, config.vision_intermediate_size, dtype=_DTYPE)
        self.linear_fc2 = nn.Linear(config.vision_intermediate_size, hidden, dtype=_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class _VisionAttention(nn.Module):
    """Full attention WITHIN each frame, never across them — hence the per-frame split."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        self.num_heads = config.vision_num_heads
        self.qkv = nn.Linear(hidden, hidden * 3, dtype=_DTYPE)
        self.proj = nn.Linear(hidden, hidden, dtype=_DTYPE)

    def forward(
        self, x: torch.Tensor, lengths: list[int], cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        tokens = x.shape[0]
        q, k, v = self.qkv(x).reshape(tokens, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = _apply_rope(q, k, cos, sin)
        out = []
        for qs, ks, vs in zip(
            torch.split(q, lengths), torch.split(k, lengths), torch.split(v, lengths), strict=True
        ):
            attended = F.scaled_dot_product_attention(
                qs.transpose(0, 1)[None], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None]
            )
            out.append(attended[0].transpose(0, 1).reshape(qs.shape[0], -1))
        return self.proj(torch.cat(out, dim=0))


class _VisionBlock(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        self.norm1 = nn.LayerNorm(hidden, eps=config.vision_norm_eps, dtype=_DTYPE)
        self.norm2 = nn.LayerNorm(hidden, eps=config.vision_norm_eps, dtype=_DTYPE)
        self.attn = _VisionAttention(config)
        self.mlp = _VisionMlp(config)

    def forward(
        self, x: torch.Tensor, lengths: list[int], cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), lengths, cos, sin)
        return x + self.mlp(self.norm2(x))


class _PatchMerger(nn.Module):
    """PRE-shuffle norm: 1152-wide, applied before the 2x2 group is folded to 4608."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        self.merge_dim = hidden * config.vision_spatial_merge_size**2
        self.norm = nn.LayerNorm(hidden, eps=config.vision_norm_eps, dtype=_DTYPE)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim, dtype=_DTYPE)
        self.linear_fc2 = nn.Linear(self.merge_dim, config.vision_out_hidden_size, dtype=_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class _DeepstackMerger(nn.Module):
    """POST-shuffle norm: 4608-wide, applied after the fold. The one shape difference from
    `_PatchMerger`, and the reason the two cannot share a class."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.merge_dim = config.vision_hidden_size * config.vision_spatial_merge_size**2
        self.norm = nn.LayerNorm(self.merge_dim, eps=config.vision_norm_eps, dtype=_DTYPE)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim, dtype=_DTYPE)
        self.linear_fc2 = nn.Linear(self.merge_dim, config.vision_out_hidden_size, dtype=_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.merge_size = config.vision_spatial_merge_size
        self.head_dim = config.vision_hidden_size // config.vision_num_heads
        #: 2304 learned positions on a 48x48 grid, bilinearly resampled to each real grid.
        self.grid_side = int(config.vision_num_position_embeddings**0.5)
        self.deepstack_layers = tuple(config.vision_deepstack_layers)
        self.patch_embed = _VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.vision_num_position_embeddings, config.vision_hidden_size, dtype=_DTYPE
        )
        self.blocks = nn.ModuleList(_VisionBlock(config) for _ in range(config.vision_depth))
        self.merger = _PatchMerger(config)
        self.deepstack_merger_list = nn.ModuleList(
            _DeepstackMerger(config) for _ in self.deepstack_layers
        )

    def _resampled_pos_embed(self, grid: list[tuple[int, int, int]]) -> torch.Tensor:
        side, merge = self.grid_side, self.merge_size
        weight = self.pos_embed.weight
        out = []
        for frames, height, width in grid:
            rows = torch.linspace(0, side - 1, height, device=weight.device)
            cols = torch.linspace(0, side - 1, width, device=weight.device)
            row_floor, col_floor = rows.int(), cols.int()
            row_ceil = (row_floor + 1).clip(max=side - 1)
            col_ceil = (col_floor + 1).clip(max=side - 1)
            drow = (rows - row_floor)[:, None]
            dcol = (cols - col_floor)[None, :]
            corners = (
                (row_floor, col_floor, (1 - drow) * (1 - dcol)),
                (row_floor, col_ceil, (1 - drow) * dcol),
                (row_ceil, col_floor, drow * (1 - dcol)),
                (row_ceil, col_ceil, drow * dcol),
            )
            terms = [
                self.pos_embed(((row * side)[:, None] + col[None, :]).flatten().long())
                * share.flatten()[:, None].to(weight.dtype)
                for row, col, share in corners
            ]
            embed = (terms[0] + terms[1] + terms[2] + terms[3]).repeat(frames, 1)
            out.append(
                embed.view(frames, height // merge, merge, width // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
        return torch.cat(out)

    def forward(
        self, patches: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """[tokens, ch*t*p*p] patches and a [n, 3] (t, h, w) grid -> merged tokens and the
        three deepstack features, all in merge order."""
        grid = [(int(t), int(h), int(w)) for t, h, w in grid_thw.tolist()]
        x = self.patch_embed(patches.to(dtype=self.patch_embed.proj.weight.dtype))
        x = x + self._resampled_pos_embed(grid)
        cos, sin = _vision_rope(grid, self.head_dim, self.merge_size, x.device)
        #: One attention sequence per FRAME, so a t-frame block is t sequences of h*w.
        lengths = [h * w for t, h, w in grid for _ in range(t)]
        deepstack = []
        for index, block in enumerate(self.blocks):
            x = block(x, lengths, cos, sin)
            if index in self.deepstack_layers:
                deepstack.append(self.deepstack_merger_list[self.deepstack_layers.index(index)](x))
        return self.merger(x), deepstack


# ------------------------------------------------------------------ the language stack


class _Attention(nn.Module):
    """GQA 64:8 with per-head q/k RMSNorm applied BEFORE the rotary, and no biases."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        inner = self.num_heads * self.head_dim
        kv_inner = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner, bias=False, dtype=_DTYPE)
        self.k_proj = nn.Linear(config.hidden_size, kv_inner, bias=False, dtype=_DTYPE)
        self.v_proj = nn.Linear(config.hidden_size, kv_inner, bias=False, dtype=_DTYPE)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False, dtype=_DTYPE)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=_DTYPE)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=_DTYPE)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(self.q_norm(q), self.k_norm(k), cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=seq > 1, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, -1))


class _Mlp(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inner, bias=False, dtype=_DTYPE)
        self.up_proj = nn.Linear(hidden, inner, bias=False, dtype=_DTYPE)
        self.down_proj = nn.Linear(inner, hidden, bias=False, dtype=_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = _Mlp(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=_DTYPE
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=_DTYPE
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class _LanguageModel(nn.Module):
    """The truncated stack: an embedding and 50 layers. No `norm`, no `lm_head` — the
    conversion cut the checkpoint at layer 50 and neither survives it."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=_DTYPE)
        self.layers = nn.ModuleList(
            _DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )


# ------------------------------------------------------------------ the component root


@dataclass(frozen=True, slots=True)
class VisionBlock:
    """One spliced vision block, as the endpoint's conditioning builder assembles it.

    `index` is the sequence position of the block's FIRST merged token — the position just
    after the vision-start token, not the vision-start token itself. `patches` is the
    already-normalized, already-flattened patch matrix and `grid_thw` its [1, 3] grid; both
    are the tokenizer's product, and neither is this module's concern to produce.
    """

    patches: torch.Tensor
    grid_thw: torch.Tensor
    index: int


def _mrope_position_ids(
    spans: list[tuple[int, int, torch.Tensor]], seq_len: int, device: torch.device
) -> torch.Tensor:
    """(3, seq) T/H/W ids: text advances on all three axes together, a vision span holds T
    flat and walks H and W across its own half-resolution grid, and the text after it
    resumes from the span's widest extent rather than from its token count."""
    ids = torch.zeros((3, seq_len), device=device)
    ids[:, : spans[0][0]] = torch.arange(0, spans[0][0], device=device)
    offset = 0
    for start, size, grid in spans:
        end = start + size
        extent = int(grid.max()) // 2
        ids[:, end:] = torch.arange(
            start + extent + offset, start + extent + offset + (seq_len - end), device=device
        )
        ids[0, start:end] = start + offset
        rows = int(grid[0][1]) // 2
        cols = int(grid[0][2]) // 2
        ids[1, start:end] = (
            torch.arange(start + offset, start + rows + offset, device=device)
            .unsqueeze(1)
            .repeat(1, math.ceil(size / rows))
            .flatten(0)[:size]
        )
        ids[2, start:end] = (
            torch.arange(start + offset, start + cols + offset, device=device)
            .unsqueeze(0)
            .repeat(math.ceil(size / cols), 1)
            .flatten(0)[:size]
        )
        offset += extent - size
    return ids


class Qwen3VLForConditionalGeneration(nn.Module):
    """The H3 text encoder component root — 902 destinations, all BF16.

    THE NAME IS UPSTREAM'S, VERBATIM (#579): this is transformers' class name for the
    component diffusers' own `ComponentSpec` binds to the `text_encoder` slot, and a port
    answers to the name of what it ports. The module path is the disambiguator —
    `h3_arch.text_encoder.Qwen3VLForConditionalGeneration` against
    `transformers.…Qwen3VLForConditionalGeneration` — which is the convention diffusers and
    ComfyUI use for the same situation. What upstream's suffix promises and this class does
    NOT carry is stated below rather than left for a reader to discover: there is no head
    here and nothing generates.

        model.embed_tokens     1 destination     151936 x 5120
        model.layers         550 destinations    50 layers, GQA 64:8, MLP 25600
        visual.patch_embed     2 destinations    Conv3d over a 2x16x16 patch
        visual.pos_embed       1 destination     2304 learned grid positions
        visual.blocks        324 destinations    27 blocks, 16 heads, MLP 4304
        visual.merger          6 destinations    2x2 fold, 4608 -> 5120
        visual.deepstack…     18 destinations    three mergers, post-shuffle norm

    There is no final norm and no LM head to build: `forward` returns the unnormalized
    hidden state after layer 50, which is what the release states it is for.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _LanguageModel(config)
        self.visual = Qwen3VLVisionModel(config)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: torch.Tensor | None = None,
        vision: Sequence[VisionBlock] = (),
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[batch, seq] token ids (or `embeds` in their place) -> [batch, seq, 5120].

        `vision` blocks are spliced in at their own indices and their deepstack features
        added at those positions over the first three decoder layers. Vision is single-
        batch: the splice indices belong to one assembled sequence, not to a stack of them.
        """
        x = self.model.embed_tokens(tokens) if embeds is None else embeds
        batch, seq, _ = x.shape
        blocks = sorted(vision, key=lambda block: block.index)

        mask: torch.Tensor | None = None
        deepstack: list[torch.Tensor] | None = None
        spans: list[tuple[int, int, torch.Tensor]] = []
        if blocks:
            if batch != 1:
                raise ValueError(
                    f"vision splicing addresses one assembled sequence, got batch {batch} — "
                    "the endpoint builds one conditioning sequence per request"
                )
            x = x.clone()
            mask = torch.zeros((batch, seq), dtype=torch.bool, device=x.device)
            for block in blocks:
                merged, features = self.visual(block.patches, block.grid_thw)
                stop = block.index + merged.shape[0]
                x[0, block.index : stop] = merged.to(x.dtype)
                mask[0, block.index : stop] = True
                spans.append((block.index, merged.shape[0], block.grid_thw))
                deepstack = (
                    features
                    if deepstack is None
                    else [
                        torch.cat((held, new))
                        for held, new in zip(deepstack, features, strict=True)
                    ]
                )

        if position_ids is None:
            position_ids = (
                _mrope_position_ids(spans, seq, x.device)
                if spans
                else torch.arange(seq, device=x.device)[None]
            )
        cos, sin = _language_rope(position_ids, self.config.head_dim, self.config.rope_theta)

        for index, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin)
            # The deepstack features land on the FIRST len(features) decoder layers, one
            # each — not on the tower layers they were taken from.
            if deepstack is not None and mask is not None and index < len(deepstack):
                x[mask] = x[mask] + deepstack[index].to(x.dtype)
        return x
