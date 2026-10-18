"""h3a-095 experiment: VSA (FastVideo FastH3's trained coarse gate) as H3's sparse attention.

Never merged. A request picks the method of its sparse steps (`SparseAttention`); the
`h3-sparse` producer (`producer.py`) serves Sol through the Runtime's own `sol-attn` and VSA
here, as Kablex/ComfyUI-Ref2VA-VSA does on ComfyUI:

- target video rows are tiled into 4x4x4 (t, h, w) cubes of 64 tokens, zero-padded at the
  edges (`block_len` holds each tile's live rows);
- every row before the target video (text, references, audio) stays dense, in tiles that never
  straddle two segments: those query tiles attend everything and those key tiles are always kept;
- each query tile keeps the top `1 - sparsity` of the key tiles exactly, with no pooled tail, and
  adds `gate * softmax(q_mean k_mean^T) v_mean`, the gate being FastH3's `to_gate_compress`
  projection of the block's input. Its 50 matrices are their own model (`H3VSAGates`), an
  as-is upload of `barelymining/ComfyUI-MiniMax-H3-FastVideo/fasth3_vsa_gate.safetensors`.

The kernel is Kitchen's chunked INT8 producer (`sol_attn_chunked`), as the Kitchen Sol producer
uses it: full Q/K/V are never materialized.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch
from torch import nn

from cozy_runtime.author import AttentionLayout, Loader, Model, sequence_parallel, uses_components
from cozy_runtime.author.kernels import PreQuantized

BLOCKS = 50
HIDDEN, INNER = 5376, 7168
TILE = 64
CUBE = 4
Method = Literal["sol", "vsa"]


@dataclass(frozen=True, slots=True)
class SparseAttention:
    """A request's sparse steps: the method, how many initial evaluations stay dense (None keeps
    the plan's), VSA's fraction of video key tiles dropped, and the gates VSA projects with."""

    method: Method = "sol"
    dense_steps: int | None = None
    sparsity: float = 0.75
    gates: Any = None


@dataclass(frozen=True, slots=True)
class Tiling:
    """One step's packed layout as VSA tiles it: prefix segment lengths, then the target grid."""

    segments: tuple[int, ...]
    grid: tuple[int, int, int]


_STEP: ContextVar[tuple[SparseAttention, Tiling] | None] = ContextVar("h3_vsa_step", default=None)


def active() -> tuple[SparseAttention, Tiling] | None:
    return _STEP.get()


def tiling(block_state: Any, layout: AttentionLayout, grid: tuple[int, int, int]) -> Tiling:
    """Segments are the runs of one row kind (text, reference video, audio) before the target video."""
    fields = block_state.denoiser_input_fields
    prefix = layout.protected_prefix
    kind = torch.full((prefix,), -1, dtype=torch.int8)
    kind[fields["text_indices"].cpu()] = 1
    kind[fields["video_indices"][: block_state.num_condition_video_rows].cpu()] = 0
    kind[fields["audio_indices"].cpu()] = 2
    if prefix and bool((kind < 0).any()):
        raise ValueError("VSA: a prefix row is neither text, reference video nor audio")
    cuts = (torch.nonzero(kind[1:] != kind[:-1]).flatten() + 1).tolist() if prefix else []
    bounds = [0, *cuts, prefix] if prefix else [0]
    if prefix + grid[0] * grid[1] * grid[2] != layout.live_tokens:
        raise ValueError(f"VSA: prefix {prefix} + grid {grid} != {layout.live_tokens} live rows")
    return Tiling(tuple(b - a for a, b in zip(bounds, bounds[1:])), grid)


@contextmanager
def serving(plan: SparseAttention, tiles: Tiling | None) -> Iterator[None]:
    token = _STEP.set((plan, tiles) if plan.method == "vsa" and tiles is not None else None)
    try:
        yield
    finally:
        _STEP.reset(token)


@dataclass(frozen=True, slots=True)
class Geometry:
    """Packed row -> tiled row (`untile`), tiled row -> packed row or -1 (`source`), each tile's
    live rows, and how many tiles are the dense prefix."""

    untile: torch.Tensor
    source: torch.Tensor
    block_len: torch.Tensor
    prefix_tiles: int
    rows: int


def geometry(tiles: Tiling, device: torch.device) -> Geometry:
    t, h, w = tiles.grid
    starts, lengths, at = [], [], 0
    for segment in tiles.segments:
        for offset in range(0, segment, TILE):
            starts.append(at + offset)
            lengths.append(min(TILE, segment - offset))
        at += segment
    prefix_rows = at
    prefix_tiles = len(lengths)
    # Packed prefix row -> tiled row: its segment tile's slot plus its offset in that tile.
    untile_prefix = torch.empty(prefix_rows, dtype=torch.long)
    for index, (start, length) in enumerate(zip(starts, lengths)):
        untile_prefix[start : start + length] = torch.arange(index * TILE, index * TILE + length)
    # Target video, frame-major then row-major: cube (t//4, h//4, w//4), live rows compacted.
    nt, nh, nw = (-(-n // CUBE) for n in (t, h, w))
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    tt, hh, ww = tt.reshape(-1), hh.reshape(-1), ww.reshape(-1)
    lt = (t - tt // CUBE * CUBE).clamp(max=CUBE)
    lh = (h - hh // CUBE * CUBE).clamp(max=CUBE)
    lw = (w - ww // CUBE * CUBE).clamp(max=CUBE)
    cube = (tt // CUBE) * nh * nw + (hh // CUBE) * nw + ww // CUBE
    rank = (tt % CUBE) * lh * lw + (hh % CUBE) * lw + ww % CUBE
    untile_video = (prefix_tiles + cube) * TILE + rank
    ct, ch, cw = torch.meshgrid(torch.arange(nt), torch.arange(nh), torch.arange(nw), indexing="ij")
    live = (
        (t - ct * CUBE).clamp(max=CUBE) * (h - ch * CUBE).clamp(max=CUBE) * (w - cw * CUBE).clamp(max=CUBE)
    ).reshape(-1)
    untile = torch.cat([untile_prefix, untile_video])
    tiles_total = prefix_tiles + nt * nh * nw
    source = torch.full((tiles_total * TILE,), -1, dtype=torch.long)
    source[untile] = torch.arange(untile.numel())
    block_len = torch.cat([torch.tensor(lengths, dtype=torch.long), live]).to(torch.int32)
    if int(block_len.sum()) != untile.numel() or int((source >= 0).sum()) != untile.numel():
        raise ValueError("VSA: tile accounting does not cover every live row exactly once")
    return Geometry(
        untile.to(device), source.to(device), block_len.to(device), prefix_tiles, tiles_total * TILE
    )


def gather(value: torch.Tensor | PreQuantized, source: torch.Tensor) -> torch.Tensor | PreQuantized:
    """Rows `source` of a batch-1 activation in tiled order; -1 rows are zero."""
    pad = source < 0
    index = source.clamp(min=0)
    if isinstance(value, PreQuantized):
        payload = value.payload.view(torch.uint8)[index]
        payload[pad] = 0
        scale = value.scale[index]
        return PreQuantized(payload.view(value.payload.dtype), scale, (1, source.numel(), value.shape[-1]))
    rows = value[0, index]
    rows[pad] = 0
    return rows.unsqueeze(0)


def dense_rows(value: torch.Tensor | PreQuantized) -> torch.Tensor:
    if isinstance(value, PreQuantized):
        return (value.payload.float() * value.scale).to(torch.bfloat16).view(value.shape)
    return value


@dataclass
class State:
    """The VSA half of one request's `h3-sparse` state on one GPU."""

    geometries: dict[Tiling, Geometry] = field(default_factory=dict)
    statistics: dict[tuple[object, ...], tuple[int, int, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    rope: tuple[torch.Tensor, Tiling, torch.Tensor] | None = None

    def close(self) -> None:
        backend = sys.modules.get("comfy_kitchen.backends.cuda")
        cached = getattr(backend, "_ROPE_FAB_CACHE", {})
        if self.rope is not None:
            for key, (reference, _packed) in list(cached.items()):
                if reference() is self.rope[2]:
                    cached.pop(key, None)
        self.geometries.clear()
        self.statistics.clear()
        self.rope = None


def attend(
    call: Any,
    hidden: torch.Tensor | PreQuantized,
    freqs: torch.Tensor,
    state: State,
    plan: SparseAttention,
    tiles: Tiling,
    project: Callable[[torch.Tensor | PreQuantized], torch.Tensor],
) -> torch.Tensor:
    """One VSA attention call: [1, T, heads, 128] in packed order. `freqs` is the packed rotary
    presentation; `project` returns one chunk's fused [M, 3*heads*128] q|k|v."""
    kitchen = sys.modules.get("comfy_kitchen")
    if kitchen is None or not hasattr(kitchen, "sol_attn_chunked"):
        raise RuntimeError("VSA needs Kitchen's sol_attn_chunked; comfy_kitchen is not loaded")
    attn, layout = call.module, call.layout
    gate = plan.gates.blocks[int(call.path.split(".")[1])].attn.to_gate_compress
    device = freqs.device
    geo = state.geometries.get(tiles)
    if geo is None:
        geo = state.geometries[tiles] = geometry(tiles, device)
    if state.rope is None or state.rope[0] is not freqs or state.rope[1] != tiles:
        tiled = freqs.new_zeros((geo.rows, *freqs.shape[1:]))
        tiled[geo.untile] = freqs
        state.rope = (freqs, tiles, tiled)
    tiled_freqs = state.rope[2]
    heads = attn.heads
    coarse = torch.empty((geo.rows, heads * 128), dtype=torch.bfloat16, device=device)
    filled = [False]

    def chunks() -> Iterator[torch.Tensor]:
        for start in range(0, geo.rows, 4096):
            stop = min(geo.rows, start + 4096)
            rows = gather(hidden, geo.source[start:stop])
            if not filled[0]:
                coarse[start:stop] = gate(dense_rows(rows)).reshape(stop - start, -1)
            yield project(rows)
        filled[0] = True

    key = (call.component, call.path)
    prior = state.statistics.pop(key, None)
    kmean, vscale = (prior[2], prior[3]) if prior is not None and prior[:2] == (layout.step - 1, geo.rows) else (None, None)
    output, kmean_next, vscale_next = kitchen.sol_attn_chunked(
        chunks,
        geo.rows,
        heads,
        tiled_freqs,
        (attn.norm_q.weight, attn.norm_k.weight),
        kmean=kmean,
        vscale=vscale,
        topk_ratio=1.0 - plan.sparsity,
        scale=128**-0.5,
        rope_eps=attn.norm_q.eps,
        sink_blocks=[0, geo.prefix_tiles],
        sink_q=[0, geo.prefix_tiles],
        tail=False,
        block_len=geo.block_len,
        coarse_gate=coarse.view(1, geo.rows, heads, 128),
        token_aug=0,
    )
    state.statistics[key] = (layout.step, geo.rows, kmean_next, vscale_next)
    del coarse
    return output[:, geo.untile]


class _Attention(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.to_gate_compress = nn.Linear(HIDDEN, INNER, bias=False, dtype=torch.bfloat16)


class _Block(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()


class GateStack(nn.Module):  # type: ignore[misc]
    """FastH3's 50 `blocks.N.attn.to_gate_compress` projections, under the checkpoint's names."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(BLOCKS))


class VSAGates:
    """The as-is upload's one component, `model`."""

    def __init__(self, config: Any) -> None:
        del config
        self.components = {"model": GateStack()}


@sequence_parallel(degrees=(2, 4, 7, 8))
class H3VSAGates(Model[VSAGates]):
    pipe: VSAGates

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(VSAGates, factory=VSAGates)

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("model")
    def sample(
        self,
        sample: Callable[..., Any],
        state: Any,
        *,
        attention: SparseAttention,
        on_step: Any,
        cancel: Any,
    ) -> Any:
        """`sample` with the gates in scope: every sparse step projects through them."""
        plan = replace(attention, gates=self.pipe.components["model"])
        return sample(state, on_step=on_step, cancel=cancel, attention=plan)
