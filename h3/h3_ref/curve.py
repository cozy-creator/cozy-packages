"""THE CURVE DELTA, and nothing else.

#531 splits H3 into two lanes. The CORRECTNESS lane is official full-AdaLN H3 on upstream
classes, and owns no architecture at all — `h3_ref/__init__.py` is a construction layer over
somebody else's modules. The OPTIMIZATION lane serves the community's pruned artifact, and
this file is the ENTIRETY of what that lane owns: three small modules and one install
function. If it grows a fourth idea it has stopped being a delta.

WHAT THE CURVE ACTUALLY IS (measured, from the banked headers — `h3-evidence`'s
`adaln-topologies` row, not from a publisher's description). The community carrier calls
itself "pruned", and the word is misleading: the per-block AdaLN projection is NOT removed.
Its INPUT COLLAPSES.

    full   `adaln_proj.linear.weight` is [96768, 2688]   and a `time_embedder` produces
           that 2688-wide timestep embedding from a sinusoidal projection
    curve  `adaln_proj.linear.weight` is [96768,    8]   with NO `time_embedder` at all,
           and one shared `adaln_t_table` F32 [1025, 8] in its place

That is 638 destinations against 532, and 66.3 GB against 40.2 GB — 26 GB of difference in
a single integer. The projections still EXECUTE; what changed is that their input is eight
coordinates read off a curve sampled on a fixed 1025-point grid instead of a learned
embedding of the timestep.

THE SEAM VERDICT: CLEAN INJECTION, no fork. Three upstream attributes are swapped, zero
upstream lines are copied, subclassed or monkeypatched, and the state-dict keys are
IDENTICAL to the ones the unmodified class would present, so the census the fill plane
performs is unchanged by the swap.

    transformer.time_proj      -> `nn.Identity`          (the timestep passes through)
    transformer.time_embedder  -> `CurveTimestepBasis`   (owns `adaln_t_table`)
    transformer.norm_out       -> `CurveNormOut`         (same `norm.weight`, `linear.*`)
    block.adaln_proj (x50)     -> `CurveModulation`      (same `linear.*`)

The last two exist for ONE reason. Upstream's modulation modules apply `silu` to `temb`
before their projection, which is right for a learned embedding and wrong for a curve — the
curve has already absorbed that activation, and `silu` is not invertible, so no choice of
returned `temb` can cancel it from outside. Swapping the module is therefore the minimal
correct edit; swapping it for one that carries the same parameter names is what keeps the
edit invisible to everything downstream.

WHY THIS IS NOT THE BAKED LANE. A baked-table artifact replaces the projection's OUTPUT and
needs the transformer's row indexing rewritten with it, because a cache spanning the whole
schedule is addressed differently from one step's distinct timesteps — v1 needed a
`with_kwargs` forward pre-hook for exactly that, and got a render that decoded to noise when
an early cut omitted it. The curve needs no such thing: `timestep_indices` keeps its
step-local meaning, the row layout is untouched, and every index in
`MiniMaxH3Transformer3DModel.forward` still means what upstream says it means.

STATE OF PROOF: BUILT. The topology is header-verified against the banked curve carrier —
`scripts/h3-diffusers-keys.py curve` checks all three shapes and reconciles the destination
count across the two key dialects. The INTERPOLATION RULE is not header-visible (the
evidence bank names that as an open item) and is taken from the same community runtime the
port read it from. Nothing here has run on a card, and the curve has no published
output-quality delta against full — #531 requires the optimization lane to prove quality and
economics independently before it serves anything.

MODULE SCOPE IS HEAVY ON PURPOSE, and `h3_ref/__init__.py` imports this file inside the
builder that needs it, exactly as `h3_arch/` does with its own architecture modules.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM
from torch import Tensor, nn

#: The released curve carrier's own two numbers, and the only two this file needs. They are
#: shapes in a banked header, not preferences: `adaln_t_table` is [1025, 8] and every
#: `adaln_proj.linear.weight` is [*, 8].
CURVE_GRID = 1025
CURVE_BASIS_DIM = 8


class CurveTimestepBasis(nn.Module):
    """The timestep basis: `(num_timesteps,) in [0, 1]` -> `(num_timesteps, basis_dim)`.

    Replaces `time_proj` and `time_embedder` as a pair. It holds `adaln_t_table` as a
    PERSISTENT buffer because the artifact carries that tensor and the fill plane needs a
    destination to put it in — a non-persistent buffer would be invisible to the census and
    the artifact's table would have nowhere to land.
    """

    def __init__(self, grid: int = CURVE_GRID, basis_dim: int = CURVE_BASIS_DIM) -> None:
        super().__init__()
        self.register_buffer("adaln_t_table", torch.empty(grid, basis_dim, dtype=torch.float32))

    def forward(self, timestep: Tensor) -> Tensor:
        table: Tensor = self.adaln_t_table
        # The grid is uniform over [0, 1] and the rule is linear interpolation between
        # neighbouring samples. The upper clamp keeps t = 1.0 on the LAST interval rather
        # than addressing one row past the end of the table.
        pos = timestep.to(torch.float32).flatten().clamp(0.0, 1.0) * (table.shape[0] - 1)
        low = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[low], table[low + 1], (pos - low).unsqueeze(1))


class CurveModulation(nn.Module):
    """One block's modulation, curve-fed. A drop-in for
    `MiniMaxH3AdaLayerNormModulation` with the same `linear.weight` / `linear.bias` keys, the
    same six-tuple return in the same `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
    gate_mlp` order, and the same `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]` row layout —
    minus the `silu` the curve already contains.
    """

    def __init__(self, basis_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(basis_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True)

    def forward(self, temb: Tensor) -> tuple[Tensor, ...]:
        projected = self.linear(temb.to(self.linear.weight.dtype))
        chunks: tuple[Tensor, ...] = projected.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)
        return chunks


class CurveNormOut(nn.Module):
    """The final norm, curve-fed. A drop-in for `MiniMaxH3AdaLayerNormOut` with the same
    `norm.weight` and `linear.*` keys and the same `shift`-then-`scale` projection halves."""

    def __init__(self, hidden_size: int, basis_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(basis_dim, 2 * hidden_size, bias=True)

    def forward(self, hidden_states: Tensor, temb: Tensor, timestep_indices: Tensor) -> Tensor:
        shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


def install(transformer: Any, *, grid: int = CURVE_GRID) -> Any:
    """Convert a constructed full-AdaLN transformer into the curve topology, IN PLACE.

    This must run BEFORE the census — the fill plane matches the graph's destinations
    against the artifact's keys, and a swap performed afterwards would present a topology
    nothing filled. `h3_ref.build_transformer` calls it at construction for exactly that
    reason.

    It refuses rather than guessing when the transformer was not built for a curve: the
    basis width comes from the transformer's OWN config (`time_embed_dim`), so a full-AdaLN
    config reaching here would silently build 2688-wide curve projections that match no
    released carrier.
    """
    config = transformer.config
    basis_dim = int(config.time_embed_dim)
    if basis_dim >= int(config.freq_dim):
        raise ValueError(
            f"time_embed_dim is {basis_dim}, which is the FULL-AdaLN embedding width and not "
            f"a curve basis: the released curve carrier's projections are [*, "
            f"{CURVE_BASIS_DIM}]. Construct the transformer with the curve config, or do not "
            "install the curve."
        )
    hidden_size = int(config.hidden_size)

    transformer.time_proj = nn.Identity()
    transformer.time_embedder = CurveTimestepBasis(grid, basis_dim)
    for block in transformer.transformer_blocks:
        block.adaln_proj = CurveModulation(basis_dim, hidden_size)
    transformer.norm_out = CurveNormOut(hidden_size, basis_dim, float(config.final_norm_eps))
    return transformer
