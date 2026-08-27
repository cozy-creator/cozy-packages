"""Exact-timestep AdaLN tables for the official Diffusers MiniMax-H3 DiT.

The extension deliberately inherits Diffusers' forward unchanged. It replaces only the
time/AdaLN modules whose checkpoint-specific outputs were precomputed at one canonical
TimestepPlan before those dynamic modules were pruned. The remaining transformer graph,
including attention, feed-forward, RoPE,
input/output projections, and packed-row indexing, stays upstream code.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import torch
from cozy_runtime.author import ConformanceError
from diffusers import MiniMaxH3Transformer3DModel
from torch import nn


class _AdaLNPrunedTimestepLookup(nn.Module):  # type: ignore[misc]
    """Map exact float32 timestep values to the AdaLN-pruned plan's global rows."""

    def __init__(self, timesteps: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer(
            "timesteps", torch.tensor(tuple(timesteps), dtype=torch.float32), persistent=False
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.rows(timestep)

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        matches = values == self.timesteps.reshape(1, -1)
        counts = matches.sum(dim=1)
        if not bool(torch.all(counts == 1)):
            unknown = timestep.detach().float().cpu().tolist()
            raise ConformanceError(
                f"timestep rows are not covered exactly by the AdaLN-pruned plan: {unknown}",
                code="adaln_pruned_timestep",
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class _AdaLNPrunedTimestepRows(nn.Module):  # type: ignore[misc]
    """Keep inherited ``forward`` intact while replacing the removed timestep MLP."""

    def __init__(self) -> None:
        super().__init__()
        # The inherited Diffusers forward calls `get_parameter_dtype(time_embedder)` before
        # this identity. Its helper raises `UnboundLocalError` for a module with no tensor,
        # and a floating anchor would cast these exact integer row ids. This zero-length
        # int64 buffer is therefore the smallest state that preserves the upstream call.
        self.register_buffer("row_dtype", torch.empty(0, dtype=torch.int64), persistent=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


class _AdaLNPrunedBlockTable(nn.Module):  # type: ignore[misc]
    """Sparse exact table with the dense local shape Diffusers blocks already consume."""

    def __init__(
        self,
        *,
        hidden_size: int,
        timestep_count: int,
        keys: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.table = nn.Parameter(torch.empty(len(keys), 6, hidden_size))
        index = torch.full((timestep_count, 3), -1, dtype=torch.int64)
        for table_row, (timestep_row, modality_tag) in enumerate(keys):
            index[timestep_row, modality_tag] = table_row
        self.register_buffer("table_index", index, persistent=False)

    def forward(self, timestep_rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        indices = self.table_index.index_select(0, timestep_rows).reshape(-1)
        # The transformer's pre-hook proves every selected pair is present. Missing cells
        # are never observed by Diffusers' existing `adaln_indices`; row zero is only the
        # placeholder needed to preserve that dense local indexing convention.
        rows = self.table.index_select(0, indices.clamp_min(0))
        return cast(
            "tuple[torch.Tensor, ...]",
            rows.reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1),
        )


class _AdaLNPrunedOutputTable(nn.Module):  # type: ignore[misc]
    """Official final RMSNorm with exact precomputed shift/scale rows."""

    def __init__(self, norm: nn.Module, *, hidden_size: int, timestep_count: int) -> None:
        super().__init__()
        self.norm = norm
        self.table = nn.Parameter(torch.empty(timestep_count, 2, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_rows: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.table.index_select(0, timestep_rows).unbind(dim=1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class AdaLNPrunedMiniMaxH3Transformer(MiniMaxH3Transformer3DModel):  # type: ignore[misc]
    """Official MiniMax-H3 transformer with only its exact modulation source replaced."""

    _keep_in_fp32_modules: ClassVar[list[str]] = [
        *MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        "time_proj",
    ]

    def __init__(
        self,
        *,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.time_proj = _AdaLNPrunedTimestepLookup(table_timesteps)
        self.time_embedder = _AdaLNPrunedTimestepRows()
        for block in self.transformer_blocks:
            block.adaln_proj = _AdaLNPrunedBlockTable(
                hidden_size=int(self.config.hidden_size),
                timestep_count=len(table_timesteps),
                keys=table_block_keys,
            )
        norm_out = cast(Any, self.norm_out)  # type: ignore[has-type]
        self.norm_out = _AdaLNPrunedOutputTable(
            norm_out.norm,
            hidden_size=int(self.config.hidden_size),
            timestep_count=len(table_timesteps),
        )
        self.register_forward_pre_hook(self._validate_table_rows, with_kwargs=True)

    def _validate_table_rows(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module
        names = (
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "timestep",
            "timestep_indices",
            "token_tags",
        )
        values = {**dict(zip(names, args, strict=False)), **kwargs}
        timestep = values.get("timestep")
        timestep_indices = values.get("timestep_indices")
        token_tags = values.get("token_tags")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (timestep, timestep_indices, token_tags)
        ):
            return  # Diffusers' inherited forward owns its ordinary argument diagnostics.
        assert isinstance(timestep, torch.Tensor)
        assert isinstance(timestep_indices, torch.Tensor)
        assert isinstance(token_tags, torch.Tensor)
        if bool(torch.any((token_tags < 0) | (token_tags > 2))):
            raise ConformanceError(
                "packed rows contain a modality tag outside the canonical 0/1/2 set",
                code="adaln_pruned_timestep",
            )
        global_rows = self.time_proj.rows(timestep)
        row_timesteps = global_rows.index_select(0, timestep_indices)
        table_index = self.transformer_blocks[0].adaln_proj.table_index
        present = table_index[row_timesteps, token_tags] >= 0
        if not bool(torch.all(present)):
            raise ConformanceError(
                "packed rows request a timestep/modality pair absent from the AdaLN-pruned plan",
                code="adaln_pruned_timestep",
            )

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
    ) -> AdaLNPrunedMiniMaxH3Transformer:
        """Construct from the same upstream config accepted by the official class."""
        parameters = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
        kwargs = {name: value for name, value in config.items() if name in parameters}
        return cls(
            table_timesteps=table_timesteps,
            table_block_keys=table_block_keys,
            **kwargs,
        )
