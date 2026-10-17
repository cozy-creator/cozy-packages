"""lightx2v 8-step turbo overlay over the AdaLN-pruned MiniMax-H3 DiT.

lightx2v's MiniMax-H3 Turbo LoRAs (`lightx2v/Minimax-h3-Turbo`: fl2v 8-step v1.0, ref2v 8-step
v1.0 768p) are rank-128, alpha-8 adapters on six linear families of every block and token-refiner
block, sampled by the ordinary Euler loop at eight evaluations. They leave AdaLN and the heads
alone, so the overlay carries the base model's own AdaLN tables at the turbo plan's timesteps,
which the pruned base does not hold, and the base heads serve unchanged.

The overlay is one component per task beside the DiT it patches. Nothing here changes the
DiT's own parameters or its destination names: the base functions see the same construction
with the overlay absent from every forward they run.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from cozy_runtime.author import ConformanceError

#: The forward-call selector: `attention_kwargs={ATTENTION_KWARG: TURBO_BANK}` names the bank
#: one DiT forward serves. Absent, the forward is the base's, whatever overlay is attached.
ATTENTION_KWARG = "cozy_h3_bank"
OVERLAY_KWARG = "cozy_h3_overlay"
TURBO_BANK = "turbo"
#: The families applied at inference, as the checkpoint names them under a block.
LORA_FAMILIES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
#: Rows per LoRA GEMM. Fixed, so a row's update never depends on how many rows a rank holds
#: (Ulysses), and large enough that kernel launches stop mattering (TensorD #26: 256-row
#: chunks cost 7.6% of a dense step on an RTX PRO 6000, one GEMM pair per site 3.3%).
_LORA_ROW_CHUNK = 4096


class LoRAFactors(nn.Module):  # type: ignore[misc]
    """One site's rank-`r` factors over the original tensor operand."""

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_down = nn.Parameter(torch.empty(rank, in_features, dtype=torch.bfloat16))
        self.lora_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch.bfloat16))
        self.scale = scale

    def accumulate(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """`out += scale * (x @ down.T) @ up.T` in fixed row chunks: the up projection adds
        into the base output in place (one rounding, no `[rows, out]` update), and only the
        tail chunk, padded to the same row count, goes through a temporary."""
        if not isinstance(x, torch.Tensor):
            raise ValueError("a LoRA update requires the original tensor operand")
        if not out.is_contiguous():
            raise ValueError("a LoRA update accumulates into a contiguous base output only")
        flat = out.view(-1, out.shape[-1])
        inputs = x.reshape(-1, x.shape[-1])
        down = self.lora_down.to(inputs.dtype)
        up = self.lora_up.to(flat.dtype).t()
        total = int(inputs.shape[0])
        for start in range(0, total, _LORA_ROW_CHUNK):
            count = min(_LORA_ROW_CHUNK, total - start)
            # Row-major operands whatever the input's layout (a transposed view's rows are
            # strided): the update's GEMMs, and so its bits, never depend on how x is laid out.
            rows = inputs[start : start + count].contiguous()
            if count == _LORA_ROW_CHUNK:
                low = F.linear(rows, down).to(flat.dtype)
                flat[start : start + count].addmm_(low, up, alpha=self.scale)
                continue
            pad = _LORA_ROW_CHUNK - count
            low = F.linear(F.pad(rows, (0, 0, 0, pad)), down).to(flat.dtype)
            tail = F.pad(flat[start:], (0, 0, 0, pad)).addmm_(low, up, alpha=self.scale)
            flat[start:].copy_(tail[:count])


class _LoRAHook:
    """A permanent consumer hook, inactive for base forwards. The weight plane bakes the armed
    overlay's factors into the site's resident bytes (`weights.LoRASource`) and the hook then
    adds nothing; a base forward or another overlay finds those bytes refilled first."""

    __slots__ = ("_cozy_baked", "owner", "path")

    def __init__(self, owner: Any, path: str) -> None:
        self.owner = owner
        self.path = path
        self._cozy_baked = False

    def __call__(self, module: nn.Module, args: tuple[Any, ...], out: torch.Tensor) -> None:
        del module
        armed = self.owner._arming
        if armed is not None and not self._cozy_baked:
            armed.overlay.get_submodule(self.path).accumulate(args[0], out)

    def lora_factors(self) -> tuple[object, torch.Tensor, torch.Tensor, float] | None:
        """The armed overlay's factors for this site, or None in a base forward."""
        armed = self.owner._arming
        if armed is None:
            return None
        factors = armed.overlay.get_submodule(self.path)
        return factors, factors.lora_down, factors.lora_up, float(factors.scale)


class _Site(nn.Module):  # type: ignore[misc]
    """A named container whose children reproduce the checkpoint's module paths."""

    def __init__(self, **children: nn.Module) -> None:
        super().__init__()
        for name, child in children.items():
            self.add_module(name, child)


class TurboTables(nn.Module):  # type: ignore[misc]
    """Exact turbo rows for one family of tables: the same shape the base tables have."""

    def __init__(self, table: torch.Tensor, index: torch.Tensor | None = None) -> None:
        super().__init__()
        self.table = nn.Parameter(table)
        if index is not None:
            self.register_buffer("table_index", index, persistent=False)


class TurboOverlay(nn.Module):  # type: ignore[misc]
    """Everything one task's turbo function adds to its trunk DiT, under the DiT's own paths.

    State-dict keys are the checkpoint's: `transformer_blocks.{i}.attn.to_q.lora_down`,
    `token_refiner.refiner_blocks.{i}.ff.net.2.lora_up`, `transformer_blocks.{i}.adaln_proj.table`,
    `norm_out.table` — so a LoRA site's path here IS its path on the DiT.
    """

    present = True

    def __init__(
        self,
        *,
        hidden_size: int,
        inner_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_refiner_layers: int,
        rank: int,
        alpha: float,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        block_table_dtype: torch.dtype,
        final_table_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        scale = alpha / rank

        def factors(in_features: int, out_features: int) -> LoRAFactors:
            return LoRAFactors(in_features, out_features, rank, scale)

        def block_sites() -> dict[str, nn.Module]:
            return {
                "attn": _Site(
                    to_q=factors(hidden_size, inner_dim),
                    to_k=factors(hidden_size, inner_dim),
                    to_v=factors(hidden_size, inner_dim),
                    to_out=nn.ModuleList([factors(inner_dim, hidden_size)]),
                ),
                "ff": _Site(
                    net=nn.ModuleList(
                        [
                            _Site(proj=factors(hidden_size, 2 * ffn_dim)),
                            nn.Identity(),
                            factors(ffn_dim, hidden_size),
                        ]
                    )
                ),
            }

        index = torch.full((len(table_timesteps), 3), -1, dtype=torch.int64)
        for table_row, (timestep_row, modality_tag) in enumerate(table_block_keys):
            index[timestep_row, modality_tag] = table_row
        self.register_buffer(
            "timesteps", torch.tensor(tuple(table_timesteps), dtype=torch.float32), persistent=False
        )
        self.token_refiner = _Site(
            refiner_blocks=nn.ModuleList(
                [_Site(**block_sites()) for _ in range(num_refiner_layers)]
            )
        )
        self.transformer_blocks = nn.ModuleList(
            [
                _Site(
                    **block_sites(),
                    adaln_proj=TurboTables(
                        torch.empty(len(table_block_keys), 6, hidden_size, dtype=block_table_dtype),
                        index,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = TurboTables(
            torch.empty(len(table_timesteps), 2, hidden_size, dtype=final_table_dtype)
        )

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        rank: int,
        alpha: float,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        block_table_dtype: torch.dtype,
        final_table_dtype: torch.dtype,
    ) -> TurboOverlay:
        return cls(
            hidden_size=int(config["hidden_size"]),
            inner_dim=int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
            ffn_dim=int(config["ffn_dim"]),
            num_layers=int(config["num_layers"]),
            num_refiner_layers=int(config["num_refiner_layers"]),
            rank=rank,
            alpha=alpha,
            table_timesteps=table_timesteps,
            table_block_keys=table_block_keys,
            block_table_dtype=block_table_dtype,
            final_table_dtype=final_table_dtype,
        )

    def lora_sites(self) -> Iterator[tuple[str, LoRAFactors]]:
        for name, module in self.named_modules():
            if isinstance(module, LoRAFactors):
                yield name, module

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        """Global turbo table rows of exact float32 timesteps; off-plan values refuse."""
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        matches = values == self.timesteps.reshape(1, -1)
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            raise ConformanceError(
                "timestep rows are not covered exactly by the turbo plan: "
                f"{timestep.detach().float().cpu().tolist()}",
                code="artifact_config",
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class TurboArming:
    """What one DiT forward under the turbo bank reads: set by the DiT's pre-hook from the
    forward's own `attention_kwargs`, cleared by its forward hook, never request state."""

    __slots__ = ("overlay",)

    def __init__(self, overlay: TurboOverlay) -> None:
        self.overlay = overlay
