"""PDD-8 turbo overlay over the AdaLN-pruned MiniMax-H3 DiT.

Parallel Decoding Distillation (arXiv 2607.26004; `alibaba-pai/MiniMax-H3-Acc-LoRAs`) is a
rank-64 LoRA on seven linear families plus a bank of per-interval output heads, sampled on a
32-point grid four intervals at a time: eight transformer evaluations. On a pruned lane six of
the families are ordinary modules and apply here, as `y += (alpha/rank) * up(down(x))` over
the fp8 base; the seventh, `adaln_proj.linear`, has no module to attach to and is baked into
the turbo AdaLN tables by the producer. The head bank collapses offline to one head per
evaluation because the reference's per-step fusion is linear in the weights and its plan
depends only on the fixed grid.

The independent LoRA checkpoint has one component per task. Nothing here changes the
DiT's own parameters or its destination names: the base functions see the same construction
with the overlay absent from every forward they run. The endpoint leases the LoRA and
base separately and attaches the overlay only for the duration of one turbo sample.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from cozy_runtime.author import ConformanceError
from torch import nn
from torch.nn import functional as F

#: The forward-call selector: `attention_kwargs={ATTENTION_KWARG: TURBO_BANK}` names the bank
#: one DiT forward serves. Absent, the forward is the base's, whatever overlay is attached.
ATTENTION_KWARG = "cozy_h3_bank"
TURBO_BANK = "turbo"
#: The families applied at inference, as the checkpoint names them under a block.
LORA_FAMILIES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
#: The family the producer bakes into the turbo tables; refused on the overlay by name.
TABLED_FAMILY = "adaln_proj.linear"
#: Rows per low-rank chunk under an fp8 operand: bounds the dequantized transient the down
#: projection reads, never the accumulated output (which is written in place).
_LORA_ROW_CHUNK = 8192


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """The reference's ascending float64 grid `0 = t_0 < ... < t_N = 1` of one modality."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shift * sigma / (1 + (shift - 1) * sigma)


def pdd_head_plan(shift: float, num_steps: int, block_size: int) -> torch.Tensor:
    """`[evaluations, num_steps]`: row `k` weights the heads of block `k` by their share of
    the block's span, the reference's `pdd_sampling_plan` for every block at once."""
    if block_size < 1 or num_steps % block_size:
        raise ValueError(f"pdd_num_steps={num_steps} is not a multiple of block {block_size}")
    step_sizes = pdd_time_grid(shift, num_steps).diff()
    plan = torch.zeros(num_steps // block_size, num_steps, dtype=torch.float64)
    for index in range(num_steps // block_size):
        start = index * block_size
        span = step_sizes[start : start + block_size]
        plan[index, start : start + block_size] = span / span.sum()
    return plan


def collapse_head_bank(
    weight: torch.Tensor, bias: torch.Tensor, plan: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[N, out, in]` heads -> `[evaluations, out, in]`: the reference's per-step
    `einsum("pn,noi->poi", plan, W)` taken once, in float64, stored float32."""
    if weight.ndim != 3 or bias.shape != weight.shape[:2] or plan.shape[1] != weight.shape[0]:
        raise ValueError("head bank, bias and plan do not describe one N-interval head")
    fused = torch.einsum("pn,noi->poi", plan, weight.to(torch.float64))
    fused_bias = torch.einsum("pn,no->po", plan, bias.to(torch.float64))
    return fused.to(torch.float32), fused_bias.to(torch.float32)


class LoRAFactors(nn.Module):  # type: ignore[misc]
    """One site's rank-`r` factors; the update accumulates IN PLACE into the base output."""

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_down = nn.Parameter(torch.empty(rank, in_features, dtype=torch.bfloat16))
        self.lora_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch.bfloat16))
        self.scale = scale

    def accumulate(self, x: Any, out: torch.Tensor) -> None:
        """`out += scale * up(down(x))` without a second `[rows, out]` tensor: `down(x)` is
        `[rows, rank]` and the up projection lands through `addmm_`. `x` is the base site's
        operand — a tensor, or the fp8 row-quantized operand a fused epilogue hands a leaf."""
        if not out.is_contiguous():
            raise ValueError("a LoRA update accumulates into a contiguous base output only")
        flat = out.view(-1, out.shape[-1])
        up = self.lora_up.t().to(flat.dtype)
        for start, rows in _operand_rows(x):
            partial = F.linear(rows, self.lora_down.to(rows.dtype)).to(flat.dtype)
            flat[start : start + rows.shape[0]].addmm_(partial, up, alpha=self.scale)


def _operand_rows(x: Any) -> Iterator[tuple[int, torch.Tensor]]:
    """`(row offset, [rows, in] bf16)` chunks of one site's input, dequantizing an fp8
    row-quantized operand (`payload * scale` per row) chunk by chunk."""
    if isinstance(x, torch.Tensor):
        yield 0, x.reshape(-1, x.shape[-1])
        return
    payload, scale = x.payload, x.scale
    for start in range(0, int(payload.shape[0]), _LORA_ROW_CHUNK):
        chunk = payload[start : start + _LORA_ROW_CHUNK]
        rows = chunk.to(torch.float32).mul_(scale[start : start + _LORA_ROW_CHUNK])
        yield start, rows.to(torch.bfloat16)


class _LoRAHook:
    """Bound to one site for one forward; removed with the DiT's own forward hook."""

    __slots__ = ("factors",)

    def __init__(self, factors: LoRAFactors) -> None:
        self.factors = factors

    def __call__(self, module: nn.Module, args: tuple[Any, ...], out: torch.Tensor) -> None:
        del module
        self.factors.accumulate(args[0], out)


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


class TurboHeads(nn.Module):  # type: ignore[misc]
    """One collapsed head per transformer evaluation, selected by the step being served."""

    def __init__(self, evaluations: int, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(evaluations, out_features, in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(evaluations, out_features, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, step: int) -> torch.Tensor:
        return F.linear(hidden_states, self.weight[step], self.bias[step])


class TurboSchedule:
    """The exact float32 timesteps of the one turbo schedule, and the step each pair names."""

    def __init__(self, video_timesteps: Sequence[float], audio_timesteps: Sequence[float]) -> None:
        video, audio = tuple(video_timesteps), tuple(audio_timesteps)
        if len(video) != len(audio) or not video:
            raise ValueError("a turbo schedule pairs one audio timestep with every video one")
        # The step is read off the timesteps a forward presents, so the two grids may only
        # meet at the shared origin: any other shared value would make a step ambiguous.
        if set(video) & set(audio) != {0.0} or len(set(video)) != len(video):
            raise ValueError("turbo video and audio grids must share only the origin")
        self.video = video
        self.audio = audio
        self.evaluations = len(video)

    def step(self, timestep: torch.Tensor) -> int:
        present = set(timestep.detach().to(torch.float32).cpu().tolist())
        steps = [index for index, value in enumerate(self.video) if value in present]
        if len(steps) != 1 or self.audio[steps[0]] not in present:
            raise ConformanceError(
                f"timesteps {sorted(present)} do not name one turbo evaluation",
                code="artifact_config",
            )
        return steps[0]


class TurboOverlay(nn.Module):  # type: ignore[misc]
    """Everything one task's turbo function adds to its trunk DiT, under the DiT's own paths.

    State-dict keys are the checkpoint's: `transformer_blocks.{i}.attn.to_q.lora_down`,
    `token_refiner.refiner_blocks.{i}.ff.net.2.lora_up`, `transformer_blocks.{i}.adaln_proj.table`,
    `norm_out.table`, `proj_out.weight` — so a LoRA site's path here IS its path on the DiT.
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
        video_out: int,
        audio_out: int,
        rank: int,
        alpha: float,
        schedule: TurboSchedule,
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
        self.schedule = schedule
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
        self.proj_out = TurboHeads(schedule.evaluations, video_out, hidden_size)
        self.audio_proj_out = TurboHeads(schedule.evaluations, audio_out, hidden_size)

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        rank: int,
        alpha: float,
        schedule: TurboSchedule,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        block_table_dtype: torch.dtype,
        final_table_dtype: torch.dtype,
    ) -> TurboOverlay:
        patch = 1
        for value in config["patch_size"]:
            patch *= int(value)
        return cls(
            hidden_size=int(config["hidden_size"]),
            inner_dim=int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
            ffn_dim=int(config["ffn_dim"]),
            num_layers=int(config["num_layers"]),
            num_refiner_layers=int(config["num_refiner_layers"]),
            video_out=int(config["in_channels"]) * patch,
            audio_out=int(config["audio_in_channels"]),
            rank=rank,
            alpha=alpha,
            schedule=schedule,
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

    __slots__ = ("hooks", "overlay", "step")

    def __init__(self, overlay: TurboOverlay, step: int) -> None:
        self.overlay = overlay
        self.step = step
        self.hooks: list[Any] = []

    def attach(self, dit: Any) -> None:
        for path, factors in self.overlay.lora_sites():
            site = dit.get_submodule(path)
            if TABLED_FAMILY in path or not path.endswith(LORA_FAMILIES):
                raise ConformanceError(
                    f"turbo overlay names a LoRA site outside the six inference families: {path}",
                    code="artifact_config",
                )
            self.hooks.append(site.register_forward_hook(_LoRAHook(factors)))

    def detach(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
