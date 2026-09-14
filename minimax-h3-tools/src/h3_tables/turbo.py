"""Build a standalone PDD-8 adapter with LoRA factors, modulation tables and heads."""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    canonical_json,
    invocable,
)
from tensorfs.derived import (
    Config,
    Derivation,
    DerivedTransaction,
    Part,
    PartSource,
    SourceInspection,
    Target,
    Tensor,
)

from .adaln_operations import (
    PLAIN,
    _asset,
    _sections,
    _validate_generators,
)
from .kernel import H3Topology, LowRankAdapter, adapter_shapes, precompute_tables, table_shapes
from .plans import Task, TimestepPlan, parse_plan
from .source import H3FullTransformer, inspection

RANK = 64
NUM_STEPS = 32
BLOCK_SIZE = 4
PLAN_DIGESTS: dict[Task, str] = {
    "fl2va": "sha256:d2eb1605c1c1e01a4c5fdaaf1912ab43f33d9ce4772febf1c75b9bbfc8f7ba9d",
    "ref2va": "sha256:896a10881805e3047f6df514dd4e70fcea078837469eeb22ce67f0bd3f679cb7",
}
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")


def turbo_plan(task: Task) -> TimestepPlan:
    plan = parse_plan(_asset(f"timestep-plan.{task}.turbo.json"), task=task, launch=False)
    if plan.digest != PLAN_DIGESTS[task]:
        raise UnsupportedInput("PDD tables require the captured eight-evaluation plan")
    return plan


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shift * sigma / (1 + (shift - 1) * sigma)


def pdd_head_plan(shift: float, num_steps: int, block_size: int) -> torch.Tensor:
    if num_steps < 1 or block_size < 1 or num_steps % block_size:
        raise ValueError("the PDD head grid must divide into complete blocks")
    sizes = pdd_time_grid(shift, num_steps).diff()
    plan = torch.zeros(num_steps // block_size, num_steps, dtype=torch.float64)
    for index in range(num_steps // block_size):
        span = sizes[index * block_size : (index + 1) * block_size]
        plan[index, index * block_size : (index + 1) * block_size] = span / span.sum()
    return plan


def collapse_head_bank(
    weight: torch.Tensor, bias: torch.Tensor, plan: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the upstream float32 head fusion, including its rounding order."""
    if (
        weight.ndim != 3
        or bias.shape != weight.shape[:2]
        or plan.ndim != 2
        or plan.shape[1] != weight.shape[0]
    ):
        raise ValueError("head bank, bias and plan do not describe one N-interval head")
    # Upstream converts its float64 grid coefficients to the head's float32 dtype
    # before einsum. Float64 accumulation followed by casting changes the result.
    plan = plan.to(device=weight.device, dtype=torch.float32)
    return (
        torch.einsum("pn,noi->poi", plan, weight.float()),
        torch.einsum("pn,no->po", plan, bias.float()),
    )


def overlay_shapes(
    config: dict[str, Any], plan: TimestepPlan
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Constructor order of the overlay, with all six inference LoRA families."""
    hidden, inner, ffn = (
        int(config["hidden_size"]),
        int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
        int(config["ffn_dim"]),
    )
    shapes: dict[str, tuple[str, tuple[int, ...]]] = {}
    families = {
        "attn.to_q": (hidden, inner),
        "attn.to_k": (hidden, inner),
        "attn.to_v": (hidden, inner),
        "attn.to_out.0": (inner, hidden),
        "ff.net.0.proj": (hidden, 2 * ffn),
        "ff.net.2": (ffn, hidden),
    }
    for prefix, count in (
        ("token_refiner.refiner_blocks", int(config["num_refiner_layers"])),
        ("transformer_blocks", int(config["num_layers"])),
    ):
        for index in range(count):
            for family, (inputs, outputs) in families.items():
                key = f"{prefix}.{index}.{family}"
                shapes[f"{key}.lora_down"] = ("bf16", (RANK, inputs))
                shapes[f"{key}.lora_up"] = ("bf16", (outputs, RANK))
            if prefix == "transformer_blocks":
                shapes[f"{prefix}.{index}.adaln_proj.table"] = (
                    "bf16",
                    (len(plan.block_rows), 6, hidden),
                )
    shapes["norm_out.table"] = ("bf16", (len(plan.timesteps), 2, hidden))
    for prefix, outputs in (
        ("proj_out", int(config["in_channels"]) * math.prod(config["patch_size"])),
        ("audio_proj_out", int(config["audio_in_channels"])),
    ):
        shapes[f"{prefix}.weight"] = ("f32", (NUM_STEPS // BLOCK_SIZE, outputs, hidden))
        shapes[f"{prefix}.bias"] = ("f32", (NUM_STEPS // BLOCK_SIZE, outputs))
    return shapes


def _adapter_component(
    structure: SourceInspection, config: dict[str, Any], plan: TimestepPlan, topology: H3Topology
) -> str:
    expected = {
        key: ("bf16", ((NUM_STEPS, *shape[1:]) if dtype == "f32" else shape))
        for key, (dtype, shape) in overlay_shapes(config, plan).items()
        if not key.endswith(".table")
    }
    expected.update(
        {key: ("bf16", shape) for key, (_, shape) in adapter_shapes(topology, RANK).items()}
    )
    components = set(structure.components)
    rows = next(iter(structure.components.values())) if len(components) == 1 else {}
    if len(components) != 1 or set(rows) != set(expected):
        raise UnsupportedInput("PDD adapter must contain exactly its LoRA factors and 32 heads")
    for key, (dtype, shape) in expected.items():
        row = rows[key]
        if (row.logical_dtype, row.shape, row.encoding) != (dtype, shape, PLAIN) or tuple(
            (role, p.dtype, p.shape) for role, p in row.parts.items()
        ) != (("value", dtype, shape),):
            raise UnsupportedInput(f"PDD adapter tensor {key} must be plain {dtype} {shape}")
    return components.pop()


def _read(
    transaction: DerivedTransaction,
    source: str,
    component: str,
    key: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = torch.empty(shape, dtype=dtype, device="cpu")
    buffer = memoryview(value.view(torch.uint8).numpy()).cast("B")
    for offset in range(0, len(buffer), 32 << 20):
        transaction.source_read_into(
            source, component, key, "value", offset, buffer[offset : offset + (32 << 20)]
        )
    return value


def _write(transaction: DerivedTransaction, component: str, key: str, value: torch.Tensor) -> None:
    transaction.add_part(
        component, key, "value", value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    )
    transaction.checkpoint()


def _progress(ctx: Context, tel: Telemetry, task: Task, done: int, total: int) -> None:
    ctx.raise_if_cancelled()
    tel.progress(done / total, stage=f"turbo-{task}")


def _produce(
    ctx: Context,
    tel: Telemetry,
    *,
    full: H3FullTransformer,
    adapters: dict[Task, H3FullTransformer],
    configs: dict[str, Any],
    topologies: dict[Task, H3Topology],
) -> ModelArtifact:
    """One adapter transaction; LoRA factors inherit their exact source objects."""
    source_models: dict[str, H3FullTransformer] = {"full": full}
    source_models.update({str(task): model for task, model in adapters.items()})
    structures = {name: inspection(ctx, model) for name, model in source_models.items()}
    plans = {task: turbo_plan(task) for task in TASKS}
    components = {
        task: _adapter_component(
            structures[task], configs[f"{task}_dit"], plans[task], topologies[task]
        )
        for task in TASKS
    }
    targets: dict[str, Target] = {}
    output_order: list[tuple[str, str]] = []
    for task in TASKS:
        component = f"{task}_turbo"
        specs = overlay_shapes(configs[f"{task}_dit"], plans[task])
        targets[component] = Target(
            "full",
            f"{task}_dit",
            drop=tuple(structures["full"].components[f"{task}_dit"]),
            add={
                key: Tensor(
                    dtype,
                    shape,
                    PLAIN,
                    {
                        "value": Part(
                            dtype,
                            shape,
                            source=(
                                PartSource(task, components[task], key, "value")
                                if key.endswith((".lora_down", ".lora_up"))
                                else None
                            ),
                        )
                    },
                )
                for key, (dtype, shape) in specs.items()
            },
        )
        output_order.extend((component, key) for key in specs)
    config = {}
    for task in TASKS:
        config[f"{task}_turbo"] = {
            **configs[f"{task}_dit"],
            "cozy_h3": {
                "task": task,
                "modulation": "adaln-pruned",
                "distillation": "pdd",
                "table_keys": plans[task].table_keys,
                "lora_rank": RANK,
                "lora_alpha": float(RANK),
                "pdd_num_steps": NUM_STEPS,
                "pdd_block_size": BLOCK_SIZE,
            },
        }
    raw = canonical_json.encode(config)
    with ctx.output("model").open(
        Derivation(
            sources={name: structure.source for name, structure in structures.items()},
            targets=targets,
            configs={"model": Config("add")},
            order=tuple(output_order),
        )
    ) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        if ctx.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        for task in TASKS:
            ctx.raise_if_cancelled()
            component = f"{task}_turbo"
            completed = {
                key
                for owner, key, role in transaction.completed_parts()
                if owner == component and role == "value"
            }
            tables = table_shapes(topologies[task], plans[task])
            reader = partial(_read, transaction, task, components[task])
            precompute_tables(
                plan=plans[task],
                topology=topologies[task],
                read=partial(_read, transaction, "full", f"{task}_dit"),
                write=partial(_write, transaction, component),
                progress=partial(_progress, ctx, tel, task),
                device=torch.device(str(ctx.device)),
                completed=frozenset(completed & tables.keys()),
                adapter=LowRankAdapter(RANK, 1.0, reader),
            )
            for prefix, shift in (("proj_out", 12.0), ("audio_proj_out", 3.0)):
                ctx.raise_if_cancelled()
                if {f"{prefix}.weight", f"{prefix}.bias"} <= completed:
                    continue
                out_shape = targets[component].add[f"{prefix}.weight"].shape
                weight = reader(f"{prefix}.weight", torch.bfloat16, (NUM_STEPS, *out_shape[1:]))
                bias = reader(f"{prefix}.bias", torch.bfloat16, (NUM_STEPS, out_shape[1]))
                fused = collapse_head_bank(
                    weight, bias, pdd_head_plan(shift, NUM_STEPS, BLOCK_SIZE)
                )
                for suffix, value in zip(("weight", "bias"), fused, strict=True):
                    if f"{prefix}.{suffix}" not in completed:
                        _write(transaction, component, f"{prefix}.{suffix}", value)
        transaction.add_config("model", raw)
        return ctx.adopt_model(transaction.commit())


def build_turbo_adapter(
    ctx: Context,
    *,
    full: H3FullTransformer,
    fl2va_adapter: H3FullTransformer,
    ref2va_adapter: H3FullTransformer,
    tel: Telemetry,
) -> ModelArtifact:
    """Prepare a PDD-8 adapter independently of any quantized base checkpoint."""
    sections = _sections()
    configs = {
        f"{task}_dit": sections["transformer" if task == "fl2va" else "transformer_ref"]
        for task in TASKS
    }
    topologies = {
        task: H3Topology.from_config(
            sections["transformer" if task == "fl2va" else "transformer_ref"]
        )
        for task in TASKS
    }
    for task in TASKS:
        _validate_generators(inspection(ctx, full), task, topologies[task])
    return _produce(
        ctx,
        tel,
        full=full,
        adapters={"fl2va": fl2va_adapter, "ref2va": ref2va_adapter},
        configs=configs,
        topologies=topologies,
    )


@invocable(memoize=True)
async def prepare_turbo(
    ctx: Context,
    *,
    full: H3FullTransformer,
    fl2va_adapter: H3FullTransformer,
    ref2va_adapter: H3FullTransformer,
    tel: Telemetry,
) -> ModelArtifact:
    return build_turbo_adapter(
        ctx,
        full=full,
        fl2va_adapter=fl2va_adapter,
        ref2va_adapter=ref2va_adapter,
        tel=tel,
    )
