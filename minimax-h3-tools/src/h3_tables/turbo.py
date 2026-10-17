"""Build the lightx2v 8-step turbo overlay: LoRA factors plus base AdaLN tables at its plans.

lightx2v's MiniMax-H3 Turbo LoRAs (`lightx2v/Minimax-h3-Turbo`) are rank-128, alpha-8 PEFT
adapters on the six attention and MLP families of every block and token-refiner block. They
leave AdaLN and the output heads untouched, so the overlay's tables are the base model's own
modulation rows at the turbo plan's timesteps, which the pruned base does not carry.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    Weights,
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

from ._memo import TURBO
from .adaln_operations import PLAIN, _asset, _sections, _validate_generators
from .kernel import H3Topology, precompute_tables, table_shapes
from .plans import Task, TimestepPlan, parse_plan
from .source import inspection
from .source import structures as _structures

RANK = 128
ALPHA = 8.0
DISTILLATION = "lightx2v"
#: fl2va: video shift 12 (fl2v 8-step v1.0); ref2va: video shift 6 (ref2v 8-step v1.0 768p).
PLAN_DIGESTS: dict[Task, str] = {
    "fl2va": "sha256:d2eb1605c1c1e01a4c5fdaaf1912ab43f33d9ce4772febf1c75b9bbfc8f7ba9d",
    "ref2va": "sha256:de82396318950cecce36718db968d96be07a662c69f33565f5989ae97bc524bc",
}
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")
_PEFT = {"lora_down": "lora_A.default.weight", "lora_up": "lora_B.default.weight"}


def turbo_plan(task: Task) -> TimestepPlan:
    plan = parse_plan(_asset(f"timestep-plan.{task}.turbo.json"), task=task, launch=False)
    if plan.digest != PLAN_DIGESTS[task]:
        raise UnsupportedInput("lightx2v tables require the reviewed eight-evaluation plan")
    return plan


def factor_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Overlay factor keys (`<site>.lora_down`/`.lora_up`) in constructor order."""
    hidden = int(config["hidden_size"])
    inner = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    ffn = int(config["ffn_dim"])
    families = {
        "attn.to_q": (hidden, inner),
        "attn.to_k": (hidden, inner),
        "attn.to_v": (hidden, inner),
        "attn.to_out.0": (inner, hidden),
        "ff.net.0.proj": (hidden, 2 * ffn),
        "ff.net.2": (ffn, hidden),
    }
    shapes: dict[str, tuple[int, ...]] = {}
    for prefix, count in (
        ("token_refiner.refiner_blocks", int(config["num_refiner_layers"])),
        ("transformer_blocks", int(config["num_layers"])),
    ):
        for index in range(count):
            for family, (inputs, outputs) in families.items():
                site = f"{prefix}.{index}.{family}"
                shapes[f"{site}.lora_down"] = (RANK, inputs)
                shapes[f"{site}.lora_up"] = (outputs, RANK)
    return shapes


def source_key(key: str) -> str:
    """The PEFT key lightx2v ships for one overlay factor key."""
    site, role = key.rsplit(".", 1)
    return f"{site}.{_PEFT[role]}"


def _adapter(structure: SourceInspection, task: Task, config: dict[str, Any]) -> str:
    """The checkpoint's one component, holding exactly this task's bf16 PEFT factors."""
    if len(structure.components) != 1:
        raise UnsupportedInput(
            f"the {task} adapter must be one component, got {sorted(structure.components)}",
            code="h3_turbo_adapter",
        )
    ((component, rows),) = structure.components.items()
    expected = {source_key(key): shape for key, shape in factor_shapes(config).items()}
    if set(rows) != set(expected):
        raise UnsupportedInput(
            f"the {task} adapter must hold exactly the {len(expected)} rank-{RANK} PEFT factors "
            f"of a lightx2v H3 Turbo LoRA ({len(rows)} rows)",
            code="h3_turbo_adapter",
        )
    for key, shape in expected.items():
        row = rows[key]
        if (row.logical_dtype, tuple(row.shape), row.encoding) != ("bf16", shape, PLAIN) or tuple(
            (role, part.dtype, tuple(part.shape)) for role, part in row.parts.items()
        ) != (("value", "bf16", shape),):
            raise UnsupportedInput(
                f"{task} adapter tensor {key} must be plain bf16 {shape}", code="h3_turbo_adapter"
            )
    return component


def _read(
    transaction: DerivedTransaction,
    component: str,
    key: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = torch.empty(shape, dtype=dtype, device="cpu")
    buffer = memoryview(value.view(torch.uint8).numpy()).cast("B")
    for offset in range(0, len(buffer), 32 << 20):
        transaction.source_read_into(
            "base", component, key, "value", offset, buffer[offset : offset + (32 << 20)]
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


@invocable(memo_version="h3-turbo-lora/2", memo_dependencies=TURBO)
async def turbo_lora(
    ctx: Context,
    *,
    fl2va: Weights[Any],
    ref2va: Weights[Any],
    base: Weights[Any],
    tel: Telemetry,
) -> ModelArtifact:
    """Build the lightx2v turbo overlay from its FL2VA and Ref2VA LoRAs.

    ``fl2va`` and ``ref2va`` are the `as-is` uploads of lightx2v's Diffusers PEFT files
    (`minimax_h3_fl2v_turbo_8step_v1.0_bf16`, `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16`).
    ``base`` is any full-precision H3 checkpoint; only its time embedder and modulation
    weights are read. Factors inherit the adapter objects under the overlay's keys; tables
    are computed and checkpointed per tensor, so an interrupted run resumes where it stopped.
    """
    sections = _sections()
    configs = {
        task: sections["transformer" if task == "fl2va" else "transformer_ref"] for task in TASKS
    }
    topologies = {task: H3Topology.from_config(configs[task]) for task in TASKS}
    structures = _structures(ctx, {"base": base, "fl2va": fl2va, "ref2va": ref2va})
    plans = {task: turbo_plan(task) for task in TASKS}
    adapters: dict[Task, str] = {}
    for task in TASKS:
        _validate_generators(structures["base"], task, topologies[task])
        adapters[task] = _adapter(structures[task], task, configs[task])
    targets: dict[str, Target] = {}
    order: list[tuple[str, str]] = []
    config: dict[str, Any] = {}
    for task in TASKS:
        component = f"{task}_turbo"
        factors = factor_shapes(configs[task])
        tables = table_shapes(topologies[task], plans[task])
        add = {
            key: Tensor(
                "bf16",
                shape,
                PLAIN,
                {"value": Part("bf16", shape, PartSource(task, adapters[task], source_key(key), "value"))},
            )
            for key, shape in factors.items()
        }
        add.update(
            {key: Tensor("bf16", shape, PLAIN, {"value": Part("bf16", shape)}) for key, shape in tables.items()}
        )
        targets[component] = Target(
            "base", f"{task}_dit", drop=tuple(structures["base"].components[f"{task}_dit"]), add=add
        )
        order.extend((component, key) for key in add)
        config[component] = {
            **configs[task],
            "cozy_h3": {
                "task": task,
                "modulation": "adaln-pruned",
                "distillation": DISTILLATION,
                "table_keys": plans[task].table_keys,
                "lora_rank": RANK,
                "lora_alpha": ALPHA,
            },
        }
    with ctx.output("bf16").open(
        Derivation(
            sources={name: structure.source for name, structure in structures.items()},
            targets=targets,
            configs={"model": Config("add")},
            order=tuple(order),
        )
    ) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        if ctx.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        reused = computed = 0
        for task in TASKS:
            component = f"{task}_turbo"
            completed = {
                key
                for owner, key, role in transaction.completed_parts()
                if owner == component and role == "value"
            }
            tables = table_shapes(topologies[task], plans[task]).keys()
            reused += len(tables & completed)
            computed += len(tables - completed)
            precompute_tables(
                plan=plans[task],
                topology=topologies[task],
                read=partial(_read, transaction, f"{task}_dit"),
                write=partial(_write, transaction, component),
                progress=partial(_progress, ctx, tel, task),
                device=torch.device(str(ctx.device)),
                completed=frozenset(completed & tables),
            )
        tel.metric("h3.turbo.reused_tensors", float(reused))
        tel.metric("h3.turbo.computed_tensors", float(computed))
        transaction.add_config("model", canonical_json.encode(config))
        return ctx.adopt_model(transaction.commit())
