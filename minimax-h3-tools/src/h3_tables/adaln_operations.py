"""Independent H3 generating-weight projections, table banks and native attachment."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import msgspec
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

from ._table_layout import TableLayout
from .kernel import H3Topology, precompute_tables, removed_keys, source_shapes, table_shapes
from .model_config import dual_adaln_pruned_config, dual_full_config, parse_production_config
from .order import current_order, full_order
from .plans import Task, TimestepPlan, parse_declared_plan
from .source import TARGET_COMPONENT, H3FullTransformer, inspection

PLAIN = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
_SOURCE_SECTION = {"fl2va": "transformer", "ref2va": "transformer_ref"}
_BINDING = "generating_projection_digest"
_METADATA = "adaln"
_NUMERICS = "h3-fp32-time-bf16-adaln/1"
_TASKS: tuple[Task, ...] = ("fl2va", "ref2va")


class Selection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    projection: ModelArtifact | None
    ready: bool


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


def _plan(task: Task) -> TimestepPlan:
    return parse_declared_plan(_asset(f"timestep-plan.{task}.json"))


def _sections() -> dict[str, dict[str, Any]]:
    return parse_production_config(_asset("model-config.json"))


def _topology(task: Task) -> H3Topology:
    return H3Topology.from_config(_sections()[_SOURCE_SECTION[task]])


def _projection_config(task: Task, topology: H3Topology | None = None) -> bytes:
    topology = topology or _topology(task)
    return canonical_json.encode(
        {
            "schema": "h3-adaln-weights/1",
            "task": task,
            "topology": {
                name: getattr(topology, name)
                for name in (
                    "hidden_size",
                    "num_layers",
                    "freq_dim",
                    "time_embed_hidden_dim",
                    "time_embed_dim",
                )
            },
        }
    )


def _body_kind(structure: SourceInspection) -> bool:
    """Return whether the exact assembled source is already pruned."""
    current = current_order(_asset("whole-order.json")).rows
    actual = {(component, key) for component, rows in structure.components.items() for key in rows}
    if actual == set(current):
        return True
    if actual == set(full_order(_sections(), current)):
        return False
    raise UnsupportedInput(
        "AdaLN preparation requires one assembled full or pruned H3 model", code="adaln_source"
    )


def _bindings(structure: SourceInspection) -> dict[str, str]:
    value = canonical_json.decode(structure.configs["model"])
    if not isinstance(value, dict):
        raise UnsupportedInput("pruned H3 model has no config bindings", code="adaln_binding")
    result: dict[str, str] = {}
    for task, component in TARGET_COMPONENT.items():
        row = value.get(component)
        stamp = row.get("cozy_h3") if isinstance(row, dict) else None
        digest = stamp.get(_BINDING) if isinstance(stamp, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in digest[7:])
        ):
            raise UnsupportedInput(
                "H3 generating-weight binding is not an exact digest", code="adaln_binding"
            )
        result[task] = digest
    return result


def _validate_body_config(structure: SourceInspection, pruned: bool) -> dict[str, dict[str, Any]]:
    if set(structure.configs) != {"model"}:
        raise UnsupportedInput(
            "H3 body requires its single construction config", code="adaln_config"
        )
    raw = structure.configs["model"]
    if not pruned:
        if raw != dual_full_config(_sections()):
            raise UnsupportedInput(
                "H3 full-body config differs from its supported constructor", code="adaln_config"
            )
        return {}
    value = canonical_json.decode(raw)
    expected = canonical_json.decode(_asset("model-config.json"))
    if not isinstance(value, dict) or set(value) != set(expected):
        raise UnsupportedInput(
            "H3 pruned config has an unexpected component set", code="adaln_config"
        )
    layouts: dict[str, dict[str, Any]] = {}
    for task, component in TARGET_COMPONENT.items():
        row = value.get(component)
        if not isinstance(row, dict):
            raise UnsupportedInput("H3 pruned component config is absent", code="adaln_config")
        stamp = row.get("cozy_h3")
        if (
            not isinstance(stamp, dict)
            or set(stamp)
            not in (
                {"task", "modulation", "table_keys"},
                {"task", "modulation", "table_keys", _BINDING},
            )
            or stamp["task"] != task
            or stamp["modulation"] != "adaln-pruned"
            or not isinstance(stamp["table_keys"], dict)
        ):
            raise UnsupportedInput(
                "H3 pruned config has no task/table metadata", code="adaln_config"
            )
        layouts[task] = stamp["table_keys"]
        TableLayout.parse(layouts[task])
        # A previous supported bank may be replaced while keeping constructor facts.
        row["cozy_h3"] = expected[component]["cozy_h3"]
    if value != expected:
        raise UnsupportedInput("H3 pruned body changed its constructor facts", code="adaln_config")
    return layouts


def _validate_generators(
    structure: SourceInspection, task: Task, topology: H3Topology | None = None
) -> None:
    tensors = {
        (component, key): tensor
        for component, rows in structure.components.items()
        for key, tensor in rows.items()
    }
    for key, (dtype, shape) in source_shapes(topology or _topology(task)).items():
        tensor = tensors.get((TARGET_COMPONENT[task], key))
        expected = "f32" if dtype == torch.float32 else "bf16"
        if (
            tensor is None
            or tensor.encoding != PLAIN
            or tensor.logical_dtype != expected
            or tensor.shape != shape
            or tensor.parts != {"value": Part(expected, shape)}
        ):
            raise UnsupportedInput(
                f"H3 generating weight {key} must preserve its exact plain dtype and shape",
                code="adaln_generating_weights",
            )


def _project(
    ctx: Context,
    source: H3FullTransformer,
    task: Task,
    output: str,
    topology: H3Topology | None = None,
) -> ModelArtifact:
    structure = inspection(ctx, source)
    topology = topology or _topology(task)
    _validate_generators(structure, task, topology)
    component = TARGET_COMPONENT[task]
    selected = removed_keys(topology)
    present = tuple(structure.components[component])
    config = _projection_config(task, topology)
    with ctx.output(output).open(
        Derivation(
            sources={"source": structure.source},
            targets={
                component: Target(
                    "source", component, drop=tuple(key for key in present if key not in selected)
                )
            },
            configs={_METADATA: Config("add")},
            order=tuple((component, key) for key in selected),
        )
    ) as transaction:
        if transaction.receipt is None:
            transaction.add_config(_METADATA, config)
        return ctx.adopt_model(transaction.commit())


@invocable(memoize=True)
async def select_adaln_weights(ctx: Context, *, source: H3FullTransformer, task: Task) -> Selection:
    ctx.raise_if_cancelled()
    structure = inspection(ctx, source)
    pruned = _body_kind(structure)
    layouts = _validate_body_config(structure, pruned)
    if pruned:
        _bindings(structure)
        if layouts[task] != _plan(task).table_keys:
            raise UnsupportedInput(
                "retabling a previous schedule requires the full generating_model",
                code="adaln_generating_weights",
            )
        _validate_table_rows(structure, task, bank=False)
        with ctx.output("model").open(
            Derivation(
                sources={"source": structure.source},
                targets={
                    component: Target("source", component) for component in structure.components
                },
                configs={"model": Config("copy", "source", "model")},
                order=current_order(_asset("whole-order.json")).rows,
            )
        ) as transaction:
            ready = ctx.adopt_model(transaction.commit())
        return Selection(ready, True)
    return Selection(_project(ctx, source, task, "model"), False)


def _bank_config(task: Task, projection: str) -> bytes:
    return canonical_json.encode(
        {
            "schema": "h3-adaln-bank/1",
            "task": task,
            "projection": projection,
            "table_keys": _plan(task).table_keys,
            "dtype": "bf16",
            "numerics": _NUMERICS,
        }
    )


def _read_weight(
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
            "source", component, key, "value", offset, buffer[offset : offset + (32 << 20)]
        )
    return value


def _compute_into(
    ctx: Context,
    tel: Telemetry,
    transaction: DerivedTransaction,
    task: Task,
    plan: TimestepPlan,
    topology: H3Topology,
) -> None:
    component = TARGET_COMPONENT[task]
    completed = frozenset(
        key
        for selected, key, role in transaction.completed_parts()
        if selected == component and role == "value"
    )

    source_bytes = 0

    def read(key: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal source_bytes
        ctx.raise_if_cancelled()
        value = _read_weight(transaction, component, key, dtype, shape)
        source_bytes += value.numel() * value.element_size()
        return value

    def write(key: str, value: torch.Tensor) -> None:
        ctx.raise_if_cancelled()
        raw = value.detach().to(device="cpu").contiguous().view(torch.uint16).numpy().tobytes()
        transaction.add_part(component, key, "value", raw)
        transaction.checkpoint()

    def progress(done: int, total: int) -> None:
        ctx.raise_if_cancelled()
        tel.progress(done / total, stage=f"timestep-table-{task}")

    if ctx.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    precompute_tables(
        plan=plan,
        topology=topology,
        read=read,
        write=write,
        progress=progress,
        device=torch.device(str(ctx.device)),
        completed=completed,
    )

    tel.metric("h3.adaln.source_bytes", float(source_bytes), unit="bytes")
    tel.metric("h3.adaln.reused_tables", float(len(completed)), unit="tables")


@invocable(memoize=True)
async def compute_adaln_tables(
    ctx: Context,
    *,
    source: H3FullTransformer,
    task: Task,
    plan_digest: str,
    tel: Telemetry,
) -> ModelArtifact:
    plan = _plan(task)
    if plan_digest != plan.digest:
        raise UnsupportedInput(
            "AdaLN plan differs from the captured approved package plan", code="adaln_plan"
        )
    structure = inspection(ctx, source)
    _validate_generators(structure, task)
    component = TARGET_COMPONENT[task]
    if {(component, key) for component, rows in structure.components.items() for key in rows} != {
        (component, key) for key in removed_keys(_topology(task))
    } or structure.configs[_METADATA] != _projection_config(task):
        raise UnsupportedInput(
            "AdaLN bank input must be one minimal generating-weight projection",
            code="adaln_projection",
        )
    topology = _topology(task)
    tables = table_shapes(topology, plan)
    metadata = _bank_config(task, source.checkpoint_ref)
    with ctx.output("model").open(
        Derivation(
            sources={"source": structure.source},
            targets={
                component: Target(
                    "source",
                    component,
                    drop=removed_keys(topology),
                    add={
                        key: Tensor("bf16", shape, PLAIN, {"value": Part("bf16", shape)})
                        for key, shape in tables.items()
                    },
                )
            },
            configs={_METADATA: Config("add")},
            order=tuple((component, key) for key in tables),
        )
    ) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        _compute_into(ctx, tel, transaction, task, plan, topology)
        transaction.add_config(_METADATA, metadata)
        return ctx.adopt_model(transaction.commit())


def _bank_projection(
    ctx: Context, bank: H3FullTransformer, task: Task, topology: H3Topology | None = None
) -> str:
    structure = inspection(ctx, bank)
    value = canonical_json.decode(structure.configs[_METADATA])
    if not isinstance(value, dict) or not isinstance(value.get("projection"), str):
        raise UnsupportedInput("AdaLN bank has no generating-weight binding", code="adaln_binding")
    projection = str(value["projection"])
    if value != canonical_json.decode(_bank_config(task, projection)):
        raise UnsupportedInput(
            "AdaLN bank task, schedule or numerical contract differs", code="adaln_binding"
        )
    _validate_table_rows(structure, task, bank=True, topology=topology)
    return projection


def _validate_table_rows(
    structure: SourceInspection, task: Task, *, bank: bool, topology: H3Topology | None = None
) -> None:
    component = TARGET_COMPONENT[task]
    tables = table_shapes(topology or _topology(task), _plan(task))
    rows = {
        key: tensor
        for key, tensor in structure.components.get(component, {}).items()
        if bank or key in tables
    }
    if bank and set(structure.components) != {component}:
        raise UnsupportedInput(
            "AdaLN bank includes another task or unrelated tensors", code="adaln_bank"
        )
    if set(rows) != set(tables):
        raise UnsupportedInput("AdaLN bank has an unexpected table set", code="adaln_bank")
    for key, tensor in rows.items():
        shape = tables[key]
        if (
            tensor.encoding != PLAIN
            or tensor.logical_dtype != "bf16"
            or tensor.shape != shape
            or tensor.parts != {"value": Part("bf16", shape)}
        ):
            raise UnsupportedInput(
                "AdaLN bank has changed table geometry or encoding", code="adaln_bank"
            )


def _require_bank_binding(
    ctx: Context,
    bank: H3FullTransformer,
    task: Task,
    projection: str,
    topology: H3Topology | None = None,
) -> None:
    if _bank_projection(ctx, bank, task, topology) != projection:
        raise UnsupportedInput(
            "AdaLN bank was computed from different generating weights", code="adaln_binding"
        )


def _apply_adaln(
    ctx: Context,
    *,
    source: H3FullTransformer,
    fl2va: H3FullTransformer,
    ref2va: H3FullTransformer,
    require_pruned: bool,
) -> ModelArtifact:
    ctx.raise_if_cancelled()
    structure = inspection(ctx, source)
    pruned = _body_kind(structure)
    if pruned != require_pruned:
        raise UnsupportedInput(
            "AdaLN attachment source has a different pruning state", code="adaln_source"
        )
    _validate_body_config(structure, pruned)
    bindings = (
        _bindings(structure)
        if pruned
        else {
            task: _project(ctx, source, task, f"{task}-weights").manifest.digest for task in _TASKS
        }
    )
    banks = {"fl2va": fl2va, "ref2va": ref2va}
    targets = {component: Target("source", component) for component in structure.components}
    for task in _TASKS:
        _require_bank_binding(ctx, banks[task], task, bindings[task])
        component = TARGET_COMPONENT[task]
        tables = table_shapes(_topology(task), _plan(task))
        targets[component] = Target(
            "source",
            component,
            drop=tuple(tables) if pruned else removed_keys(_topology(task)),
            add={
                key: Tensor(
                    "bf16",
                    shape,
                    PLAIN,
                    {
                        "value": Part(
                            "bf16", shape, source=PartSource(task, component, key, "value")
                        )
                    },
                )
                for key, shape in tables.items()
            },
        )
    config = canonical_json.decode(
        dual_adaln_pruned_config(_sections(), _plan("fl2va"), _plan("ref2va"))
    )
    for bound_task in _TASKS:
        config[TARGET_COMPONENT[bound_task]]["cozy_h3"][_BINDING] = bindings[bound_task]
    configs = {
        name: Config("copy", "source", name) for name in structure.configs if name != "model"
    }
    configs["model"] = Config("add")
    with ctx.output("model").open(
        Derivation(
            sources={
                "source": structure.source,
                **{name: inspection(ctx, bank).source for name, bank in banks.items()},
            },
            targets=targets,
            configs=configs,
            order=current_order(_asset("whole-order.json")).rows,
        )
    ) as transaction:
        if transaction.receipt is None:
            transaction.add_config("model", canonical_json.encode(config))
        return ctx.adopt_model(transaction.commit())


@invocable(memoize=True)
async def apply_adaln(
    ctx: Context,
    *,
    source: H3FullTransformer,
    fl2va: H3FullTransformer,
    ref2va: H3FullTransformer,
) -> ModelArtifact:
    return _apply_adaln(ctx, source=source, fl2va=fl2va, ref2va=ref2va, require_pruned=False)


@invocable(memoize=True)
async def retable_adaln(
    ctx: Context,
    *,
    source: H3FullTransformer,
    fl2va: H3FullTransformer,
    ref2va: H3FullTransformer,
) -> ModelArtifact:
    return _apply_adaln(ctx, source=source, fl2va=fl2va, ref2va=ref2va, require_pruned=True)
