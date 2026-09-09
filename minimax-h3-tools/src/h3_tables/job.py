"""Typed jobs producing the exact MiniMax H3 checkpoint variants and their AdaLN tables."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Annotated, Any, Literal, get_args

import msgspec
import torch
from cozy_runtime.author import (
    App,
    Context,
    Model,
    Telemetry,
    UnsupportedInput,
    WeightsConfig,
    WeightsOutput,
    WeightsPart,
    WeightsReceipt,
    WeightsSink,
    WeightsSource,
    WeightsTarget,
    WeightsTensor,
    WeightsTransaction,
    canonical_json,
)
from cozy_runtime.derive.quantization import (
    MAX_OUTPUT_BYTES,
    ArtifactQuantizationRequest,
    h3_quantization_plan,
    prepare_quantization,
    quantize_component_into,
)
from h3_table_layout import TableLayout

from . import adaln_operations as _adaln_operations
from . import lanes as _lanes
from .kernel import (
    H3Topology,
    LowRankAdapter,
    removed_keys,
    source_shapes,
    table_bytes,
    table_shapes,
)
from .kernel import precompute_tables as compute_tables
from .lanes import (
    COMPONENT_MAX_NEW_BYTES,
    LANES,
    Lane,
    Selection,
    lane_max_new_bytes,
    lane_treatments,
    write_cast,
)
from .legacy_config import upgrade_legacy_table_config
from .model_config import (
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
)
from .operations import assemble_full as assemble_full_artifact
from .order import current_order
from .order import full_order as _full_order
from .plans import (
    LAUNCH_PLAN_DIGESTS,
    TASKS,
    TURBO_PLAN_DIGESTS,
    Task,
    TimestepPlan,
    parse_declared_plan,
)
from .source import (
    ADAPTER_ALPHA,
    ADAPTER_RANK,
    TARGET_COMPONENT,
    H3FullTransformer,
    H3TurboAdapter,
    adapter_slice,
)
from .source import full_targets as _full_targets
from .source import select_full_targets as _select_full_targets
from .source import structures as _structures

app = App()

PLAIN_SPEC = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
FP8_SPEC = "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890"
MXFP8_SPEC = "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd"

SOURCE_READ_CHUNK = 32 << 20
SOURCE_SECTION: Mapping[Task, str] = {"fl2va": "transformer", "ref2va": "transformer_ref"}
TORCH_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32}
CUDA = torch.device("cuda")


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


@dataclass(frozen=True, slots=True)
class TableSet:
    """One admitted table set: the plan each task's rows are computed at, and whether the
    adapter's `adaln_proj.linear` slice is fused into them. A set is two output slots —
    decorator-time facts, so each set spells its own."""

    name: str
    checkpoint: str
    bank: str
    assets: Mapping[Task, str]
    digests: Mapping[Task, str]
    adapted: bool


#: `launch` is the served 30/40/50 union. `turbo` is PDD-8: eight evaluations on the
#: released grid, with the acceleration LoRA's modulation slice fused into the rows because
#: an AdaLN-pruned lane has no `adaln_proj.linear` for an adapter to attach to at inference.
TABLE_SETS: tuple[TableSet, ...] = (
    TableSet(
        "launch",
        "adaln-pruned",
        "tables",
        {task: f"timestep-plan.{task}.json" for task in TASKS},
        LAUNCH_PLAN_DIGESTS,
        adapted=False,
    ),
    TableSet(
        "turbo",
        "turbo-adaln-pruned",
        "turbo-tables",
        {task: f"timestep-plan.{task}.turbo.json" for task in TASKS},
        TURBO_PLAN_DIGESTS,
        adapted=True,
    ),
)
LAUNCH_SET = TABLE_SETS[0]
_ADAPTER_ALIAS: Mapping[Task, str] = {"fl2va": "fl2va_adapter", "ref2va": "ref2va_adapter"}


def _production_plan(task: Task, table_set: TableSet = LAUNCH_SET) -> TimestepPlan:
    plan = parse_declared_plan(_asset(table_set.assets[task]), digests=table_set.digests)
    if plan.task != task:
        raise ValueError(f"package timestep plan is {plan.task!r}, expected {task!r}")
    return plan


def _plans(table_set: TableSet) -> dict[Task, TimestepPlan]:
    return {task: _production_plan(task, table_set) for task in TASKS}


def _topologies(sections: Mapping[str, Mapping[str, Any]]) -> dict[Task, H3Topology]:
    return {task: H3Topology.from_config(dict(sections[SOURCE_SECTION[task]])) for task in TASKS}


def _check_table_budget(declared: int) -> None:
    """Refuse before native writes if any admitted task plan exceeds the per-slot ceiling."""
    topologies = _topologies(parse_production_config(_asset("model-config.json")))
    for table_set in TABLE_SETS:
        for task, plan in _plans(table_set).items():
            measured = table_bytes(topologies[task], plan)
            if measured > declared:
                raise ValueError(
                    f"the {table_set.name} {task} plan needs {measured} table bytes, "
                    f"above budget {declared}"
                )


# An author-declared per-task ceiling, not a second copy of table geometry.
# The committed plans currently use about 95% of this budget; shape validation
# and native byte accounting still require exactly the declared output tensors.
MAX_TABLE_BYTES = 1 << 30
MAX_FULL_BYTES = 64 << 10
MAX_PRUNED_BYTES = 2 * MAX_TABLE_BYTES + (128 << 10)
MAX_QUANTIZED_BYTES = 2 * MAX_OUTPUT_BYTES + MAX_PRUNED_BYTES

_check_table_budget(MAX_TABLE_BYTES)

#: The declared output slots. cr-114 reads this decorator statically from source over a
#: closed literal vocabulary, so it cannot be a comprehension over the catalogue and every
#: lane spells its own slot here. Drift is impossible rather than merely discouraged:
#: `_check_lane_outputs` proves this tuple IS the catalogue, name for name, with each
#: ceiling equal to what the lane's own treatments imply. Adding a lane is one catalogue
#: row, one line here and one `LaneName` member; getting any of the three wrong refuses at
#: import, before a worker is ever asked to produce anything.
#: Every lane carries the producer-wide video VAE normalisation, so every ceiling below
#: includes that component's own bound — including `bf16-full`, which authors nothing.
#:
#: Spelled as a literal, not as `COMPONENT_MAX_NEW_BYTES["video_vae"]`, because cr-114's
#: static reader folds the decorator from SOURCE over a closed vocabulary and refuses a
#: subscript: `describe` fails the whole package with `static_computed` rather than
#: guessing. The catalogue is still the authority — the equality below is checked at
#: import, so the two cannot drift; only the spelling is duplicated.
MAX_VIDEO_VAE_BYTES = 12 << 30
LANE_OUTPUTS = (
    WeightsOutput("bf16-full", max_new_bytes=MAX_FULL_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("bf16-adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("fp8-adaln-pruned", max_new_bytes=MAX_QUANTIZED_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("mxfp8-adaln-pruned", max_new_bytes=MAX_QUANTIZED_BYTES + MAX_VIDEO_VAE_BYTES),
)
LaneName = Literal["bf16-full", "bf16-adaln-pruned", "fp8-adaln-pruned", "mxfp8-adaln-pruned"]


def _check_lane_outputs() -> None:
    if COMPONENT_MAX_NEW_BYTES["video_vae"] != MAX_VIDEO_VAE_BYTES:
        raise ValueError(
            f"the declared video VAE bound {MAX_VIDEO_VAE_BYTES} is not the catalogue's "
            f"{COMPONENT_MAX_NEW_BYTES['video_vae']}"
        )
    declared = {output.name: output.max_new_bytes for output in LANE_OUTPUTS}
    if set(declared) != set(LANES) or set(get_args(LaneName)) != set(LANES):
        raise ValueError(
            f"declared outputs {sorted(declared)} and request literal "
            f"{sorted(get_args(LaneName))} are not the catalogue {sorted(LANES)}"
        )
    for name, lane in LANES.items():
        needed = lane_max_new_bytes(
            lane, full_bytes=MAX_FULL_BYTES, pruned_bytes=MAX_PRUNED_BYTES
        )
        if declared[name] != needed:
            raise ValueError(
                f"lane {name!r} declares {declared[name]} new bytes, but its treatments "
                f"need {needed}"
            )


_check_lane_outputs()


class ProductionRequest(msgspec.Struct, forbid_unknown_fields=True):
    pass


class LaneRequest(msgspec.Struct, forbid_unknown_fields=True):
    """Which reviewed lanes this attempt produces. The recipes are code, not request.

    ``max_relative_frobenius`` is the caller's tier-1 threshold over the worst per-tensor
    round trip of any treatment in any requested lane; each encoding's and each carrier's
    own representational bound is enforced underneath it regardless.
    """

    lanes: Annotated[tuple[LaneName, ...], msgspec.Meta(min_length=1)] = get_args(LaneName)
    max_relative_frobenius: float | None = None


class TreatmentStats(msgspec.Struct):
    """What one treatment of one component recorded on this attempt.

    The cast and the encoding keep separate round-trip numbers: they measure different
    carriers against different bounds, and a maximum over both would hide which one moved.
    """

    cast_keys: int = 0
    cast_worst_relative_frobenius: float | None = None
    encoded_keys: int = 0
    saturated_elements: int = 0
    worst_relative_frobenius: float | None = None
    reused_keys: int = 0
    source_bytes_read: int = 0
    new_bytes_written: int = 0


class WeightFidelity(msgspec.Struct):
    output_slot: str
    component: str
    treatment: str
    stats: TreatmentStats


class LaneReceipt(msgspec.Struct):
    lane: str
    modulation: str
    treated_components: list[str]
    tensorfs_receipt_digest: str
    weights_transaction_id: str
    replayed: bool


class LanesResult(msgspec.Struct):
    lanes: list[LaneReceipt]
    replayed_outputs: int
    source_bytes_read_this_run: int
    quantized_keys_this_run: int
    cast_keys_this_run: int
    weight_fidelity_this_run: list[WeightFidelity]


class AssemblyResult(msgspec.Struct):
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool


class TableSetReceipt(msgspec.Struct):
    table_set: str
    steps: list[int]
    plan_digests: dict[str, str]
    adapters: list[str]
    tensorfs_receipt_digest: str
    table_bank_receipt_digest: str
    replayed: bool
    table_tensors: int
    table_bytes_this_run: int


class RetableResult(msgspec.Struct):
    source_checkpoint: str
    table_sets: list[TableSetReceipt]
    source_bytes_read_this_run: int


def _read_source(
    transaction: WeightsTransaction,
    source: str,
    component: str,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, int]:
    length = math.prod(shape) * dtype.itemsize
    raw = bytearray(length)
    view = memoryview(raw)
    for offset in range(0, length, SOURCE_READ_CHUNK):
        transaction.source_read_into(
            source,
            component,
            name,
            "value",
            offset,
            view[offset : min(offset + SOURCE_READ_CHUNK, length)],
        )
    return torch.frombuffer(raw, dtype=dtype).reshape(shape), length


def _compute_table_parts(
    task: str,
    plan: TimestepPlan,
    topology: H3Topology,
    ctx: Context,
    transaction: WeightsTransaction,
    source: str,
    source_component: str,
    write_part: Callable[[str, bytes], None],
    tel: Telemetry,
    overall_range: tuple[float, float] = (0.0, 1.0),
    *,
    adapter: tuple[str, str] | None = None,
    device: torch.device = CUDA,
) -> tuple[int, int]:
    """One task's table pass; `adapter` names the (source alias, component) whose
    `adaln_proj.linear` LoRA slice is fused into the block rows."""
    source_bytes = 0
    written = 0

    def read(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal source_bytes
        value, length = _read_source(transaction, source, source_component, name, dtype, shape)
        source_bytes += length
        return value

    fused: LowRankAdapter | None = None
    if adapter is not None:
        alias, component = adapter

        def read_adapter(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
            nonlocal source_bytes
            value, length = _read_source(transaction, alias, component, name, dtype, shape)
            source_bytes += length
            return value

        fused = LowRankAdapter(ADAPTER_RANK, ADAPTER_ALPHA / ADAPTER_RANK, read_adapter)

    def write(name: str, value: torch.Tensor) -> None:
        nonlocal written
        raw = value.detach().to(device="cpu").contiguous().view(torch.uint16).numpy().tobytes()
        write_part(name, raw)
        written += len(raw)

    def progress(done: int, total: int) -> None:
        ctx.raise_if_cancelled()
        stage_fraction = done / total
        overall = overall_range[0] + stage_fraction * (overall_range[1] - overall_range[0])
        tel.progress(
            stage_fraction,
            stage=f"timestep-table-{task}",
            overall_fraction=overall,
        )
        tel.log(
            "precomputed timestep table",
            level="info",
            task=task,
            done=done,
            total=total,
        )

    compute_tables(
        plan=plan,
        topology=topology,
        read=read,
        write=write,
        progress=progress,
        device=device,
        adapter=fused,
    )
    expected = table_bytes(topology, plan)
    if written != expected or len(table_shapes(topology, plan)) != topology.num_layers + 1:
        raise ValueError(f"{task} emitted {written} table bytes, expected {expected}")
    return source_bytes, written


def _assembly_result(receipt: WeightsReceipt) -> AssemblyResult:
    return AssemblyResult(
        receipt.weights_transaction_id,
        receipt.tensorfs_receipt_digest,
        receipt.replayed,
    )


@app.job(
    weights=(WeightsOutput("model", max_new_bytes=64 << 10),),
)
def assemble_full(
    payload: ProductionRequest,
    dits: H3FullTransformer,
    shared: H3FullTransformer,
    artifacts: WeightsSink,
) -> AssemblyResult:
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    config = dual_full_config(sections)
    sources = {"dits": dits, "shared": shared}
    receipt = artifacts.derive(
        "model",
        sources=sources,
        targets=_select_full_targets(artifacts, sources),
        configs={"model": WeightsConfig(data=config, length=len(config))},
        order=_full_order(sections, current.rows),
    )
    return _assembly_result(receipt)


def _table_additions(
    topologies: Mapping[Task, H3Topology], plans: Mapping[Task, TimestepPlan]
) -> dict[Task, dict[str, WeightsTensor]]:
    return {
        task: {
            key: WeightsTensor(
                logical_dtype="bf16",
                shape=shape,
                encoding=PLAIN_SPEC,
                parts={"value": WeightsPart("bf16", shape)},
            )
            for key, shape in sorted(table_shapes(topologies[task], plans[task]).items())
        }
        for task in TASKS
    }


def _lane_targets(
    lane: Lane,
    sections: dict[str, dict[str, Any]],
    tables: Mapping[Task, Mapping[str, WeightsTensor]],
    full_targets: Mapping[str, WeightsTarget],
    selections: Mapping[str, Selection],
) -> dict[str, WeightsTarget]:
    """The five component targets of one lane.

    Every component starts as the FULL inheriting target: TensorFS copies its tensor
    metadata and ObjectRefs unchanged, so a component this lane neither prunes nor treats
    costs zero new bytes and stays byte-shared with every other lane. AdaLN pruning
    replaces 106 modulation rows per DiT with the computed table rows; each resolved
    treatment then drops exactly the keys it rewrites.
    """
    targets = dict(full_targets)
    if lane.modulation == "adaln-pruned":
        for task in TASKS:
            topology = H3Topology.from_config(sections[SOURCE_SECTION[task]])
            component = TARGET_COMPONENT[task]
            targets[component] = replace(
                full_targets[component],
                drop=tuple(
                    sorted(set(full_targets[component].drop) | set(removed_keys(topology)))
                ),
                add=dict(tables[task]),
            )
    for component, selection in selections.items():
        targets[component] = _lanes.apply(targets[component], selection)
    return targets


def _write_tables(
    task: Task,
    plan: TimestepPlan,
    topology: H3Topology,
    ctx: Context,
    source_transaction: WeightsTransaction,
    transactions: Mapping[str, WeightsTransaction],
    tel: Telemetry,
    overall_range: tuple[float, float],
    *,
    source: str = "dits",
    adapter: tuple[str, str] | None = None,
    device: torch.device = CUDA,
) -> tuple[int, int]:
    component = TARGET_COMPONENT[task]

    def write_part(name: str, raw: bytes) -> None:
        for transaction in transactions.values():
            transaction.add_part(component, name, "value", raw)

    return _compute_table_parts(
        task,
        plan,
        topology,
        ctx,
        source_transaction,
        source,
        component,
        write_part,
        tel,
        overall_range,
        adapter=adapter,
        device=device,
    )


def _receipt(transaction: WeightsTransaction) -> WeightsReceipt:
    receipt = transaction.receipt if transaction.replayed else transaction.commit()
    assert receipt is not None
    return receipt


def _requested(payload: LaneRequest) -> tuple[str, ...]:
    """The requested lanes in catalogue order, so commit order never depends on argv."""
    selected = set(payload.lanes)
    if len(selected) != len(payload.lanes):
        raise UnsupportedInput("requested lanes must be unique", code="h3_lanes_repeated")
    return tuple(name for name in LANES if name in selected)


def _settles_first(lane: Lane, selections: Mapping[str, Selection]) -> bool:
    """Whether this lane can finish before any table or encoding pass runs.

    The property being kept is retention, not idleness: a lane settles first so that an
    interrupted table pass cannot strand a checkpoint that was already complete. Before the
    producer-wide video VAE normalisation a FULL lane did no work at all and this read
    `not lane.components`; now it has one cast, which is cheap, local to a component no
    later pass touches, and no reason to hold the lane behind two table passes. So the test
    is what the lane still NEEDS — a table pass, or an encoding — and not whether it has
    any work at all.
    """
    return lane.modulation != "adaln-pruned" and not any(
        selection.plan is not None for selection in selections.values()
    )


def _bands(count: int, start: float, stop: float) -> list[tuple[float, float]]:
    width = (stop - start) / count if count else 0.0
    return [(start + index * width, start + (index + 1) * width) for index in range(count)]


def _treat(
    transaction: WeightsTransaction,
    ctx: Context,
    tel: Telemetry,
    request: ArtifactQuantizationRequest,
    *,
    selection: Selection,
    source: str,
) -> TreatmentStats:
    """Run one component's resolved treatment inside an already-open lane transaction.

    The cast runs first: the encoding reads the source at source precision either way, so
    ordering costs nothing, and a component that both casts and encodes then has exactly
    one pass over each of its two disjoint key sets.
    """
    component = selection.component
    cast = write_cast(
        transaction,
        ctx,
        tel,
        selection=selection,
        source=source,
        source_component=component,
        target_component=component,
    )
    stats = TreatmentStats(
        cast_keys=cast.converted_keys,
        cast_worst_relative_frobenius=cast.worst_relative_frobenius,
        reused_keys=cast.reused_keys,
        source_bytes_read=cast.source_bytes_read,
        new_bytes_written=cast.new_bytes_written,
    )
    if selection.plan is None or selection.treatment.encode is None:
        return stats
    encoded = quantize_component_into(
        transaction,
        ctx,
        request,
        tel,
        encoding=selection.treatment.encode,
        plan=selection.plan,
        component=selection.plan_component,
        source=source,
        source_component=component,
        target_component=component,
    )
    return msgspec.structs.replace(
        stats,
        encoded_keys=encoded.encoded_keys,
        saturated_elements=encoded.saturated_elements,
        worst_relative_frobenius=encoded.worst_relative_frobenius,
        reused_keys=stats.reused_keys + encoded.reused_keys,
        source_bytes_read=stats.source_bytes_read + encoded.source_bytes_read,
        new_bytes_written=stats.new_bytes_written + encoded.new_bytes_written,
    )


@app.job(
    name="lanes",
    weights=LANE_OUTPUTS,
    # Default [bindings] for these slots arrive once the minimax-h3 model repo
    # exists on the hub (H3 ingest, se-022/th-109 residue); until then no slot
    # facts derive at publish (cr-077) and these slots bind per-invocation like
    # this package's other jobs.
)
def lanes(
    ctx: Context,
    payload: LaneRequest,
    dits: H3FullTransformer,
    shared: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> LanesResult:
    """Produce the requested reviewed lanes from one pinned BF16 source, in one attempt.

    Every lane is a row of ``lanes.LANES``: a modulation plus, per component, what this
    producer does to it. A component no lane names is inherited by reference, so the
    conditioner and both VAEs stay byte-shared across every lane that leaves them alone.
    Outputs stay declared for the whole catalogue — the descriptor is static — and an
    unrequested lane is simply never opened.
    """
    requested = _requested(payload)
    sources = {"dits": dits, "shared": shared}
    granted = _structures(artifacts, sources)
    full_targets = _select_full_targets(artifacts, sources, granted)
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    topologies = _topologies(sections)
    launch = _plans(LAUNCH_SET)
    tables = _table_additions(topologies, launch)
    dit_plan = prepare_quantization(h3_quantization_plan())

    # Resolve every treatment against the granted structure BEFORE opening anything: a
    # refused component, an unmatched keep entry or an inert treatment must cost no
    # transaction and no byte.
    selections = {
        name: {
            component: _lanes.select(
                component,
                treatment,
                _lanes.carried(
                    full_targets[component],
                    granted[full_targets[component].source].tensors,
                ),
                dit_plan=dit_plan,
            )
            for component, treatment in lane_treatments(LANES[name]).items()
        }
        for name in requested
    }

    configs = {
        "full": dual_full_config(sections),
        "adaln-pruned": dual_adaln_pruned_config(sections, launch["fl2va"], launch["ref2va"]),
    }
    orders = {
        "full": _full_order(sections, current.rows),
        "adaln-pruned": current.rows,
    }
    receipts: dict[str, WeightsReceipt] = {}
    fidelity: list[WeightFidelity] = []
    source_bytes = 0
    quant_request = ArtifactQuantizationRequest(
        max_relative_frobenius=payload.max_relative_frobenius
    )

    with ExitStack() as stack:
        transactions = {
            name: stack.enter_context(
                artifacts.open(
                    name,
                    sources=sources,
                    targets=_lane_targets(
                        LANES[name], sections, tables, full_targets, selections[name]
                    ),
                    configs={
                        "model": WeightsConfig(
                            data=configs[LANES[name].modulation],
                            length=len(configs[LANES[name].modulation]),
                        )
                    },
                    order=orders[LANES[name].modulation],
                )
            )
            for name in requested
        }
        active = {
            name: transaction
            for name, transaction in transactions.items()
            if not transaction.replayed
        }
        # A lane needing neither a table pass nor an encoding settles first, so its
        # retention never depends on a later lane's. Its casts run here rather than in the
        # loop below, because the point is to be finished BEFORE the expensive passes.
        for name in [n for n in active if _settles_first(LANES[n], selections[n])]:
            transaction = active.pop(name)
            with tel.stage(name, overall_range=(0.0, 0.0)):
                for component, selection in selections[name].items():
                    stats = _treat(
                        transaction,
                        ctx,
                        tel,
                        quant_request,
                        selection=selection,
                        source=full_targets[component].source,
                    )
                    source_bytes += stats.source_bytes_read
                    fidelity.append(
                        WeightFidelity(
                            name, component, selection.treatment.describe(), stats
                        )
                    )
            transaction.add_config("model", configs[LANES[name].modulation])
            receipts[name] = transaction.commit()

        pruned = {
            name: transaction
            for name, transaction in active.items()
            if LANES[name].modulation == "adaln-pruned"
        }
        computed = 0.35 if pruned else 0.0
        if pruned:
            if not torch.cuda.is_available():
                raise ValueError("H3 AdaLN timestep-table precompute requires a CUDA worker")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            source_transaction = next(iter(pruned.values()))
            for task, overall_range in zip(TASKS, _bands(len(TASKS), 0.0, computed), strict=True):
                with tel.stage(f"timestep-table-{task}", overall_range=overall_range):
                    source_bytes += _write_tables(
                        task,
                        launch[task],
                        topologies[task],
                        ctx,
                        source_transaction,
                        pruned,
                        tel,
                        overall_range,
                    )[0]

        for name, overall_range in zip(
            list(active), _bands(len(active), computed, 1.0), strict=True
        ):
            transaction = active[name]
            with tel.stage(name, overall_range=overall_range):
                for component, selection in selections[name].items():
                    stats = _treat(
                        transaction,
                        ctx,
                        tel,
                        quant_request,
                        selection=selection,
                        source=full_targets[component].source,
                    )
                    source_bytes += stats.source_bytes_read
                    fidelity.append(
                        WeightFidelity(
                            name, component, selection.treatment.describe(), stats
                        )
                    )
                    tel.log(
                        "weight fidelity",
                        level="info",
                        output_slot=name,
                        component=component,
                        treatment=selection.treatment.describe(),
                        **msgspec.to_builtins(stats),
                    )
                # Keep a finished checkpoint replayable if a later lane fails.
                transaction.add_config("model", configs[LANES[name].modulation])
                receipts[name] = transaction.commit()
            tel.progress(1.0, stage=f"commit-{name}", overall_fraction=overall_range[1])
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = _receipt(transaction)

    measured = [row.stats for row in fidelity]
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric(
        "h3.new_bytes",
        float(sum(stat.new_bytes_written for stat in measured)),
        unit="bytes",
    )
    return LanesResult(
        lanes=[
            LaneReceipt(
                lane=name,
                modulation=LANES[name].modulation,
                treated_components=sorted(lane_treatments(LANES[name])),
                tensorfs_receipt_digest=receipts[name].tensorfs_receipt_digest,
                weights_transaction_id=receipts[name].weights_transaction_id,
                replayed=receipts[name].replayed,
            )
            for name in requested
        ],
        replayed_outputs=sum(receipt.replayed for receipt in receipts.values()),
        source_bytes_read_this_run=source_bytes,
        quantized_keys_this_run=sum(stat.encoded_keys for stat in measured),
        cast_keys_this_run=sum(stat.cast_keys for stat in measured),
        weight_fidelity_this_run=fidelity,
    )




def _retable_targets(
    pruned: WeightsSource,
    full: WeightsSource,
    topologies: Mapping[Task, H3Topology],
    tables: Mapping[Task, Mapping[str, WeightsTensor]],
    adapters: Mapping[Task, WeightsSource] | None = None,
) -> tuple[dict[str, WeightsTarget], dict[str, WeightsTarget], tuple[tuple[str, str], ...]]:
    """Declare one table set: its bank derived from `full`, its checkpoint from `pruned`.

    A transaction reads only through source components its targets derive from, so the
    modulation weights are read through the bank transaction (both DiTs from `full`, every
    other row dropped) while the retabled checkpoint inherits `pruned` by reference and
    replaces exactly its table keys. An adapted set's bank also carries each adapter's
    `adaln_proj.linear` slice by reference as `<task>_adapter` — the rows its tables were
    fused from, and the smallest derivation TensorFS admits (an unused source alias and an
    empty component both refuse); those rows are returned as the bank's trailing
    construction order. Refuses before any read unless `pruned` carries table rows and no
    dynamic modulation weights for both DiTs, `full` carries the exact modulation weights
    those rows are computed from, and each adapter carries the complete slice.
    """
    present = {(tensor.component, tensor.key): tensor for tensor in pruned.tensors}
    full_present = {(tensor.component, tensor.key): tensor for tensor in full.tensors}
    components = {tensor.component for tensor in pruned.tensors}
    if components != set(_full_targets()):
        raise ValueError(f"retable source components are {sorted(components)}")
    bank: dict[str, WeightsTarget] = {}
    trailing: list[tuple[str, str]] = []
    retabled = {
        component: WeightsTarget(source="pruned", source_component=component)
        for component in components - set(TARGET_COMPONENT.values())
    }
    for task in TASKS:
        component = TARGET_COMPONENT[task]
        topology = topologies[task]
        keys = {key for owner, key in present if owner == component}
        if not set(tables[task]) <= keys or set(removed_keys(topology)) & keys:
            raise ValueError(f"{component} is not an AdaLN-pruned source carrying table rows only")
        for key, (dtype, shape) in source_shapes(topology).items():
            tensor = full_present.get((component, key))
            if (
                tensor is None
                or tensor.shape != shape
                or TORCH_DTYPE.get(tensor.logical_dtype) != dtype
            ):
                raise ValueError(f"full source lacks modulation weight {component}/{key}")
        bank[component] = WeightsTarget(
            source="full",
            source_component=component,
            drop=tuple(sorted(key for owner, key in full_present if owner == component)),
            add=tables[task],
        )
        retabled[component] = WeightsTarget(
            source="pruned",
            source_component=component,
            drop=tuple(sorted(tables[task])),
            add=tables[task],
        )
        if adapters is not None:
            alias = _ADAPTER_ALIAS[task]
            adapter_component, unread = adapter_slice(adapters[task], topology)
            bank[alias] = WeightsTarget(
                source=alias, source_component=adapter_component, drop=unread
            )
            dropped = set(unread)
            trailing.extend(
                (alias, tensor.key)
                for tensor in adapters[task].tensors
                if tensor.key not in dropped
            )
    return bank, retabled, tuple(trailing)


#: Two slots per table set, spelled literally for the static reader; `_check_retable_outputs`
#: proves this tuple IS `TABLE_SETS`, slot for slot.
RETABLE_OUTPUTS = (
    WeightsOutput("adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
    WeightsOutput("tables", max_new_bytes=MAX_PRUNED_BYTES),
    WeightsOutput("turbo-adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
    WeightsOutput("turbo-tables", max_new_bytes=MAX_PRUNED_BYTES),
)


def _check_retable_outputs() -> None:
    declared = tuple((output.name, output.max_new_bytes) for output in RETABLE_OUTPUTS)
    spelled = tuple(
        (slot, MAX_PRUNED_BYTES)
        for table_set in TABLE_SETS
        for slot in (table_set.checkpoint, table_set.bank)
    )
    if declared != spelled:
        raise ValueError(f"retable declares {declared}, but its table sets need {spelled}")


_check_retable_outputs()


@dataclass(frozen=True, slots=True)
class RetableWork:
    """What one retable attempt derives from its package assets, keyed by table-set name.

    The orchestration takes these as values so the store proofs drive the identical code
    path over a tiny topology on CPU.
    """

    topologies: Mapping[Task, H3Topology]
    plans: Mapping[str, Mapping[Task, TimestepPlan]]
    configs: Mapping[str, bytes]
    order: tuple[tuple[str, str], ...]
    device: torch.device


def _retable(
    ctx: Context,
    tel: Telemetry,
    artifacts: WeightsSink,
    work: RetableWork,
    *,
    full: H3FullTransformer,
    pruned: H3FullTransformer,
    adapters: Mapping[Task, H3TurboAdapter],
) -> RetableResult:
    full_structure = artifacts.structure(full)
    pruned_structure = artifacts.structure(pruned)
    adapter_structures = {task: artifacts.structure(model) for task, model in adapters.items()}
    declared: dict[str, tuple[dict[str, WeightsTarget], dict[str, WeightsTarget], int]] = {}
    orders: dict[str, tuple[tuple[str, str], ...]] = {}
    for table_set in TABLE_SETS:
        tables = _table_additions(work.topologies, work.plans[table_set.name])
        bank_targets, targets, trailing = _retable_targets(
            pruned_structure,
            full_structure,
            work.topologies,
            tables,
            adapter_structures if table_set.adapted else None,
        )
        declared[table_set.name] = (
            bank_targets,
            targets,
            sum(len(rows) for rows in tables.values()),
        )
        orders[table_set.name] = (
            *(
                row
                for row in work.order
                if row[0] in bank_targets and row[1] in bank_targets[row[0]].add
            ),
            *trailing,
        )

    source_bytes = 0
    written: dict[str, int] = {}
    receipts: dict[str, WeightsReceipt] = {}
    with ExitStack() as stack:
        opened: dict[str, tuple[WeightsTransaction, WeightsTransaction]] = {}
        for table_set in TABLE_SETS:
            bank_targets, targets, _ = declared[table_set.name]
            raw = work.configs[table_set.name]
            config = {"model": WeightsConfig(data=raw, length=len(raw))}
            sources: dict[str, Model[object]] = {"full": full}
            if table_set.adapted:
                sources.update({_ADAPTER_ALIAS[task]: adapters[task] for task in TASKS})
            bank = stack.enter_context(
                artifacts.open(
                    table_set.bank,
                    sources=sources,
                    targets=bank_targets,
                    configs=config,
                    order=orders[table_set.name],
                )
            )
            checkpoint = stack.enter_context(
                artifacts.open(
                    table_set.checkpoint,
                    sources={"pruned": pruned},
                    targets=targets,
                    configs=config,
                    order=work.order,
                )
            )
            opened[table_set.name] = (checkpoint, bank)
        active = [
            table_set
            for table_set in TABLE_SETS
            if not all(transaction.replayed for transaction in opened[table_set.name])
        ]
        if active:
            if work.device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("H3 timestep-table precompute requires a CUDA worker")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        bands = iter(_bands(len(active) * len(TASKS), 0.0, 1.0))
        for table_set in active:
            checkpoint, bank = opened[table_set.name]
            if bank.replayed:
                raise ValueError(
                    f"the {table_set.name} table bank was retained without its retabled "
                    "checkpoint; rerun under a new request identity"
                )
            writers = {
                name: transaction
                for name, transaction in (("checkpoint", checkpoint), ("bank", bank))
                if not transaction.replayed
            }
            bank_targets = declared[table_set.name][0]
            for task in TASKS:
                alias = _ADAPTER_ALIAS[task]
                overall_range = next(bands)
                stage = f"timestep-table-{table_set.name}-{task}"
                with tel.stage(stage, overall_range=overall_range):
                    read, emitted = _write_tables(
                        task,
                        work.plans[table_set.name][task],
                        work.topologies[task],
                        ctx,
                        bank,
                        writers,
                        tel,
                        overall_range,
                        source="full",
                        adapter=(
                            (alias, bank_targets[alias].source_component)
                            if table_set.adapted
                            else None
                        ),
                        device=work.device,
                    )
                source_bytes += read
                written[table_set.name] = written.get(table_set.name, 0) + emitted
            # The retabled checkpoint commits first: its retention never strands the bank.
            for slot, transaction in ((table_set.checkpoint, checkpoint), (table_set.bank, bank)):
                if not transaction.replayed:
                    transaction.add_config("model", work.configs[table_set.name])
                    receipts[slot] = transaction.commit()
        for table_set in TABLE_SETS:
            checkpoint, bank = opened[table_set.name]
            for slot, transaction in ((table_set.checkpoint, checkpoint), (table_set.bank, bank)):
                if slot not in receipts:
                    receipts[slot] = _receipt(transaction)
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    return RetableResult(
        pruned.checkpoint_ref,
        [
            TableSetReceipt(
                table_set=table_set.name,
                steps=list(work.plans[table_set.name]["fl2va"].steps),
                plan_digests={task: work.plans[table_set.name][task].digest for task in TASKS},
                adapters=(
                    [adapters[task].checkpoint_ref for task in TASKS] if table_set.adapted else []
                ),
                tensorfs_receipt_digest=receipts[table_set.checkpoint].tensorfs_receipt_digest,
                table_bank_receipt_digest=receipts[table_set.bank].tensorfs_receipt_digest,
                replayed=receipts[table_set.checkpoint].replayed,
                table_tensors=declared[table_set.name][2],
                table_bytes_this_run=written.get(table_set.name, 0),
            )
            for table_set in TABLE_SETS
        ],
        source_bytes,
    )


@app.job(name="retable", weights=RETABLE_OUTPUTS)
def retable(
    ctx: Context,
    payload: ProductionRequest,
    full: H3FullTransformer,
    pruned: H3FullTransformer,
    fl2va_adapter: H3TurboAdapter,
    ref2va_adapter: H3TurboAdapter,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> RetableResult:
    """Recompute one AdaLN-pruned checkpoint's tables for every admitted table set.

    Every non-table tensor of `pruned` (BF16, FP8 or MXFP8 alike) is inherited by reference;
    only the modulation rows are read from `full` and recomputed, so a plan that adds
    schedules costs table bytes, never a requantization. Each set commits a retabled
    checkpoint and a two-DiT table bank, the transaction its modulation reads are scoped to.
    The turbo set fuses each task's PDD adapter into its block rows — only the
    `adaln_proj.linear` slice; the adapter's other six target families apply at inference.
    """
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    plans = {table_set.name: _plans(table_set) for table_set in TABLE_SETS}
    work = RetableWork(
        topologies=_topologies(sections),
        plans=plans,
        configs={
            name: dual_adaln_pruned_config(sections, chosen["fl2va"], chosen["ref2va"])
            for name, chosen in plans.items()
        },
        order=current_order(_asset("whole-order.json")).rows,
        device=CUDA,
    )
    return _retable(
        ctx,
        tel,
        artifacts,
        work,
        full=full,
        pruned=pruned,
        adapters={"fl2va": fl2va_adapter, "ref2va": ref2va_adapter},
    )


# Register after defining the source capability used by the managed operation.
from . import operations  # noqa: E402

app.job(
    operations.quantize,
    name="quantize-artifact",
    weights=(WeightsOutput("model", max_new_bytes=MAX_QUANTIZED_BYTES),),
)


app.job(
    _adaln_operations.select_adaln_weights,
    name="select-adaln-weights",
    weights=(WeightsOutput("model", 0),),
)
app.job(
    _adaln_operations.compute_adaln_tables,
    name="compute-adaln-tables",
    weights=(WeightsOutput("model", MAX_TABLE_BYTES),),
)
app.job(
    _adaln_operations.apply_adaln,
    name="apply-adaln",
    weights=(
        WeightsOutput("model", 0),
        WeightsOutput("fl2va-weights", 0),
        WeightsOutput("ref2va-weights", 0),
    ),
)

app.job(assemble_full_artifact, name="assemble-full-artifact", weights=(WeightsOutput("model", 0),))

app.job(_adaln_operations.retable_adaln, name="retable-adaln", weights=(WeightsOutput("model", 0),))


class RestampResult(msgspec.Struct):
    """What one restamp re-emitted, and what it inherited untouched."""

    weights_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool
    modulation: str
    inherited_components: tuple[str, ...]
    normalised_components: tuple[str, ...]
    cast_keys: int
    reused_keys: int
    cast_worst_relative_frobenius: float | None
    source_bytes_read: int
    new_bytes_written: int


def _source_modulation(source: WeightsSource, sections: Mapping[str, dict[str, Any]]) -> str:
    """Read the source's own modulation off its DiT rows rather than off a request field.

    An AdaLN-pruned checkpoint carries the timestep table rows and none of the dynamic
    modulation weights; a FULL one carries the modulation weights and no tables. Anything
    else is not a lane this producer emitted, and the restamp refuses rather than guessing.
    """
    present = {(tensor.component, tensor.key) for tensor in source.tensors}
    verdicts: set[str] = set()
    for task, section in SOURCE_SECTION.items():
        component = TARGET_COMPONENT[task]
        topology = H3Topology.from_config(sections[section])
        keys = {key for owner, key in present if owner == component}
        if not keys:
            raise UnsupportedInput(
                f"restamp source has no {component}", code="h3_component_absent"
            )
        dynamic = set(removed_keys(topology)) & keys
        tables = set(table_shapes(topology, _production_plan(task))) & keys
        if tables and not dynamic:
            verdicts.add("adaln-pruned")
        elif dynamic and not tables:
            verdicts.add("full")
        else:
            raise UnsupportedInput(
                f"{component} carries neither a clean FULL nor a clean AdaLN-pruned row set "
                f"({len(dynamic)} modulation rows, {len(tables)} table rows)",
                code="h3_restamp_source_shape",
            )
    if len(verdicts) != 1:
        raise UnsupportedInput(
            f"the two DiTs disagree about modulation: {sorted(verdicts)}",
            code="h3_restamp_source_shape",
        )
    return verdicts.pop()



def _check_emitted_config(
    document: bytes, modulation: str, source: WeightsSource
) -> None:
    """Validate row meanings and stored table dimensions before inheriting table bytes."""
    value = canonical_json.decode(document)
    fields = {"task", "modulation"} | (
        {"table_keys"} if modulation == "adaln-pruned" else set()
    )
    tensors = {(tensor.component, tensor.key): tensor for tensor in source.tensors}
    for task, component in TARGET_COMPONENT.items():
        extension = value[component]["cozy_h3"]
        if set(extension) not in (fields, fields | {"generating_projection_digest"}):
            raise UnsupportedInput(
                f"emitted {component} cozy_h3 has unexpected metadata fields",
                code="h3_restamp_config_shape",
            )
        if extension["modulation"] != modulation or extension["task"] != task:
            raise UnsupportedInput(
                f"emitted {component} task/modulation differs from the source tensors",
                code="h3_restamp_config_shape",
            )
        if modulation == "adaln-pruned":
            layout = TableLayout.parse(extension["table_keys"])
            plan = replace(
                _production_plan(task), timesteps=layout.timesteps, block_rows=layout.block_keys
            )
            topology = H3Topology.from_config(value[component])
            for key, shape in table_shapes(topology, plan).items():
                tensor = tensors.get((component, key))
                if (
                    tensor is None
                    or tensor.logical_dtype != "bf16"
                    or tensor.encoding != PLAIN_SPEC
                    or tensor.shape != shape
                    or len(tensor.parts) != 1
                    or (tensor.parts[0].name, tensor.parts[0].dtype, tensor.parts[0].shape)
                    != ("value", "bf16", shape)
                ):
                    raise UnsupportedInput(
                        f"{component}.{key} stored dimensions/encoding differ from its table keys",
                        code="h3_restamp_table_layout",
                    )


@app.job(
    name="restamp",
    weights=(
        WeightsOutput("restamped", max_new_bytes=MAX_VIDEO_VAE_BYTES + (128 << 10)),
    ),
)
def restamp(
    ctx: Context,
    payload: ProductionRequest,
    lane: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> RestampResult:
    """Upgrade a lane's legacy table metadata and normalize video-VAE decode operands.

    Existing explicit row labels are preserved. Legacy plan stamps are migrated only after
    proving their historical table order and stored dimensions. DiTs, conditioner and audio
    VAE bytes are inherited; only video-VAE operands that require casting are read/written.
    """
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    granted = artifacts.structure(lane)
    modulation = _source_modulation(granted, sections)
    components = sorted({tensor.component for tensor in granted.tensors})
    if set(components) != set(_full_targets()):
        raise UnsupportedInput(
            f"restamp source components are {components}", code="h3_restamp_source_shape"
        )

    targets: dict[str, WeightsTarget] = {
        component: WeightsTarget(source="lane", source_component=component)
        for component in components
    }
    selections = {
        component: _lanes.select(
            component,
            treatment,
            _lanes.carried(targets[component], granted.tensors),
            allow_inert=True,
        )
        for component, treatment in _lanes.NORMALISED_COMPONENTS.items()
    }
    pending = {name: s for name, s in selections.items() if s.cast}
    for component, selection in pending.items():
        targets[component] = _lanes.apply(targets[component], selection)

    document = upgrade_legacy_table_config(
        artifacts.config(lane, "model"),
        {task: _production_plan(task) for task in TASKS},
    )
    _check_emitted_config(document, modulation, granted)
    order = current_order(_asset("whole-order.json"))
    rows = _full_order(sections, order.rows) if modulation == "full" else order.rows

    stats = _lanes.CastStats(0, 0, 0, 0, None)
    with artifacts.open(
        "restamped",
        sources={"lane": lane},
        targets=targets,
        configs={"model": WeightsConfig(data=document, length=len(document))},
        order=rows,
    ) as transaction:
        if transaction.replayed:
            receipt = _receipt(transaction)
        else:
            for component, selection in pending.items():
                with tel.stage(f"cast-{component}"):
                    cast = write_cast(
                        transaction,
                        ctx,
                        tel,
                        selection=selection,
                        source="lane",
                        source_component=component,
                        target_component=component,
                    )
                stats = _lanes.CastStats(
                    stats.converted_keys + cast.converted_keys,
                    stats.reused_keys + cast.reused_keys,
                    stats.source_bytes_read + cast.source_bytes_read,
                    stats.new_bytes_written + cast.new_bytes_written,
                    max(
                        (
                            value
                            for value in (
                                stats.worst_relative_frobenius,
                                cast.worst_relative_frobenius,
                            )
                            if value is not None
                        ),
                        default=None,
                    ),
                )
            # Metadata migration still runs when every VAE operand is already normalized.
            transaction.add_config("model", document)
            receipt = transaction.commit()
    tel.metric("h3.source_bytes", float(stats.source_bytes_read), unit="bytes")
    return RestampResult(
        receipt.weights_transaction_id,
        receipt.tensorfs_receipt_digest,
        receipt.replayed,
        modulation,
        tuple(component for component in components if component not in pending),
        tuple(sorted(pending)),
        stats.converted_keys,
        stats.reused_keys,
        stats.worst_relative_frobenius,
        stats.source_bytes_read,
        stats.new_bytes_written,
    )
