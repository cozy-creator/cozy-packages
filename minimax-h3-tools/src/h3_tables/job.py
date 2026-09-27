"""Typed jobs producing the exact MiniMax H3 checkpoint variants and their AdaLN tables."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from importlib.resources import files
from typing import Any, Literal, cast, get_args

import msgspec
import torch
from cozy_runtime.author import (
    App,
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    canonical_json,
    invocable,
)
from cozy_runtime.derive.quantization import (
    MAX_OUTPUT_BYTES,
    ArtifactQuantizationRequest,
    QuantizationStats,
    prepare_quantization,
    quantize_component_into,
)
from cozy_runtime.models.minimax_h3.table_layout import TableLayout
from tensorfs.derived import (
    Config,
    Derivation,
    DerivedTransaction,
    Part,
    SourceInspection,
    Target,
    Tensor,
)

from . import adaln_operations as _adaln_operations
from . import lanes as _lanes
from ._memo import LANES as LANES_MEMO
from .kernel import H3Topology, removed_keys, source_shapes, table_bytes, table_shapes
from .kernel import precompute_tables as compute_tables
from .lanes import (
    COMPONENT_MAX_NEW_BYTES,
    LANES,
    NORMALISED_COMPONENTS,
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
from .plans import TASKS, TimestepPlan, parse_declared_plan
from .quantization import h3_quantization_plan
from .source import (
    TARGET_COMPONENT,
    H3FullTransformer,
    inspection,
)
from .source import full_targets as _full_targets
from .source import select_full_targets as _select_full_targets
from .source import structures as _structures

app = App()

PLAIN_SPEC = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
FP8_SPEC = "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890"
MXFP8_SPEC = "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd"

SOURCE_READ_CHUNK = 32 << 20
SOURCE_SECTION = {"fl2va": "transformer", "ref2va": "transformer_ref"}
TORCH_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32}


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


def _production_plan(task: str) -> TimestepPlan:
    plan = parse_declared_plan(_asset(f"timestep-plan.{task}.json"))
    if plan.task != task:
        raise ValueError(f"package timestep plan is {plan.task!r}, expected {task!r}")
    return plan


def _check_table_budget(declared: int) -> None:
    """Refuse before native writes if either committed task plan exceeds the ceiling."""
    sections = parse_production_config(_asset("model-config.json"))
    measured = max(
        table_bytes(H3Topology.from_config(sections[section]), _production_plan(task))
        for task, section in SOURCE_SECTION.items()
    )
    if measured > declared:
        raise ValueError(
            f"the committed plans need {measured} table bytes, above budget {declared}"
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
    WeightsOutput("bf16-pruned", max_new_bytes=MAX_PRUNED_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("fp8-pruned", max_new_bytes=MAX_QUANTIZED_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("mxfp8-pruned", max_new_bytes=MAX_QUANTIZED_BYTES + MAX_VIDEO_VAE_BYTES),
)
LaneName = Literal["bf16-full", "bf16-pruned", "fp8-pruned", "mxfp8-pruned"]


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
        needed = lane_max_new_bytes(lane, full_bytes=MAX_FULL_BYTES, pruned_bytes=MAX_PRUNED_BYTES)
        if declared[name] != needed:
            raise ValueError(
                f"lane {name!r} declares {declared[name]} new bytes, but its treatments "
                f"need {needed}"
            )


_check_lane_outputs()


class ProductionRequest(msgspec.Struct, forbid_unknown_fields=True):
    pass


ALL_LANES: tuple[LaneName, ...] = ("bf16-full", "bf16-pruned", "fp8-pruned", "mxfp8-pruned")


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


class LaneArtifact(msgspec.Struct, frozen=True):
    lane: str
    modulation: str
    treated_components: list[str]
    model: ModelArtifact


class H3Lanes(msgspec.Struct, frozen=True):
    """The committed lanes. Per-attempt measurements are telemetry, never memoized."""

    lanes: list[LaneArtifact]


class AssemblyResult(msgspec.Struct):
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool


class RetableResult(msgspec.Struct, frozen=True):
    steps: list[int]
    adaln_pruned: ModelArtifact
    tables: ModelArtifact


def _read_source(
    transaction: DerivedTransaction,
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
    transaction: DerivedTransaction,
    source: str,
    source_component: str,
    write_part: Callable[[str, bytes], None],
    tel: Telemetry,
    overall_range: tuple[float, float] = (0.0, 1.0),
    completed: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    source_bytes = 0
    written = 0

    def read(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal source_bytes
        value, length = _read_source(transaction, source, source_component, name, dtype, shape)
        source_bytes += length
        return value

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
        device=torch.device("cuda"),
        completed=completed,
    )
    expected = sum(
        math.prod(shape) * 2
        for key, shape in table_shapes(topology, plan).items()
        if key not in completed
    )
    if written != expected or len(table_shapes(topology, plan)) != topology.num_layers + 1:
        raise ValueError(f"{task} emitted {written} table bytes, expected {expected}")
    return source_bytes, written


def _assembly_result(receipt: Mapping[str, Any], replayed: bool) -> AssemblyResult:
    return AssemblyResult(receipt["transaction_id"], canonical_json.digest(receipt), replayed)


@app.job(
    weights=(WeightsOutput("model", max_new_bytes=64 << 10),),
)
def assemble_full(
    ctx: Context,
    payload: ProductionRequest,
    dits: H3FullTransformer,
    shared: H3FullTransformer,
) -> AssemblyResult:
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    config = dual_full_config(sections)
    sources = {"dits": dits, "shared": shared}
    with ctx.output("model").open(
        Derivation(
            sources={name: info.source for name, info in _structures(ctx, sources).items()},
            targets=_select_full_targets(ctx, sources),
            configs={"model": Config("add")},
            order=_full_order(sections, current.rows),
        )
    ) as transaction:
        replayed = transaction.receipt is not None
        if not replayed:
            transaction.add_config("model", config)
        receipt = transaction.commit()
        ctx.adopt_model(receipt)
    return _assembly_result(receipt, replayed)


def _table_additions(
    sections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Tensor]]:
    additions: dict[str, dict[str, Tensor]] = {}
    for task, section in SOURCE_SECTION.items():
        plan = _production_plan(task)
        topology = H3Topology.from_config(sections[section])
        additions[task] = {
            key: Tensor(
                logical_dtype="bf16",
                shape=shape,
                encoding=PLAIN_SPEC,
                parts={"value": Part("bf16", shape)},
            )
            for key, shape in sorted(table_shapes(topology, plan).items())
        }
    return additions


def _lane_targets(
    lane: Lane,
    sections: dict[str, dict[str, Any]],
    tables: Mapping[str, Mapping[str, Tensor]],
    full_targets: Mapping[str, Target],
    selections: Mapping[str, Selection],
) -> dict[str, Target]:
    """The five component targets of one lane.

    Every component starts as the FULL inheriting target: TensorFS copies its tensor
    metadata and ObjectRefs unchanged, so a component this lane neither prunes nor treats
    costs zero new bytes and stays byte-shared with every other lane. AdaLN pruning
    replaces 106 modulation rows per DiT with the computed table rows; each resolved
    treatment then drops exactly the keys it rewrites.
    """
    targets = dict(full_targets)
    if lane.modulation == "adaln-pruned":
        for task, section in SOURCE_SECTION.items():
            topology = H3Topology.from_config(sections[section])
            component = TARGET_COMPONENT[task]
            targets[component] = replace(
                full_targets[component],
                drop=tuple(sorted(set(full_targets[component].drop) | set(removed_keys(topology)))),
                add=dict(tables[task]),
            )
    for component, selection in selections.items():
        targets[component] = _lanes.apply(targets[component], selection)
    return targets


def _write_tables(
    task: str,
    ctx: Context,
    source_transaction: DerivedTransaction,
    transactions: Mapping[str, DerivedTransaction],
    tel: Telemetry,
    overall_range: tuple[float, float],
    source: str = "dits",
) -> tuple[int, int, int]:
    """Compute one task's tables into every transaction; (read, written, reused) counts."""
    sections = parse_production_config(_asset("model-config.json"))
    plan = _production_plan(task)
    topology = H3Topology.from_config(sections[SOURCE_SECTION[task]])
    component = TARGET_COMPONENT[task]
    completed = {
        name: {
            key
            for owner, key, role in transaction.completed_parts()
            if owner == component and role == "value"
        }
        for name, transaction in transactions.items()
    }
    shared_completed = (
        frozenset.intersection(*(frozenset(keys) for keys in completed.values()))
        if completed
        else frozenset()
    )

    def write_part(name: str, raw: bytes) -> None:
        for output, transaction in transactions.items():
            if name not in completed[output]:
                transaction.add_part(component, name, "value", raw)
                transaction.checkpoint()

    read, written = _compute_table_parts(
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
        completed=shared_completed,
    )
    return read, written, len(shared_completed & table_shapes(topology, plan).keys())


def _requested(requested: tuple[LaneName, ...]) -> tuple[str, ...]:
    """The requested lanes in catalogue order, so commit order never depends on argv."""
    if not requested or len(set(requested)) != len(requested):
        raise UnsupportedInput(
            f"lanes must name each of {sorted(LANES)} at most once", code="h3_lanes_repeated"
        )
    return tuple(name for name in LANES if name in requested)


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
    transaction: DerivedTransaction,
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
    casted = write_cast(
        transaction,
        ctx,
        tel,
        selection=selection,
        source=source,
        source_component=component,
        target_component=component,
    )
    stats = TreatmentStats(
        cast_keys=casted.converted_keys,
        cast_worst_relative_frobenius=casted.worst_relative_frobenius,
        reused_keys=casted.reused_keys,
        source_bytes_read=casted.source_bytes_read,
        new_bytes_written=casted.new_bytes_written,
    )
    if selection.plan is None or selection.treatment.encode is None:
        return stats
    plan, encoding = selection.plan, selection.treatment.encode
    tensors = [tensor for tensor in plan.tensors if tensor.component == selection.plan_component]
    count = max(1, min(len(tensors), len(os.sched_getaffinity(0))))
    shared = cast(Telemetry, _SharedTelemetry(tel, len(tensors)))

    def encode(shard: list[Any]) -> QuantizationStats:
        return quantize_component_into(
            transaction,
            ctx,
            request,
            shared,
            encoding=encoding,
            plan=msgspec.structs.replace(plan, tensors=shard),
            component=selection.plan_component,
            source=source,
            source_component=component,
            target_component=component,
        )

    # Tensors encode independently: numpy releases the GIL, the handle serializes reads and
    # writes, and the declared order fixes the output, so shards only change the wall time.
    with ThreadPoolExecutor(count) as pool:
        shards = list(pool.map(encode, [tensors[index::count] for index in range(count)]))
    worsts = [shard.worst_relative_frobenius for shard in shards]
    return msgspec.structs.replace(
        stats,
        encoded_keys=sum(shard.encoded_keys for shard in shards),
        saturated_elements=sum(shard.saturated_elements for shard in shards),
        worst_relative_frobenius=None if None in worsts else max(cast(list[float], worsts)),
        reused_keys=stats.reused_keys + sum(shard.reused_keys for shard in shards),
        source_bytes_read=stats.source_bytes_read + sum(s.source_bytes_read for s in shards),
        new_bytes_written=stats.new_bytes_written + sum(s.new_bytes_written for s in shards),
    )


class _SharedTelemetry:
    """Shard threads report through one lock as a single monotone per-stage tensor count."""

    def __init__(self, tel: Telemetry, total: int) -> None:
        self._tel, self._total, self._done = tel, total, 0
        self._lock = threading.Lock()

    def progress(self, _fraction: float, *, stage: str, **_: Any) -> None:
        with self._lock:
            self._done += 1
            self._tel.progress(self._done / self._total, stage=stage)

    def log(self, message: str, **fields: Any) -> None:
        with self._lock:
            self._tel.log(message, **fields)


@invocable(memoize=True, memo_version="h3-lanes/1", memo_dependencies=LANES_MEMO)
async def lanes(
    ctx: Context,
    *,
    source: H3FullTransformer,
    lanes: tuple[LaneName, ...] | None = None,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> H3Lanes:
    """Convert one full-precision H3 checkpoint into the requested reviewed lanes.

    ``source`` is any checkpoint carrying the five H3 components at full precision: the
    converted upstream release or a ``bf16-full`` lane. ``lanes`` selects outputs (default
    all). Each lane is a row of
    ``lanes.LANES`` and commits to its own output slot, named after the lane. A component
    no lane treats is inherited by reference, so the conditioner and both VAEs stay
    byte-shared across lanes. A fresh identical request adopts every completed table and
    quantized tensor of a retained stopped attempt (a paused run) on the same worker.
    """
    requested = _requested(ALL_LANES if lanes is None else lanes)
    sources = {"dits": source, "shared": source}
    granted = _structures(ctx, sources)
    full_targets = _select_full_targets(ctx, sources, granted)
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    tables = _table_additions(sections)
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
                    granted[full_targets[component].source],
                ),
                dit_plan=dit_plan,
                # A bf16-full input already carries the normalised video VAE.
                allow_inert=component in NORMALISED_COMPONENTS,
            )
            for component, treatment in lane_treatments(LANES[name]).items()
        }
        for name in requested
    }

    configs = {
        "full": dual_full_config(sections),
        "adaln-pruned": dual_adaln_pruned_config(
            sections, _production_plan("fl2va"), _production_plan("ref2va")
        ),
    }
    orders = {
        "full": _full_order(sections, current.rows),
        "adaln-pruned": current.rows,
    }
    receipts: dict[str, dict[str, Any]] = {}
    source_bytes = new_bytes = reused = computed = 0
    quant_request = ArtifactQuantizationRequest(max_relative_frobenius=max_relative_frobenius)

    def treat(name: str, transaction: DerivedTransaction) -> None:
        nonlocal source_bytes, new_bytes, reused, computed
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
            new_bytes += stats.new_bytes_written
            reused += stats.reused_keys
            computed += stats.encoded_keys + stats.cast_keys
            tel.log(
                "weight fidelity",
                level="info",
                output_slot=name,
                component=component,
                treatment=selection.treatment.describe(),
                encoded_keys=stats.encoded_keys,
                reused_keys=stats.reused_keys,
                worst_relative_frobenius=stats.worst_relative_frobenius,
                cast_worst_relative_frobenius=stats.cast_worst_relative_frobenius,
            )

    with ExitStack() as stack:
        transactions = {
            name: stack.enter_context(
                ctx.output(name).open(
                    Derivation(
                        sources={alias: info.source for alias, info in granted.items()},
                        targets=_lane_targets(
                            LANES[name], sections, tables, full_targets, selections[name]
                        ),
                        configs={"model": Config("add")},
                        order=orders[LANES[name].modulation],
                    )
                )
            )
            for name in requested
        }
        active = {
            name: transaction
            for name, transaction in transactions.items()
            if transaction.receipt is None
        }
        # A lane needing neither a table pass nor an encoding settles first, so its
        # retention never depends on a later lane's.
        for name in [n for n in active if _settles_first(LANES[n], selections[n])]:
            transaction = active.pop(name)
            with tel.stage(name, overall_range=(0.0, 0.0)):
                treat(name, transaction)
            transaction.add_config("model", configs[LANES[name].modulation])
            receipts[name] = transaction.commit()

        pruned = {
            name: transaction
            for name, transaction in active.items()
            if LANES[name].modulation == "adaln-pruned"
        }
        table_share = 0.35 if pruned else 0.0
        if pruned:
            if not torch.cuda.is_available():
                raise UnsupportedInput(
                    "H3 AdaLN timestep tables need a CUDA worker; run on a GPU rental",
                    code="h3_tables_need_cuda",
                )
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            source_transaction = next(iter(pruned.values()))
            for task, overall_range in zip(
                SOURCE_SECTION, _bands(len(SOURCE_SECTION), 0.0, table_share), strict=True
            ):
                with tel.stage(f"timestep-table-{task}", overall_range=overall_range):
                    read, written, kept = _write_tables(
                        task, ctx, source_transaction, pruned, tel, overall_range
                    )
                    source_bytes += read
                    new_bytes += written
                    reused += kept
                    computed += len(tables[task]) - kept

        for name, overall_range in zip(
            list(active), _bands(len(active), table_share, 1.0), strict=True
        ):
            transaction = active[name]
            with tel.stage(name, overall_range=overall_range):
                treat(name, transaction)
                # Keep a finished checkpoint replayable if a later lane fails.
                transaction.add_config("model", configs[LANES[name].modulation])
                receipts[name] = transaction.commit()
            tel.progress(1.0, stage=f"commit-{name}", overall_fraction=overall_range[1])
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = transaction.commit()

    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric("h3.new_bytes", float(new_bytes), unit="bytes")
    tel.metric("h3.reused_tensors", float(reused))
    tel.metric("h3.computed_tensors", float(computed))
    return H3Lanes(
        [
            LaneArtifact(
                lane=name,
                modulation=LANES[name].modulation,
                treated_components=sorted(lane_treatments(LANES[name])),
                model=ctx.adopt_model(receipts[name]),
            )
            for name in requested
        ]
    )


app.job(lanes, name="lanes", weights=LANE_OUTPUTS)


def _retable_targets(
    pruned: SourceInspection,
    full: SourceInspection,
    sections: dict[str, dict[str, Any]],
    tables: Mapping[str, Mapping[str, Tensor]],
) -> tuple[dict[str, Target], dict[str, Target]]:
    """Declare the table bank derived from `full` and the retabled checkpoint from `pruned`.

    A transaction may read only the source components its targets derive from, so the
    modulation weights are read through the bank transaction (both DiTs from `full`, every
    other row dropped) while the retabled checkpoint inherits `pruned` by reference and
    replaces exactly its table keys. Refuses before any read unless `pruned` carries table
    rows and no dynamic modulation weights for both DiTs and `full` carries the exact
    modulation weights those rows are computed from.
    """
    present = {
        (component, key): tensor
        for component, rows in pruned.components.items()
        for key, tensor in rows.items()
    }
    full_present = {
        (component, key): tensor
        for component, rows in full.components.items()
        for key, tensor in rows.items()
    }
    components = set(pruned.components)
    if components != set(_full_targets()):
        raise ValueError(f"retable source components are {sorted(components)}")
    bank: dict[str, Target] = {}
    retabled = {
        component: Target(source="pruned", source_component=component)
        for component in components - set(TARGET_COMPONENT.values())
    }
    for task, section in SOURCE_SECTION.items():
        component = TARGET_COMPONENT[task]
        topology = H3Topology.from_config(sections[section])
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
        bank[component] = Target(
            source="full",
            source_component=component,
            drop=tuple(sorted(key for owner, key in full_present if owner == component)),
            add=tables[task],
        )
        retabled[component] = Target(
            source="pruned",
            source_component=component,
            drop=tuple(sorted(tables[task])),
            add=tables[task],
        )
    return bank, retabled


@invocable(memoize=True, memo_version="h3-retable/1", memo_dependencies=LANES_MEMO)
async def retable(
    ctx: Context,
    *,
    pruned: H3FullTransformer,
    full: H3FullTransformer,
    tel: Telemetry,
) -> RetableResult:
    """Recompute one AdaLN-pruned checkpoint's tables for the current plans.

    Every non-table tensor of `pruned` (BF16, FP8 or MXFP8 alike) is inherited by reference;
    only the modulation rows are read from `full` and recomputed, so a plan that adds
    schedules costs table bytes, never a requantization. The same rows also commit as a
    two-DiT table bank, the transaction the modulation reads are scoped to.
    """
    sections = parse_production_config(_asset("model-config.json"))
    plans = {task: _production_plan(task) for task in SOURCE_SECTION}
    pruned_config = dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])
    config = {"model": Config("add")}
    tables = _table_additions(sections)
    bank_targets, targets = _retable_targets(
        inspection(ctx, pruned), inspection(ctx, full), sections, tables
    )
    order = current_order(_asset("whole-order.json"))
    bank_order = tuple(
        row for row in order.rows if row[0] in bank_targets and row[1] in bank_targets[row[0]].add
    )
    source_bytes = written = 0
    receipts: dict[str, dict[str, Any]] = {}
    with ExitStack() as stack:
        bank = stack.enter_context(
            ctx.output("tables").open(
                Derivation(
                    sources={
                        name: info.source for name, info in _structures(ctx, {"full": full}).items()
                    },
                    targets=bank_targets,
                    configs=config,
                    order=bank_order,
                )
            )
        )
        retabled = stack.enter_context(
            ctx.output("adaln-pruned").open(
                Derivation(
                    sources={
                        name: info.source
                        for name, info in _structures(ctx, {"pruned": pruned}).items()
                    },
                    targets=targets,
                    configs=config,
                    order=order.rows,
                )
            )
        )
        transactions = {"adaln-pruned": retabled, "tables": bank}
        active = {name: t for name, t in transactions.items() if t.receipt is None}
        if active:
            if bank.receipt is not None:
                raise ValueError(
                    "the table bank was retained without its retabled checkpoint; "
                    "rerun under a new request identity"
                )
            if not torch.cuda.is_available():
                raise UnsupportedInput(
                    "H3 AdaLN timestep tables need a CUDA worker; run on a GPU rental",
                    code="h3_tables_need_cuda",
                )
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            for task, overall_range in (("fl2va", (0.0, 0.5)), ("ref2va", (0.5, 1.0))):
                with tel.stage(f"timestep-table-{task}", overall_range=overall_range):
                    read, emitted, _ = _write_tables(
                        task, ctx, bank, active, tel, overall_range, source="full"
                    )
                source_bytes += read
                written += emitted
            # The retabled checkpoint commits first: its retention never strands the bank.
            for name, transaction in active.items():
                transaction.add_config("model", pruned_config)
                receipts[name] = transaction.commit()
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = transaction.commit()
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric("h3.new_bytes", float(written), unit="bytes")
    return RetableResult(
        list(plans["fl2va"].steps),
        ctx.adopt_model(receipts["adaln-pruned"]),
        ctx.adopt_model(receipts["tables"]),
    )


app.job(
    retable,
    name="retable",
    weights=(
        WeightsOutput("adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
        WeightsOutput("tables", max_new_bytes=MAX_PRUNED_BYTES),
    ),
)


# Native output bounds include newly written construction and provenance configs.
app.job(
    _adaln_operations.select_adaln_weights,
    name="select-adaln-weights",
    weights=(WeightsOutput("model", 1 << 20),),
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
        WeightsOutput("model", 1 << 20),
        WeightsOutput("fl2va-weights", 1 << 20),
        WeightsOutput("ref2va-weights", 1 << 20),
    ),
)

app.job(
    assemble_full_artifact,
    name="assemble-full-artifact",
    weights=(WeightsOutput("model", 1 << 20),),
)

app.job(
    _adaln_operations.retable_adaln,
    name="retable-adaln",
    weights=(WeightsOutput("model", 1 << 20),),
)


def _source_modulation(source: SourceInspection, sections: Mapping[str, dict[str, Any]]) -> str:
    """Read the source's own modulation off its DiT rows rather than off a request field.

    An AdaLN-pruned checkpoint carries the timestep table rows and none of the dynamic
    modulation weights; a FULL one carries the modulation weights and no tables. Anything
    else is not a lane this producer emitted, and the restamp refuses rather than guessing.
    """
    present = {(component, key) for component, rows in source.components.items() for key in rows}
    verdicts: set[str] = set()
    for task, section in SOURCE_SECTION.items():
        component = TARGET_COMPONENT[task]
        topology = H3Topology.from_config(sections[section])
        keys = {key for owner, key in present if owner == component}
        if not keys:
            raise UnsupportedInput(f"restamp source has no {component}", code="h3_component_absent")
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


def _check_emitted_config(document: bytes, modulation: str, source: SourceInspection) -> None:
    """Validate row meanings and stored table dimensions before inheriting table bytes."""
    value = canonical_json.decode(document)
    fields = {"task", "modulation"} | ({"table_keys"} if modulation == "adaln-pruned" else set())
    tensors = {
        (component, key): tensor
        for component, rows in source.components.items()
        for key, tensor in rows.items()
    }
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
                    or tensor.parts != {"value": Part("bf16", shape)}
                ):
                    raise UnsupportedInput(
                        f"{component}.{key} stored dimensions/encoding differ from its table keys",
                        code="h3_restamp_table_layout",
                    )


@app.job(name="restamp", weights=(WeightsOutput("restamped", max_new_bytes=128 << 10),))
def restamp(
    ctx: Context,
    payload: ProductionRequest,
    lane: H3FullTransformer,
    tel: Telemetry,
) -> ModelArtifact:
    """Migrate the exact frame-stamped AdaLN plan without changing any tensor bytes."""
    del payload
    ctx.raise_if_cancelled()
    granted = inspection(ctx, lane)
    sections = parse_production_config(_asset("model-config.json"))
    modulation = _source_modulation(granted, sections)
    document = upgrade_legacy_table_config(
        granted.configs["model"], {task: _production_plan(task) for task in TASKS}
    )
    _check_emitted_config(document, modulation, granted)
    components = sorted(granted.components)
    if set(components) != set(_full_targets()):
        raise UnsupportedInput("restamp requires exactly the five H3 base components")
    with ctx.output("restamped").open(
        Derivation(
            sources={name: info.source for name, info in _structures(ctx, {"lane": lane}).items()},
            targets={component: Target("lane", component) for component in components},
            configs={"model": Config("add")},
            order=tuple(
                (component, key) for component, rows in granted.components.items() for key in rows
            ),
        )
    ) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.commit())
        transaction.add_config("model", document)
        receipt = transaction.commit()
    tel.metric("h3.source_bytes", 0, unit="bytes")
    return ctx.adopt_model(receipt)


from .turbo import turbo_lora  # noqa: E402

app.job(turbo_lora, name="turbo-lora", weights=(WeightsOutput("pdd8", 1 << 29),))
