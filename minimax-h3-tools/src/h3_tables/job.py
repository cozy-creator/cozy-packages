"""Typed functions producing the exact MiniMax H3 checkpoint variants and their AdaLN tables."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
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
    Weights,
    WeightsOutput,
    invocable,
)
from cozy_runtime.derive.quantization import (
    ArtifactQuantizationRequest,
    QuantizationStats,
    quantize_component_into,
)

from . import adaln_operations as _adaln_operations
from . import lanes as _lanes
from ._memo import LANES as LANES_MEMO
from .kernel import H3Topology, removed_keys, source_shapes, table_bytes, table_shapes
from .kernel import precompute_tables as compute_tables
from .lanes import (
    COMPONENT_MAX_NEW_BYTES,
    MAX_DIT_NEW_BYTES,
    LANES,
    NORMALISED_COMPONENTS,
    Lane,
    Selection,
    lane_max_new_bytes,
    lane_treatments,
    write_cast,
)
from .model_config import (
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
)
from .order import current_order
from .order import full_order as _full_order
from .plans import TimestepPlan, parse_declared_plan
from .quantization import reviewed_plan
from .metadata import (ComponentSelection, SourceLayout, TensorShape, PartShape,
                       PLAIN, SCHEMES, NATIVE_DTYPES, model_output, configure)
from cozy_runtime.derive import Checkpoint, Output, read_into
from .source import TARGET_COMPONENT, inspection
from .source import full_targets as _full_targets
from .source import select_full_targets as _select_full_targets
from .source import structures as _structures

app = App()

PLAIN_SPEC = PLAIN
FP8_SPEC = "fp8.rowwise"
INT8_SPEC = "int8.convrot"
NVFP4_SPEC = "nvfp4"

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
MAX_QUANTIZED_BYTES = 2 * MAX_DIT_NEW_BYTES + MAX_PRUNED_BYTES

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
#: Spelled as a literal, not as `COMPONENT_MAX_NEW_BYTES["vae"]`, because cr-114's
#: static reader folds the decorator from SOURCE over a closed vocabulary and refuses a
#: subscript: `describe` fails the whole package with `static_computed` rather than
#: guessing. The catalogue is still the authority — the equality below is checked at
#: import, so the two cannot drift; only the spelling is duplicated.
MAX_VIDEO_VAE_BYTES = 12 << 30
MAX_TEXT_ENCODER_BYTES = 56 << 30
MAX_QUANTIZED_LANE_BYTES = MAX_QUANTIZED_BYTES + MAX_VIDEO_VAE_BYTES + MAX_TEXT_ENCODER_BYTES
LANE_OUTPUTS = (
    WeightsOutput("bf16-full", max_new_bytes=MAX_FULL_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("bf16", max_new_bytes=MAX_PRUNED_BYTES + MAX_VIDEO_VAE_BYTES),
    WeightsOutput("fp8", max_new_bytes=MAX_QUANTIZED_LANE_BYTES),
)
LaneName = Literal["bf16-full", "bf16", "fp8"]


def _check_lane_outputs() -> None:
    for component, declared_bytes in (
        ("vae", MAX_VIDEO_VAE_BYTES),
        ("text_encoder", MAX_TEXT_ENCODER_BYTES),
    ):
        if COMPONENT_MAX_NEW_BYTES[component] != declared_bytes:
            raise ValueError(
                f"the declared {component} bound {declared_bytes} is not the catalogue's "
                f"{COMPONENT_MAX_NEW_BYTES[component]}"
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


class RetableResult(msgspec.Struct, frozen=True):
    steps: list[int]
    adaln_pruned: ModelArtifact
    tables: ModelArtifact


def _read_source(
    source: Checkpoint,
    component: str,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, int]:
    length = math.prod(shape) * dtype.itemsize
    raw = bytearray(length)
    view = memoryview(raw)
    for offset in range(0, length, SOURCE_READ_CHUNK):
        read_into(
            source,
            component,
            name,
            "",
            offset,
            view[offset : min(offset + SOURCE_READ_CHUNK, length)],
        )
    return torch.frombuffer(raw, dtype=dtype).reshape(shape), length


def _compute_table_parts(
    task: str,
    plan: TimestepPlan,
    topology: H3Topology,
    ctx: Context,
    source: Checkpoint,
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
        value, length = _read_source(source, source_component, name, dtype, shape)
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
        device=torch.device(str(ctx.device)),
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


def _table_additions(
    sections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, TensorShape]]:
    additions: dict[str, dict[str, TensorShape]] = {}
    for task, section in SOURCE_SECTION.items():
        plan = _production_plan(task)
        topology = H3Topology.from_config(sections[section])
        additions[task] = {
            key: TensorShape(
                logical_dtype="bf16",
                shape=shape,
                encoding=PLAIN_SPEC,
                parts={"value": PartShape("bf16", shape)},
            )
            for key, shape in sorted(table_shapes(topology, plan).items())
        }
    return additions


def _lane_targets(
    lane: Lane,
    sections: dict[str, dict[str, Any]],
    tables: Mapping[str, Mapping[str, TensorShape]],
    full_targets: Mapping[str, ComponentSelection],
    selections: Mapping[str, Selection],
) -> dict[str, ComponentSelection]:
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
    source_checkpoint: Checkpoint,
    transactions: Mapping[str, Output],
    tel: Telemetry,
    overall_range: tuple[float, float],
) -> tuple[int, int, int]:
    """Compute one task's tables into every transaction; (read, written, reused) counts."""
    sections = parse_production_config(_asset("model-config.json"))
    plan = _production_plan(task)
    topology = H3Topology.from_config(sections[SOURCE_SECTION[task]])
    component = TARGET_COMPONENT[task]
    completed = {
        name: {
            key for key in table_shapes(topology, plan)
            if transaction.has(component, key)
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
                transaction.put(component, name, "BF16", table_shapes(topology, plan)[name], raw)

    read, written = _compute_table_parts(
        task,
        plan,
        topology,
        ctx,
        source_checkpoint,
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
    transaction: Output,
    ctx: Context,
    tel: Telemetry,
    request: ArtifactQuantizationRequest,
    *,
    selection: Selection,
    source: Checkpoint,
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
    plan, encoding = selection.plan, SCHEMES[selection.treatment.encode]
    # The public encoder already parallelizes encoding and serializes bounded writes.
    # An outer shard pool would race Output's byte budget and component metadata.
    encoded = quantize_component_into(
        source, transaction, ctx, request, tel, encoding=encoding, plan=plan,
        component=component,
        logical_dtype=lambda tensor: selection.treatment.cast or tensor.logical_dtype,
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



def _produce_lanes(
    ctx: Context,
    tel: Telemetry,
    source: Weights[Any],
    lanes: tuple[LaneName, ...],
    max_relative_frobenius: float | None = None,
) -> dict[str, ModelArtifact]:
    """Convert one full-precision H3 checkpoint into the named reviewed lanes.

    ``source`` is any checkpoint carrying the five H3 components at full precision: the
    converted upstream release or a ``bf16-full`` lane. Each lane is a row of
    ``lanes.LANES`` and commits to its own output slot, named after the lane. A component
    no lane treats is inherited by reference, so the conditioner and both VAEs stay
    byte-shared across lanes. A fresh identical request adopts every completed table and
    quantized tensor of a canceled or failed attempt on the same machine.
    """
    requested = _requested(lanes)
    sources = {"dits": source, "shared": source}
    granted = _structures(ctx, sources)
    full_targets = _select_full_targets(ctx, sources, granted)
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    tables = _table_additions(sections)
    dit_plan = reviewed_plan()

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
    receipts: dict[str, ModelArtifact] = {}
    source_bytes = new_bytes = reused = computed = 0
    quant_request = ArtifactQuantizationRequest(max_relative_frobenius=max_relative_frobenius)

    def treat(name: str, transaction: Output) -> None:
        nonlocal source_bytes, new_bytes, reused, computed
        for component, selection in selections[name].items():
            stats = _treat(
                transaction,
                ctx,
                tel,
                quant_request,
                selection=selection,
                source=granted[full_targets[component].source].checkpoint,
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
                model_output(
                    ctx, name,
                    sources={alias: info.checkpoint for alias, info in granted.items()},
                    components=_lane_targets(
                        LANES[name], sections, tables, full_targets, selections[name]
                    ),
                    configs={"model": configs[LANES[name].modulation]},
                    order=orders[LANES[name].modulation],
                )
            )
            for name in requested
        }
        active = {
            name: transaction
            for name, transaction in transactions.items()
            if transaction.done is None
        }
        # A lane needing neither a table pass nor an encoding settles first, so its
        # retention never depends on a later lane's.
        for name in [n for n in active if _settles_first(LANES[n], selections[n])]:
            transaction = active.pop(name)
            with tel.stage(name, overall_range=(0.0, 0.0)):
                treat(name, transaction)
            receipts[name] = ctx.adopt_model(transaction)

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
            source_checkpoint = granted["dits"].checkpoint
            for task, overall_range in zip(
                SOURCE_SECTION, _bands(len(SOURCE_SECTION), 0.0, table_share), strict=True
            ):
                with tel.stage(f"timestep-table-{task}", overall_range=overall_range):
                    read, written, kept = _write_tables(
                        task, ctx, source_checkpoint, pruned, tel, overall_range
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
                # Commit each lane before the next, preserving interrupted-work replay.
                receipts[name] = ctx.adopt_model(transaction)
            tel.progress(1.0, stage=f"commit-{name}", overall_fraction=overall_range[1])
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = ctx.adopt_model(transaction)

    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric("h3.new_bytes", float(new_bytes), unit="bytes")
    tel.metric("h3.reused_tensors", float(reused))
    tel.metric("h3.computed_tensors", float(computed))
    return {name: receipts[name] for name in requested}


# One function per lane: Runtime requires a receipt for EVERY declared weights output, so a
# function declares exactly the lane it always produces.
@invocable(memo_version="h3-lane/1", memo_dependencies=LANES_MEMO)
async def bf16_full(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """The reviewed full-precision lane: source-only rows dropped, video VAE normalised."""
    return _produce_lanes(ctx, tel, source, ("bf16-full",))["bf16-full"]


@invocable(memo_version="h3-lane/1", memo_dependencies=LANES_MEMO)
async def bf16_pruned(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    """AdaLN-pruned BF16: 106 modulation rows per DiT replaced by timestep tables."""
    return _produce_lanes(ctx, tel, source, ("bf16",))["bf16"]


@invocable(memo_version="h3-lane/1", memo_dependencies=LANES_MEMO)
async def fp8_pruned(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """AdaLN-pruned with both DiTs row-wise FP8."""
    lane = _produce_lanes(ctx, tel, source, ("fp8",), max_relative_frobenius)
    return lane["fp8"]


app.entrypoint(
    bf16_full,
    name="bf16-full",
    weights=(WeightsOutput("bf16-full", max_new_bytes=MAX_FULL_BYTES + MAX_VIDEO_VAE_BYTES),),
    accelerator=False,
)
# The pruned lanes compute AdaLN tables on the GPU.
app.entrypoint(
    bf16_pruned,
    name="bf16-pruned",
    weights=(WeightsOutput("bf16", max_new_bytes=MAX_PRUNED_BYTES + MAX_VIDEO_VAE_BYTES),),
)
app.entrypoint(
    fp8_pruned,
    name="fp8-pruned",
    weights=(WeightsOutput("fp8", max_new_bytes=MAX_QUANTIZED_LANE_BYTES),),
)


#: `cozy model quantize <H3 checkpoint> --fp8|--int8|--nvfp4`: the quantized lanes' recipe over an
#: AdaLN-pruned checkpoint. Both DiTs encode through the reviewed plan and the text
#: conditioner's 350 decoder-layer linears through the structural selection minus
#: `TEXT_ENCODER_KEEP` (embeddings and the vision tower stay bf16). A recipe component the source still holds
#: plain is encoded; one that already carries this encoding is inherited; one carrying
#: another encoding refuses. Every other tensor, config and component inherits by reference.
#: nvfp4 takes the DiTs minus `NVFP4_DIT_KEEP`; its conditioner is int8-convrot, because the nvfp4 leaf is
#: W4A4 at prompt length (the published H3 conditioner is nvfp4 weights-only) and it runs once a request.
QUANTIZER_RECIPES: Mapping[str, Mapping[str, _lanes.Treatment]] = {
    encoding: {
        component: _lanes.Treatment(encode=encoding, keep=treatment.keep)
        for component, treatment in LANES["fp8"].components.items()
    }
    for encoding in ("fp8-rowwise/1", "int8-convrot256/1")
} | {
    "nvfp4-w4a4/1": {
        **{dit: _lanes.Treatment(encode="nvfp4-w4a4/1", keep=_lanes.NVFP4_DIT_KEEP)
           for dit in sorted(_lanes.DIT_COMPONENTS)},
        "text_encoder": _lanes.Treatment(encode="int8-convrot256/1", keep=_lanes.TEXT_ENCODER_KEEP),
    }
}
QUANTIZER_SPECS = {
    "fp8-rowwise/1": FP8_SPEC,
    "int8-convrot256/1": INT8_SPEC,
    "nvfp4-w4a4/1": NVFP4_SPEC,
}
#: Both DiTs and the conditioner at most; the transaction's own accounting is exact.
MAX_QUANTIZER_BYTES = 2 * MAX_DIT_NEW_BYTES + MAX_TEXT_ENCODER_BYTES


def _quantizer_selections(
    structure: SourceLayout, encoding: _lanes.Encoding
) -> list[Selection]:
    """The recipe's components this source still holds plain, resolved for `encoding`."""
    sections = parse_production_config(_asset("model-config.json"))
    for task, section in SOURCE_SECTION.items():
        component = TARGET_COMPONENT[task]
        if set(removed_keys(H3Topology.from_config(sections[section]))) & set(
            structure.components.get(component, {})
        ):
            raise UnsupportedInput(
                f"{component} still carries its AdaLN modulation rows; quantize an AdaLN-pruned "
                "lane (bf16 or fp8)",
                code="h3_quantize_unpruned",
            )
    dit_plan = reviewed_plan()
    selections = []
    for component, treatment in QUANTIZER_RECIPES[encoding].items():
        assert treatment.encode is not None
        encodings = {tensor.encoding for tensor in structure.components.get(component, {}).values()}
        if QUANTIZER_SPECS[treatment.encode] in encodings:
            continue
        foreign = encodings - {PLAIN_SPEC}
        if foreign:
            raise UnsupportedInput(
                f"{component} already carries another encoding ({sorted(foreign)}); quantize "
                f"a lane whose {component} is plain or already {treatment.encode}",
                code="h3_quantize_mixed",
            )
        selections.append(_lanes.select(component, treatment, structure, dit_plan=dit_plan))
    if not selections:
        raise UnsupportedInput(
            f"every component of H3's recipe already carries {encoding}",
            code="h3_quantize_nothing",
        )
    return selections


def _quantize_lane(
    ctx: Context,
    tel: Telemetry,
    source: Weights[Any],
    lane: str,
    encoding: _lanes.Encoding,
    max_relative_frobenius: float | None,
) -> ModelArtifact:
    structure = inspection(ctx, source)
    if set(structure.components) != set(_lanes.COMPONENTS):
        raise UnsupportedInput(
            f"source components are {sorted(structure.components)}, not H3's five",
            code="h3_quantize_source",
        )
    selections = _quantizer_selections(structure, encoding)
    targets = {
        component: ComponentSelection(source="source", source_component=component)
        for component in structure.components
    }
    for selection in selections:
        targets[selection.component] = _lanes.apply(targets[selection.component], selection)
    request = ArtifactQuantizationRequest(max_relative_frobenius=max_relative_frobenius)
    with model_output(
        ctx, lane, sources={"source": structure.checkpoint}, components=targets,
        configs=structure.configs,
        order=[(component, key) for component, rows in structure.components.items() for key in rows],
    ) as transaction:
        if transaction.done is not None:
            return ctx.adopt_model(transaction)
        for selection in selections:
            with tel.stage(f"{lane}-{selection.component}"):
                stats = _treat(transaction, ctx, tel, request, selection=selection, source=structure.checkpoint)
            tel.log(
                "weight fidelity",
                level="info",
                output_slot=lane,
                component=selection.component,
                treatment=selection.treatment.describe(),
                encoded_keys=stats.encoded_keys,
                reused_keys=stats.reused_keys,
                saturated_elements=stats.saturated_elements,
                worst_relative_frobenius=stats.worst_relative_frobenius,
                new_bytes=stats.new_bytes_written,
            )
        tel.log(
            "h3 quantize",
            lane=lane,
            encoded=",".join(selection.component for selection in selections),
            inherited=",".join(sorted(set(QUANTIZER_RECIPES[encoding]) - {s.component for s in selections})),
        )
        return ctx.adopt_model(transaction)


@invocable(memo_version="h3-quantize/1", memo_dependencies=LANES_MEMO)
async def fp8(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """H3's recipe at row-wise FP8 (`fp8-rowwise/1`): DiTs and text-conditioner linears."""
    return _quantize_lane(ctx, tel, source, "fp8", "fp8-rowwise/1", max_relative_frobenius)


@invocable(memo_version="h3-quantize/1", memo_dependencies=LANES_MEMO)
async def nvfp4(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """H3's recipe at `nvfp4-w4a4/1`: W4A4 on Blackwell's FP4 tensor cores, the same weights as int8."""
    return _quantize_lane(ctx, tel, source, "nvfp4", "nvfp4-w4a4/1", max_relative_frobenius)


@invocable(memo_version="h3-quantize/1", memo_dependencies=LANES_MEMO)
async def int8(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """H3's recipe at `int8-convrot256/1`: W8A8 on the int8 tensor cores, the same weights as fp8."""
    return _quantize_lane(ctx, tel, source, "int8", "int8-convrot256/1", max_relative_frobenius)


app.entrypoint(
    fp8,
    name="fp8",
    weights=(WeightsOutput("fp8", max_new_bytes=MAX_QUANTIZER_BYTES),),
    accelerator=False,
)
app.entrypoint(
    int8,
    name="int8",
    weights=(WeightsOutput("int8", max_new_bytes=MAX_QUANTIZER_BYTES),),
    accelerator=False,
)
app.entrypoint(
    nvfp4,
    name="nvfp4",
    weights=(WeightsOutput("nvfp4", max_new_bytes=MAX_QUANTIZER_BYTES),),
    accelerator=False,
)


#: The function that writes each lane: the pruned lanes' functions keep their operation names.
LANE_FUNCTIONS = {"bf16-full": "bf16-full", "bf16": "bf16-pruned", "fp8": "fp8-pruned"}


def _check_lane_functions() -> None:
    """Each lane's function declares exactly the catalogue's output for that lane."""
    declared = {output.name: output for output in LANE_OUTPUTS}
    for name in LANES:
        registered = app.get(LANE_FUNCTIONS[name]).weights_outputs
        if registered != (declared[name],):
            raise ValueError(f"lane function {name!r} declares {registered}, not {declared[name]}")


_check_lane_functions()


def _retable_targets(
    pruned: SourceLayout,
    full: SourceLayout,
    sections: dict[str, dict[str, Any]],
    tables: Mapping[str, Mapping[str, TensorShape]],
) -> tuple[dict[str, ComponentSelection], dict[str, ComponentSelection]]:
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
    bank: dict[str, ComponentSelection] = {}
    retabled = {
        component: ComponentSelection(source="pruned", source_component=component)
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
        bank[component] = ComponentSelection(
            source="full",
            source_component=component,
            drop=tuple(sorted(key for owner, key in full_present if owner == component)),
            add=tables[task],
        )
        retabled[component] = ComponentSelection(
            source="pruned",
            source_component=component,
            drop=tuple(sorted(tables[task])),
            add=tables[task],
        )
    return bank, retabled


@invocable(memo_version="h3-retable/1", memo_dependencies=LANES_MEMO)
async def retable(
    ctx: Context,
    *,
    pruned: Weights[Any],
    full: Weights[Any],
    tel: Telemetry,
) -> RetableResult:
    """Recompute one AdaLN-pruned checkpoint's tables for the current plans.

    Every non-table tensor of `pruned` (BF16, FP8 or int8 alike) is inherited by reference;
    only the modulation rows are read from `full` and recomputed, so a plan that adds
    schedules costs table bytes, never a requantization. The same rows also commit as a
    two-DiT table bank, the transaction the modulation reads are scoped to.
    """
    sections = parse_production_config(_asset("model-config.json"))
    plans = {task: _production_plan(task) for task in SOURCE_SECTION}
    pruned_config = dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])
    config = {"model": pruned_config}
    tables = _table_additions(sections)
    bank_targets, targets = _retable_targets(
        inspection(ctx, pruned), inspection(ctx, full), sections, tables
    )
    order = current_order(_asset("whole-order.json"))
    bank_order = tuple(
        row for row in order.rows if row[0] in bank_targets and row[1] in bank_targets[row[0]].add
    )
    source_bytes = written = 0
    receipts: dict[str, ModelArtifact] = {}
    with ExitStack() as stack:
        bank = stack.enter_context(
            model_output(
                ctx, "tables",
                sources={name: info.checkpoint for name, info in _structures(ctx, {"full": full}).items()},
                components=bank_targets, configs=config, order=bank_order,
            )
        )
        retabled = stack.enter_context(
            model_output(
                ctx, "adaln-pruned",
                sources={name: info.checkpoint for name, info in _structures(ctx, {"pruned": pruned}).items()},
                components=targets, configs=config, order=order.rows,
            )
        )
        transactions = {"adaln-pruned": retabled, "tables": bank}
        active = {name: t for name, t in transactions.items() if t.done is None}
        if active:
            if bank.done is not None:
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
                        task, ctx, inspection(ctx, full).checkpoint, active, tel, overall_range
                    )
                source_bytes += read
                written += emitted
            # The retabled checkpoint commits first: its retention never strands the bank.
            for name, transaction in active.items():
                receipts[name] = ctx.adopt_model(transaction)
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = ctx.adopt_model(transaction)
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric("h3.new_bytes", float(written), unit="bytes")
    return RetableResult(
        list(plans["fl2va"].steps),
        receipts["adaln-pruned"],
        receipts["tables"],
    )


app.entrypoint(
    retable,
    name="retable",
    weights=(
        WeightsOutput("adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
        WeightsOutput("tables", max_new_bytes=MAX_PRUNED_BYTES),
    ),
)


# Native output bounds include newly written construction and provenance configs.
app.entrypoint(
    _adaln_operations.select_adaln_weights,
    name="select-adaln-weights",
    weights=(WeightsOutput("model", 1 << 20),),
    accelerator=False,
)
app.entrypoint(
    _adaln_operations.compute_adaln_tables,
    name="compute-adaln-tables",
    weights=(WeightsOutput("model", MAX_TABLE_BYTES),),
)
app.entrypoint(
    _adaln_operations.apply_adaln,
    name="apply-adaln",
    weights=(
        WeightsOutput("model", 1 << 20),
        WeightsOutput("fl2va-weights", 1 << 20),
        WeightsOutput("ref2va-weights", 1 << 20),
    ),
    accelerator=False,
)

app.entrypoint(
    _adaln_operations.retable_adaln,
    name="retable-adaln",
    weights=(WeightsOutput("model", 1 << 20),),
    accelerator=False,
)


from .turbo import turbo_lora  # noqa: E402

app.entrypoint(turbo_lora, name="turbo-lora", weights=(WeightsOutput("pdd8", 1 << 29),))
