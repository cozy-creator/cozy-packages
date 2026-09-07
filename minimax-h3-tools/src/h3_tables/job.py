"""Typed jobs producing the exact MiniMax H3 checkpoint variants and their AdaLN tables."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from importlib.resources import files
from typing import Any

import msgspec
import torch
from cozy_runtime.author import (
    App,
    Context,
    Model,
    Telemetry,
    WeightsConfig,
    WeightsOutput,
    WeightsPart,
    WeightsReceipt,
    WeightsSink,
    WeightsSource,
    WeightsTarget,
    WeightsTensor,
    WeightsTransaction,
)
from cozy_runtime.derive.quantization import (
    MAX_OUTPUT_BYTES,
    ArtifactQuantizationPlan,
    ArtifactQuantizationRequest,
    QuantizationStats,
    h3_quantization_plan,
    prepare_quantization,
    quantization_additions,
    quantize_component_into,
)

from .kernel import H3Topology, removed_keys, source_shapes, table_bytes, table_shapes
from .kernel import precompute_tables as compute_tables
from .model_config import (
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
)
from .order import current_order
from .plans import TimestepPlan, parse_declared_plan
from .source import official_full_specs, source_only_keys, text_source_only_keys

app = App()

PLAIN_SPEC = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
FP8_SPEC = "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890"
MXFP8_SPEC = "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd"

SOURCE_READ_CHUNK = 32 << 20
TARGET_COMPONENT = {"fl2va": "fl2va_dit", "ref2va": "ref2va_dit"}
SOURCE_SECTION = {"fl2va": "transformer", "ref2va": "transformer_ref"}
TORCH_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32}


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


def _production_plan(task: str) -> TimestepPlan:
    plan = parse_declared_plan(_asset(f"timestep-plan.{task}.json"))
    if plan.task != task:
        raise ValueError(f"package timestep plan is {plan.task!r}, expected {task!r}")
    return plan


def _table_budget() -> int:
    """The exact table bytes one task's plan occupies, the larger task governing."""
    sections = parse_production_config(_asset("model-config.json"))
    return max(
        table_bytes(H3Topology.from_config(sections[section]), _production_plan(task))
        for task, section in SOURCE_SECTION.items()
    )


TABLE_BYTES = _table_budget()
MAX_FULL_BYTES = 64 << 10
MAX_PRUNED_BYTES = 2 * TABLE_BYTES + (128 << 10)
MAX_QUANTIZED_BYTES = 2 * MAX_OUTPUT_BYTES + MAX_PRUNED_BYTES


class H3FullTransformer(Model[object]):
    def load(self, loader: Any) -> None:
        del loader


class ProductionRequest(msgspec.Struct, forbid_unknown_fields=True):
    pass


class WeightFidelity(msgspec.Struct):
    output_slot: str
    component: str
    stats: QuantizationStats


class FourLaneResult(msgspec.Struct):
    bf16_full_receipt_digest: str
    bf16_adaln_pruned_receipt_digest: str
    fp8_adaln_pruned_receipt_digest: str
    mxfp8_adaln_pruned_receipt_digest: str
    replayed_outputs: int
    source_bytes_read_this_run: int
    quantized_keys_this_run: int
    weight_fidelity_this_run: list[WeightFidelity]


class AssemblyResult(msgspec.Struct):
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool


class RetableResult(msgspec.Struct):
    source_checkpoint: str
    steps: list[int]
    tensorfs_receipt_digest: str
    table_bank_receipt_digest: str
    replayed: bool
    table_tensors: int
    table_bytes_this_run: int
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
) -> tuple[int, int]:
    source_bytes = 0
    written = 0

    def read(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal source_bytes
        value, length = _read_source(
            transaction, source, source_component, name, dtype, shape
        )
        source_bytes += length
        return value

    def write(name: str, value: torch.Tensor) -> None:
        nonlocal written
        raw = (
            value.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .tobytes()
        )
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
    )
    expected = table_bytes(topology, plan)
    if written != expected or len(table_shapes(topology, plan)) != topology.num_layers + 1:
        raise ValueError(f"{task} emitted {written} table bytes, expected {expected}")
    return source_bytes, written


def _full_order(
    sections: dict[str, dict[str, Any]], current: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    fl = tuple(
        ("fl2va_dit", key) for key in official_full_specs(sections["transformer"])
    )
    ref = tuple(
        ("ref2va_dit", key)
        for key in official_full_specs(sections["transformer_ref"])
    )
    shared = tuple(row for row in current if row[0] not in TARGET_COMPONENT.values())
    if len(fl) != 638 or len(ref) != 638 or len(shared) != 2692:
        raise ValueError(
            f"full H3 order is {len(fl)}/{len(ref)}/{len(shared)}, expected 638/638/2692"
        )
    return (*fl, *ref, *shared)


def _full_targets() -> dict[str, WeightsTarget]:
    return {
        "fl2va_dit": WeightsTarget(
            source="dits",
            source_component="fl2va_dit",
            drop=source_only_keys(),
        ),
        "ref2va_dit": WeightsTarget(
            source="dits",
            source_component="ref2va_dit",
            drop=source_only_keys(),
        ),
        "text_encoder": WeightsTarget(
            source="shared",
            source_component="text_encoder",
            drop=text_source_only_keys(),
        ),
        "video_vae": WeightsTarget(
            source="shared", source_component="video_vae"
        ),
        "audio_vae": WeightsTarget(
            source="shared", source_component="audio_vae"
        ),
    }


def _select_full_targets(
    artifacts: WeightsSink, sources: Mapping[str, H3FullTransformer]
) -> dict[str, WeightsTarget]:
    """Drop source-only rows that remain in these exact granted checkpoints."""
    keys: dict[str, set[tuple[str, str]]] = {}
    for source in sources.values():
        if source.checkpoint_ref not in keys:
            keys[source.checkpoint_ref] = {
                (tensor.component, tensor.key) for tensor in artifacts.structure(source).tensors
            }
    targets = _full_targets()
    return {
        component: replace(
            target,
            drop=tuple(
                key
                for key in target.drop
                if (target.source_component, key) in keys[sources[target.source].checkpoint_ref]
            ),
        )
        for component, target in targets.items()
    }



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
        configs={
            "model": WeightsConfig(data=config, length=len(config))
        },
        order=_full_order(sections, current.rows),
    )
    return _assembly_result(receipt)


def _table_additions(
    sections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, WeightsTensor]]:
    additions: dict[str, dict[str, WeightsTensor]] = {}
    for task, section in SOURCE_SECTION.items():
        plan = _production_plan(task)
        topology = H3Topology.from_config(sections[section])
        additions[task] = {
            key: WeightsTensor(
                logical_dtype="bf16",
                shape=shape,
                encoding=PLAIN_SPEC,
                parts={"value": WeightsPart("bf16", shape)},
            )
            for key, shape in sorted(table_shapes(topology, plan).items())
        }
    return additions


def _pruned_targets(
    sections: dict[str, dict[str, Any]],
    tables: Mapping[str, Mapping[str, WeightsTensor]],
    full_targets: Mapping[str, WeightsTarget],
    quantization: ArtifactQuantizationPlan | None = None,
    encoding: str | None = None,
) -> dict[str, WeightsTarget]:
    encoded = (
        quantization_additions(encoding, quantization, "dit")
        if encoding is not None and quantization is not None
        else {}
    )
    targets = dict(full_targets)
    for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
        topology = H3Topology.from_config(sections[section])
        component = TARGET_COMPONENT[task]
        additions = {**tables[task], **encoded}
        drop = tuple(
            sorted(set(full_targets[component].drop) | set(removed_keys(topology)) | set(encoded))
        )
        targets[component] = replace(
            full_targets[component],
            drop=drop,
            add=additions,
        )
    return targets


def _write_tables(
    task: str,
    ctx: Context,
    source_transaction: WeightsTransaction,
    transactions: Mapping[str, WeightsTransaction],
    tel: Telemetry,
    overall_range: tuple[float, float],
    source: str = "dits",
) -> tuple[int, int]:
    sections = parse_production_config(_asset("model-config.json"))
    plan = _production_plan(task)
    topology = H3Topology.from_config(sections[SOURCE_SECTION[task]])
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
    )


def _receipt(transaction: WeightsTransaction) -> WeightsReceipt:
    receipt = transaction.receipt if transaction.replayed else transaction.commit()
    assert receipt is not None
    return receipt


@app.job(
    name="four-lane",
    weights=(
        WeightsOutput("bf16-full", max_new_bytes=MAX_FULL_BYTES),
        WeightsOutput("bf16-adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
        WeightsOutput("fp8-adaln-pruned", max_new_bytes=MAX_QUANTIZED_BYTES),
        WeightsOutput("mxfp8-adaln-pruned", max_new_bytes=MAX_QUANTIZED_BYTES),
    ),
    # Default [bindings] for these slots arrive once the minimax-h3 model repo
    # exists on the hub (H3 ingest, se-022/th-109 residue); until then no slot
    # facts derive at publish (cr-077) and these slots bind per-invocation like
    # this package's other jobs.
)
def four_lane(
    ctx: Context,
    payload: ProductionRequest,
    dits: H3FullTransformer,
    shared: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> FourLaneResult:
    """Produce full, AdaLN-pruned, FP8, and MXFP8 checkpoints in one attempt."""
    del payload
    sources = {"dits": dits, "shared": shared}
    full_targets = _select_full_targets(artifacts, sources)
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))

    full_config = dual_full_config(sections)
    tel.progress(0.0, stage="bf16-full", overall_fraction=0.0)
    with tel.stage("bf16-full", overall_range=(0.00, 0.15)):
        full = artifacts.derive(
            "bf16-full",
            sources=sources,
            targets=full_targets,
            configs={
                "model": WeightsConfig(
                    data=full_config,
                    length=len(full_config),
                )
            },
            order=_full_order(sections, current.rows),
        )
    tel.progress(1.0, stage="bf16-full", overall_fraction=0.15)

    pruned_config = dual_adaln_pruned_config(
        sections, _production_plan("fl2va"), _production_plan("ref2va")
    )
    config = {
        "model": WeightsConfig(
            data=pruned_config,
            length=len(pruned_config),
        )
    }
    tables = _table_additions(sections)
    quantization = prepare_quantization(h3_quantization_plan())
    targets = {
        "bf16-adaln-pruned": _pruned_targets(sections, tables, full_targets),
        "fp8-adaln-pruned": _pruned_targets(
            sections, tables, full_targets, quantization, "fp8-rowwise/1"
        ),
        "mxfp8-adaln-pruned": _pruned_targets(
            sections, tables, full_targets, quantization, "mxfp8/1"
        ),
    }
    receipts: dict[str, WeightsReceipt] = {"bf16-full": full}
    fidelity: list[WeightFidelity] = []
    source_bytes = 0

    with ExitStack() as stack:
        transactions = {
            name: stack.enter_context(
                artifacts.open(
                    name,
                    sources=sources,
                    targets=target,
                    configs=config,
                    order=current.rows,
                )
            )
            for name, target in targets.items()
        }
        active = {
            name: transaction
            for name, transaction in transactions.items()
            if not transaction.replayed
        }
        if active:
            if not torch.cuda.is_available():
                raise ValueError("H3 four-lane production requires a CUDA worker")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            source_transaction = next(iter(active.values()))
            for task, overall_range in (
                ("fl2va", (0.15, 0.25)),
                ("ref2va", (0.25, 0.35)),
            ):
                with tel.stage(f"timestep-table-{task}", overall_range=overall_range):
                    source_bytes += _write_tables(
                        task, ctx, source_transaction, active, tel, overall_range
                    )[0]

        quant_request = ArtifactQuantizationRequest()
        for name, encoding, overall_range in (
            ("bf16-adaln-pruned", None, (0.35, 0.40)),
            ("fp8-adaln-pruned", "fp8-rowwise/1", (0.40, 0.70)),
            ("mxfp8-adaln-pruned", "mxfp8/1", (0.70, 1.00)),
        ):
            transaction = active.get(name)
            if transaction is None:
                continue
            with tel.stage(name, overall_range=overall_range):
                if encoding is not None:
                    for component in TARGET_COMPONENT.values():
                        stats = quantize_component_into(
                            transaction,
                            ctx,
                            quant_request,
                            tel,
                            encoding=encoding,
                            plan=quantization,
                            component="dit",
                            source="dits",
                            source_component=component,
                            target_component=component,
                        )
                        fidelity.append(WeightFidelity(name, component, stats))
                        tel.log(
                            "weight fidelity",
                            level="info",
                            output_slot=name,
                            component=component,
                            **msgspec.to_builtins(stats),
                        )
                # Keep a finished checkpoint replayable if a later lane fails.
                transaction.add_config("model", pruned_config)
                receipts[name] = transaction.commit()
            tel.progress(1.0, stage=f"commit-{name}", overall_fraction=overall_range[1])
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = _receipt(transaction)

    measured = [row.stats for row in fidelity]
    source_bytes += sum(stat.source_bytes_read for stat in measured)
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric(
        "h3.quantized_bytes",
        float(sum(stat.new_bytes_written for stat in measured)),
        unit="bytes",
    )
    return FourLaneResult(
        bf16_full_receipt_digest=receipts["bf16-full"].tensorfs_receipt_digest,
        bf16_adaln_pruned_receipt_digest=receipts["bf16-adaln-pruned"].tensorfs_receipt_digest,
        fp8_adaln_pruned_receipt_digest=receipts["fp8-adaln-pruned"].tensorfs_receipt_digest,
        mxfp8_adaln_pruned_receipt_digest=receipts["mxfp8-adaln-pruned"].tensorfs_receipt_digest,
        replayed_outputs=sum(receipt.replayed for receipt in receipts.values()),
        source_bytes_read_this_run=source_bytes,
        quantized_keys_this_run=sum(stat.encoded_keys for stat in measured),
        weight_fidelity_this_run=fidelity,
    )


def _retable_targets(
    pruned: WeightsSource,
    full: WeightsSource,
    sections: dict[str, dict[str, Any]],
    tables: Mapping[str, Mapping[str, WeightsTensor]],
) -> tuple[dict[str, WeightsTarget], dict[str, WeightsTarget]]:
    """Declare the table bank derived from `full` and the retabled checkpoint from `pruned`.

    A transaction may read only the source components its targets derive from, so the
    modulation weights are read through the bank transaction (both DiTs from `full`, every
    other row dropped) while the retabled checkpoint inherits `pruned` by reference and
    replaces exactly its table keys. Refuses before any read unless `pruned` carries table
    rows and no dynamic modulation weights for both DiTs and `full` carries the exact
    modulation weights those rows are computed from.
    """
    present = {(tensor.component, tensor.key): tensor for tensor in pruned.tensors}
    full_present = {(tensor.component, tensor.key): tensor for tensor in full.tensors}
    components = {tensor.component for tensor in pruned.tensors}
    if components != set(_full_targets()):
        raise ValueError(f"retable source components are {sorted(components)}")
    bank: dict[str, WeightsTarget] = {}
    retabled = {
        component: WeightsTarget(source="pruned", source_component=component)
        for component in components - set(TARGET_COMPONENT.values())
    }
    for task, section in SOURCE_SECTION.items():
        component = TARGET_COMPONENT[task]
        topology = H3Topology.from_config(sections[section])
        keys = {key for owner, key in present if owner == component}
        if not set(tables[task]) <= keys or set(removed_keys(topology)) & keys:
            raise ValueError(
                f"{component} is not an AdaLN-pruned source carrying table rows only"
            )
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
    return bank, retabled


@app.job(
    name="retable",
    weights=(
        WeightsOutput("adaln-pruned", max_new_bytes=MAX_PRUNED_BYTES),
        WeightsOutput("tables", max_new_bytes=MAX_PRUNED_BYTES),
    ),
)
def retable(
    ctx: Context,
    payload: ProductionRequest,
    full: H3FullTransformer,
    pruned: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> RetableResult:
    """Recompute one AdaLN-pruned checkpoint's tables for the current plans.

    Every non-table tensor of `pruned` (BF16, FP8 or MXFP8 alike) is inherited by reference;
    only the modulation rows are read from `full` and recomputed, so a plan that adds
    schedules costs table bytes, never a requantization. The same rows also commit as a
    two-DiT table bank, the transaction the modulation reads are scoped to.
    """
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    plans = {task: _production_plan(task) for task in SOURCE_SECTION}
    pruned_config = dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])
    config = {"model": WeightsConfig(data=pruned_config, length=len(pruned_config))}
    tables = _table_additions(sections)
    bank_targets, targets = _retable_targets(
        artifacts.structure(pruned), artifacts.structure(full), sections, tables
    )
    order = current_order(_asset("whole-order.json"))
    bank_order = tuple(
        row
        for row in order.rows
        if row[0] in bank_targets and row[1] in bank_targets[row[0]].add
    )
    source_bytes = written = 0
    receipts: dict[str, WeightsReceipt] = {}
    with ExitStack() as stack:
        bank = stack.enter_context(
            artifacts.open(
                "tables", sources={"full": full}, targets=bank_targets, configs=config,
                order=bank_order,
            )
        )
        retabled = stack.enter_context(
            artifacts.open(
                "adaln-pruned", sources={"pruned": pruned}, targets=targets, configs=config,
                order=order.rows,
            )
        )
        transactions = {"adaln-pruned": retabled, "tables": bank}
        active = {name: t for name, t in transactions.items() if not t.replayed}
        if active:
            if bank.replayed:
                raise ValueError(
                    "the table bank was retained without its retabled checkpoint; "
                    "rerun under a new request identity"
                )
            if not torch.cuda.is_available():
                raise ValueError("H3 timestep-table precompute requires a CUDA worker")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            for task, overall_range in (("fl2va", (0.0, 0.5)), ("ref2va", (0.5, 1.0))):
                with tel.stage(f"timestep-table-{task}", overall_range=overall_range):
                    read, emitted = _write_tables(
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
                receipts[name] = _receipt(transaction)
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    return RetableResult(
        pruned.checkpoint_ref,
        list(plans["fl2va"].steps),
        receipts["adaln-pruned"].tensorfs_receipt_digest,
        receipts["tables"].tensorfs_receipt_digest,
        receipts["adaln-pruned"].replayed,
        sum(len(rows) for rows in tables.values()),
        written,
        source_bytes,
    )
