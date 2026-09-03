"""One typed job producing the four exact MiniMax H3 checkpoint variants."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import ExitStack
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

from .kernel import H3Topology, removed_keys, table_shapes
from .kernel import precompute_tables as compute_tables
from .model_config import (
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
    task_config_bytes,
)
from .order import current_order
from .plans import TimestepPlan, parse_declared_plan, parse_plan
from .source import official_full_specs, source_only_keys, text_source_only_keys

app = App()

PLAIN_SPEC = "sha256:71409d585d82c513f364ac730d2f573f0261828c4700ce653333a758d15e14bd"
FP8_SPEC = "sha256:8c86b26daec5bcd287401d9873b70aa3594855f97ac4d01cf75cc2bafd5d50b5"
MXFP8_SPEC = "sha256:796a50ed4d63e0b7cfc70a8cd797a5fe796bba64cb3143de4d35f059653f5d80"

TABLE_BYTES = 288_347_136
SOURCE_READ_CHUNK = 32 << 20
TARGET_COMPONENT = {"fl2va": "fl2va_dit", "ref2va": "ref2va_dit"}
TASK_MARKER = "__cozy_task"
TASK_VALUE = {"fl2va": b"\x00", "ref2va": b"\x01"}
MAX_NEW_BYTES = TABLE_BYTES + 619
MAX_FULL_BYTES = 64 << 10
MAX_PRUNED_BYTES = 2 * TABLE_BYTES + (128 << 10)
MAX_QUANTIZED_BYTES = 2 * MAX_OUTPUT_BYTES + MAX_PRUNED_BYTES
class H3FullTransformer(Model[object]):
    def load(self, loader: Any) -> None:
        del loader


class ProductionRequest(msgspec.Struct, forbid_unknown_fields=True):
    pass


class FourLaneResult(msgspec.Struct):
    bf16_full_receipt_digest: str
    bf16_adaln_pruned_receipt_digest: str
    fp8_adaln_pruned_receipt_digest: str
    mxfp8_adaln_pruned_receipt_digest: str
    replayed_outputs: int
    source_bytes_read_this_run: int
    quantized_keys_this_run: int


class TimestepTableResult(msgspec.Struct):
    task: str
    plan_digest: str
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool
    table_tensors: int
    table_bytes_this_run: int
    source_bytes_read_this_run: int


class AssemblyResult(msgspec.Struct):
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool


def _asset(name: str) -> bytes:
    return files(__package__).joinpath("assets", name).read_bytes()


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
    table_bytes = 0

    def read(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal source_bytes
        value, length = _read_source(
            transaction, source, source_component, name, dtype, shape
        )
        source_bytes += length
        return value

    def write(name: str, value: torch.Tensor) -> None:
        nonlocal table_bytes
        raw = (
            value.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .tobytes()
        )
        write_part(name, raw)
        table_bytes += len(raw)

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
    if table_bytes != TABLE_BYTES or len(table_shapes(topology, plan)) != 51:
        raise ValueError(
            f"{task} emitted {table_bytes} table bytes, expected {TABLE_BYTES}"
        )
    return source_bytes, table_bytes


def _generate_timestep_table(
    task: str,
    ctx: Context,
    payload: ProductionRequest,
    source: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> TimestepTableResult:
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    plan = parse_declared_plan(_asset(f"timestep-plan.{task}.json"))
    if plan.task != task:
        raise ValueError(f"package timestep plan is {plan.task!r}, expected {task!r}")
    source_section = "transformer" if task == "fl2va" else "transformer_ref"
    topology = H3Topology.from_config(sections[source_section])
    source_component = TARGET_COMPONENT[task]
    source_order = current_order(_asset("whole-order.json")).select(source_component)
    table_additions = {
        key: WeightsTensor(
            logical_dtype="bf16",
            shape=shape,
            encoding=PLAIN_SPEC,
            parts={"value": WeightsPart("bf16", shape)},
        )
        for key, shape in sorted(table_shapes(topology, plan).items())
    }
    additions = {
        **table_additions,
        TASK_MARKER: WeightsTensor(
            logical_dtype="u8",
            shape=(1,),
            encoding=PLAIN_SPEC,
            parts={"value": WeightsPart("u8", (1,))},
        ),
    }
    config_bytes = task_config_bytes(sections, plan)
    with artifacts.open(
        "pruned_dit",
        sources={"full": source},
        targets={
            "dit": WeightsTarget(
                source="full",
                source_component=source_component,
                drop=tuple(sorted(removed_keys(topology))),
                add=additions,
            )
        },
        configs={
            "model": WeightsConfig(
                data=config_bytes,
                length=len(config_bytes),
            )
        },
        order=(
            *(("dit", key) for _component, key in source_order.rows),
            ("dit", TASK_MARKER),
        ),
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return TimestepTableResult(
                plan.task,
                plan.digest,
                receipt.weights_transaction_id,
                receipt.tensorfs_receipt_digest,
                True,
                len(table_additions),
                0,
                0,
            )
        if not torch.cuda.is_available():
            raise ValueError("H3 timestep-table precompute requires a CUDA worker")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        def write_part(name: str, raw: bytes) -> None:
            transaction.add_part("dit", name, "value", raw)

        source_bytes, table_bytes = _compute_table_parts(
            task,
            plan,
            topology,
            ctx,
            transaction,
            "full",
            source_component,
            write_part,
            tel,
        )
        transaction.add_part("dit", TASK_MARKER, "value", TASK_VALUE[task])
        transaction.add_config("model", config_bytes)
        receipt = transaction.commit()
    return TimestepTableResult(
        plan.task,
        plan.digest,
        receipt.weights_transaction_id,
        receipt.tensorfs_receipt_digest,
        False,
        len(table_additions),
        table_bytes,
        source_bytes,
    )


@app.job(
    weights=(WeightsOutput("pruned_dit", max_new_bytes=MAX_NEW_BYTES),),
)
def generate_timestep_table_fl2va(
    ctx: Context,
    payload: ProductionRequest,
    source: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> TimestepTableResult:
    return _generate_timestep_table("fl2va", ctx, payload, source, artifacts, tel)


@app.job(
    weights=(WeightsOutput("pruned_dit", max_new_bytes=MAX_NEW_BYTES),),
)
def generate_timestep_table_ref2va(
    ctx: Context,
    payload: ProductionRequest,
    source: H3FullTransformer,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> TimestepTableResult:
    return _generate_timestep_table("ref2va", ctx, payload, source, artifacts, tel)


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
    receipt = artifacts.derive(
        "model",
        sources={"dits": dits, "shared": shared},
        targets=_full_targets(),
        configs={
            "model": WeightsConfig(data=config, length=len(config))
        },
        order=_full_order(sections, current.rows),
    )
    return _assembly_result(receipt)


def _dual_targets() -> dict[str, WeightsTarget]:
    return {
        "fl2va_dit": WeightsTarget(
            source="fl2va", source_component="dit", drop=(TASK_MARKER,)
        ),
        "ref2va_dit": WeightsTarget(
            source="ref2va", source_component="dit", drop=(TASK_MARKER,)
        ),
        **{
            component: WeightsTarget(source="shared", source_component=component)
            for component in ("text_encoder", "video_vae", "audio_vae")
        },
    }


def _require_task_markers(markers: Mapping[str, bytes]) -> None:
    for source, expected in TASK_VALUE.items():
        if markers.get(source) != expected:
            raise ValueError(
                f"{source} input does not carry its exact task marker; "
                "FL2VA and Ref2VA inputs may not be swapped"
            )


@app.job(
    weights=(WeightsOutput("model", max_new_bytes=128 << 10),),
)
def assemble_dual(
    payload: ProductionRequest,
    fl2va: H3FullTransformer,
    ref2va: H3FullTransformer,
    shared: H3FullTransformer,
    artifacts: WeightsSink,
) -> AssemblyResult:
    del payload
    sections = parse_production_config(_asset("model-config.json"))
    fl_plan = parse_plan(_asset("timestep-plan.fl2va.json"), task="fl2va")
    ref_plan = parse_plan(_asset("timestep-plan.ref2va.json"), task="ref2va")
    config = dual_adaln_pruned_config(sections, fl_plan, ref_plan)
    order = current_order(_asset("whole-order.json"))
    with artifacts.open(
        "model",
        sources={"fl2va": fl2va, "ref2va": ref2va, "shared": shared},
        targets=_dual_targets(),
        configs={
            "model": WeightsConfig(data=config, length=len(config))
        },
        order=order.rows,
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return _assembly_result(receipt)
        markers: dict[str, bytes] = {}
        for source in TASK_VALUE:
            observed = bytearray(1)
            transaction.source_read_into(
                source, "dit", TASK_MARKER, "value", 0, observed
            )
            markers[source] = bytes(observed)
        _require_task_markers(markers)
        transaction.add_config("model", config)
        receipt = transaction.commit()
    return _assembly_result(receipt)


def _table_additions(
    sections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, WeightsTensor]]:
    additions: dict[str, dict[str, WeightsTensor]] = {}
    for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
        plan = parse_declared_plan(_asset(f"timestep-plan.{task}.json"))
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
    quantization: ArtifactQuantizationPlan | None = None,
    encoding: str | None = None,
) -> dict[str, WeightsTarget]:
    encoded = (
        quantization_additions(encoding, quantization, "dit")
        if encoding is not None and quantization is not None
        else {}
    )
    targets: dict[str, WeightsTarget] = {}
    for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
        topology = H3Topology.from_config(sections[section])
        component = TARGET_COMPONENT[task]
        additions = {**tables[task], **encoded}
        drop = tuple(
            sorted(
                set(source_only_keys())
                | set(removed_keys(topology))
                | set(encoded)
            )
        )
        targets[component] = WeightsTarget(
            source="dits",
            source_component=component,
            drop=drop,
            add=additions,
        )
    targets.update(
        {
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
    )
    return targets


def _write_tables(
    task: str,
    ctx: Context,
    source_transaction: WeightsTransaction,
    transactions: Mapping[str, WeightsTransaction],
    tel: Telemetry,
    overall_range: tuple[float, float],
) -> int:
    sections = parse_production_config(_asset("model-config.json"))
    plan = parse_declared_plan(_asset(f"timestep-plan.{task}.json"))
    if plan.task != task:
        raise ValueError(f"package timestep plan is {plan.task!r}, expected {task!r}")
    section = "transformer" if task == "fl2va" else "transformer_ref"
    topology = H3Topology.from_config(sections[section])
    component = TARGET_COMPONENT[task]

    def write_part(name: str, raw: bytes) -> None:
        for transaction in transactions.values():
            transaction.add_part(component, name, "value", raw)

    source_bytes, _table_bytes = _compute_table_parts(
        task,
        plan,
        topology,
        ctx,
        source_transaction,
        "dits",
        component,
        write_part,
        tel,
        overall_range,
    )
    return source_bytes


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
    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))

    full_config = dual_full_config(sections)
    tel.progress(0.0, stage="bf16-full", overall_fraction=0.0)
    with tel.stage("bf16-full", overall_range=(0.00, 0.15)):
        full = artifacts.derive(
            "bf16-full",
            sources=sources,
            targets=_full_targets(),
            configs={
                "model": WeightsConfig(
                    data=full_config,
                    length=len(full_config),
                )
            },
            order=_full_order(sections, current.rows),
        )
    tel.progress(1.0, stage="bf16-full", overall_fraction=0.15)

    fl_plan = parse_plan(_asset("timestep-plan.fl2va.json"), task="fl2va")
    ref_plan = parse_plan(_asset("timestep-plan.ref2va.json"), task="ref2va")
    pruned_config = dual_adaln_pruned_config(sections, fl_plan, ref_plan)
    config = {
        "model": WeightsConfig(
            data=pruned_config,
            length=len(pruned_config),
        )
    }
    tables = _table_additions(sections)
    quantization = prepare_quantization(h3_quantization_plan())
    targets = {
        "bf16-adaln-pruned": _pruned_targets(sections, tables),
        "fp8-adaln-pruned": _pruned_targets(
            sections, tables, quantization, "fp8-rowwise/1"
        ),
        "mxfp8-adaln-pruned": _pruned_targets(
            sections, tables, quantization, "mxfp8/1"
        ),
    }
    receipts: dict[str, WeightsReceipt] = {"bf16-full": full}
    stats: list[QuantizationStats] = []
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
                    )

        quant_request = ArtifactQuantizationRequest()
        for name, encoding, overall_range in (
            ("fp8-adaln-pruned", "fp8-rowwise/1", (0.35, 0.60)),
            ("mxfp8-adaln-pruned", "mxfp8/1", (0.60, 0.85)),
        ):
            transaction = active.get(name)
            if transaction is None:
                continue
            with tel.stage(name, overall_range=overall_range):
                for component in TARGET_COMPONENT.values():
                    stats.append(
                        quantize_component_into(
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
                    )

        commit_ranges = {
            "bf16-adaln-pruned": (0.85, 0.90),
            "fp8-adaln-pruned": (0.90, 0.95),
            "mxfp8-adaln-pruned": (0.95, 1.00),
        }
        for name, transaction in active.items():
            overall_range = commit_ranges[name]
            tel.progress(0.0, stage=f"commit-{name}", overall_fraction=overall_range[0])
            transaction.add_config("model", pruned_config)
            receipts[name] = transaction.commit()
            tel.progress(1.0, stage=f"commit-{name}", overall_fraction=overall_range[1])
        for name, transaction in transactions.items():
            if name not in receipts:
                receipts[name] = _receipt(transaction)

    source_bytes += sum(stat.source_bytes_read for stat in stats)
    tel.metric("h3.source_bytes", float(source_bytes), unit="bytes")
    tel.metric(
        "h3.quantized_bytes",
        float(sum(stat.new_bytes_written for stat in stats)),
        unit="bytes",
    )
    return FourLaneResult(
        bf16_full_receipt_digest=receipts["bf16-full"].tensorfs_receipt_digest,
        bf16_adaln_pruned_receipt_digest=receipts[
            "bf16-adaln-pruned"
        ].tensorfs_receipt_digest,
        fp8_adaln_pruned_receipt_digest=receipts[
            "fp8-adaln-pruned"
        ].tensorfs_receipt_digest,
        mxfp8_adaln_pruned_receipt_digest=receipts[
            "mxfp8-adaln-pruned"
        ].tensorfs_receipt_digest,
        replayed_outputs=sum(receipt.replayed for receipt in receipts.values()),
        source_bytes_read_this_run=source_bytes,
        quantized_keys_this_run=sum(stat.encoded_keys for stat in stats),
    )
