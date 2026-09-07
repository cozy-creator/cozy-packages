"""Exact FC1 row repair for the four checkpoints produced by the old H3 converter."""

from __future__ import annotations

import math
from collections.abc import Mapping

import msgspec
from cozy_runtime.author import (
    Model,
    Telemetry,
    WeightsConfig,
    WeightsPart,
    WeightsSink,
    WeightsSourceTensor,
    WeightsTarget,
    WeightsTensor,
    WeightsTransaction,
)

# A half swap is an involution. Only these recorded bad roots may enter this
# migration; a repaired or independently converted source must never be swapped again.
SOURCES = {
    "sha256:3f6c224a010fbff36adce0f19a8b866f4f1739a1d30e2202c140ba994dc0b4df": "plain",
    "sha256:9b74399051e2ed3ca4df21ddbbbc8b2d1548ef790b0960fdd9ad392ac64a23cd": "plain",
    "sha256:5e24c4cbf15c542410455301565aecc72586ce9d8c4d553fa2530238009c1a87": "fp8",
    "sha256:debacaad38d01a730722b200f2b03ebe3bc75bb581bca9485114befed70fc3ec": "mxfp8",
}
# Exact encoding identities present in those fixed source roots; the native proof
# checks these immutable references against TensorFS's own installed seed set.
ENCODINGS = {
    "plain": "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f",
    "fp8": "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890",
    "mxfp8": "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd",
}
COMPONENTS = ("fl2va_dit", "ref2va_dit")
FC1_KEYS = frozenset(
    [f"transformer_blocks.{i}.ff.net.0.proj.weight" for i in range(50)]
    + [f"token_refiner.refiner_blocks.{i}.ff.net.0.proj.weight" for i in range(2)]
)
DTYPE_BYTES = {"bf16": 2, "f32": 4, "f8_e4m3fn": 1, "u8": 1}
READ_BYTES = 32 << 20
MAX_ROLE_BYTES = 28672 * 5376 * 2
MAX_NEW_BYTES = 32 << 30


class RepairResult(msgspec.Struct):
    source_checkpoint: str
    tensorfs_receipt_digest: str
    weights_transaction_id: str
    repaired_tensors: int
    source_bytes_read_this_run: int
    replayed_parts: int
    replayed: bool


def _declaration(
    tensor: WeightsSourceTensor, variant: str, encodings: Mapping[str, str]
) -> WeightsTensor:
    if (
        tensor.logical_dtype != "bf16"
        or len(tensor.shape) != 2
        or min(tensor.shape) <= 0
        or tensor.shape[0] % 2
    ):
        raise ValueError(f"unexpected H3 FC1 geometry: {tensor.component}/{tensor.key}")
    parts = {part.name: WeightsPart(part.dtype, part.shape) for part in tensor.parts}
    if parts == {"value": WeightsPart("bf16", tensor.shape)}:
        encoding = encodings["plain"]
    else:
        scale = (
            WeightsPart("f32", (tensor.shape[0],))
            if variant == "fp8"
            else WeightsPart("u8", (tensor.shape[0], tensor.shape[1] // 32))
        )
        expected = {"data": WeightsPart("f8_e4m3fn", tensor.shape), "scale": scale}
        if variant == "plain" or parts != expected or (variant == "mxfp8" and tensor.shape[1] % 32):
            raise ValueError(f"unexpected H3 FC1 roles: {tensor.component}/{tensor.key}")
        encoding = encodings[variant]
    return WeightsTensor(tensor.logical_dtype, tensor.shape, encoding, parts)


def _read_swapped(
    transaction: WeightsTransaction, component: str, key: str, role: str, length: int
) -> bytearray:
    """One bounded role buffer; no source callback re-enters a locked native writer."""
    if length <= 0 or length > MAX_ROLE_BYTES or length % 2:
        raise ValueError("H3 row repair exceeds its bounded whole-role buffer")
    raw = bytearray(length)
    view = memoryview(raw)
    half = length // 2
    for destination, source in ((0, half), (half, 0)):
        for offset in range(0, half, READ_BYTES):
            size = min(READ_BYTES, half - offset)
            transaction.source_read_into(
                "source",
                component,
                key,
                role,
                source + offset,
                view[destination + offset : destination + offset + size],
            )
    return raw


def repair(
    source: Model[object], artifacts: WeightsSink, tel: Telemetry, encodings: Mapping[str, str]
) -> RepairResult:
    variant = SOURCES.get(source.checkpoint_ref)
    if variant is None:
        raise ValueError(
            "repair accepts only the four recorded H3 checkpoints with unswapped FC1 rows"
        )
    structure = artifacts.structure(source)
    targets: dict[str, WeightsTarget] = {}
    changed: dict[tuple[str, str], WeightsTensor] = {}
    for tensor in structure.tensors:
        targets.setdefault(
            tensor.component, WeightsTarget(source="source", source_component=tensor.component)
        )
        if tensor.component in COMPONENTS and tensor.key in FC1_KEYS:
            changed[tensor.component, tensor.key] = _declaration(tensor, variant, encodings)
    wanted = {(component, key) for component in COMPONENTS for key in FC1_KEYS}
    if set(changed) != wanted:
        raise ValueError("repair source does not contain all 104 declared H3 FC1 matrices")
    for component in COMPONENTS:
        additions = {key: row for (owner, key), row in changed.items() if owner == component}
        targets[component] = WeightsTarget(
            source="source",
            source_component=component,
            drop=tuple(additions),
            add=additions,
        )
    configs = {
        name: WeightsConfig(source="source", source_config=name) for name in structure.configs
    }
    order = tuple((tensor.component, tensor.key) for tensor in structure.tensors)
    source_bytes, replayed_parts = 0, 0
    with artifacts.open(
        "checkpoint",
        sources={"source": source},
        targets=targets,
        configs=configs,
        order=order,
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
        else:
            completed = transaction.completed_parts
            for index, ((component, key), declaration) in enumerate(changed.items()):
                for role, part in declaration.parts.items():
                    if (component, key, role) in completed:
                        replayed_parts += 1
                        continue
                    length = math.prod(part.shape) * DTYPE_BYTES[part.dtype]
                    raw = _read_swapped(transaction, component, key, role, length)
                    transaction.add_part(component, key, role, raw)
                    source_bytes += length
                    del raw
                transaction.checkpoint()
                tel.progress((index + 1) / len(changed), stage="repair-fc1")
            receipt = transaction.commit()
    return RepairResult(
        source.checkpoint_ref,
        receipt.tensorfs_receipt_digest,
        receipt.weights_transaction_id,
        len(changed),
        source_bytes,
        replayed_parts,
        transaction.replayed,
    )
