# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.7.0,<1"]
# [tool.cozy.models]
# source = "paul/minimax-h3@sha256:3f6c224a010fbff36adce0f19a8b866f4f1739a1d30e2202c140ba994dc0b4df"
# [tool.cozy.weights]
# checkpoint = 34359738368
# ///
"""One-time FC1 row repair for the four original H3 checkpoints.

Run: cozy run ./h3_checkpoint_repair.py --rental-only
Select another recorded bad root with model.source=paul/minimax-h3@sha256:...
All four production repairs are already complete (se-031). This example preserves
that operation; corrected and unknown inputs refuse before any writes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from cozy_runtime.author import (
    Context,
    Model,
    ModelArtifact,
    Telemetry,
)
from tensorfs.derived import Config, Derivation, DerivedTransaction, Part, Target, Tensor

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


def _declaration(
    tensor: Tensor, variant: str, encodings: Mapping[str, str], *, component: str, key: str
) -> Tensor:
    if (
        tensor.logical_dtype != "bf16"
        or len(tensor.shape) != 2
        or min(tensor.shape) <= 0
        or tensor.shape[0] % 2
    ):
        raise ValueError(f"unexpected H3 FC1 geometry: {component}/{key}")
    parts = dict(tensor.parts)
    if parts == {"value": Part("bf16", tensor.shape)}:
        encoding = encodings["plain"]
    else:
        scale = (
            Part("f32", (tensor.shape[0],))
            if variant == "fp8"
            else Part("u8", (tensor.shape[0], tensor.shape[1] // 32))
        )
        expected = {"data": Part("f8_e4m3fn", tensor.shape), "scale": scale}
        if variant == "plain" or parts != expected or (variant == "mxfp8" and tensor.shape[1] % 32):
            raise ValueError(f"unexpected H3 FC1 roles: {component}/{key}")
        encoding = encodings[variant]
    return Tensor(tensor.logical_dtype, tensor.shape, encoding, parts)


def _read_swapped(
    transaction: DerivedTransaction, component: str, key: str, role: str, length: int
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
    source: Model[object], ctx: Context, tel: Telemetry, encodings: Mapping[str, str]
) -> ModelArtifact:
    variant = SOURCES.get(source.checkpoint_ref)
    if variant is None:
        raise ValueError(
            "repair accepts only the four recorded H3 checkpoints with unswapped FC1 rows"
        )
    with ctx.tensorfs_source(source) as capability:
        structure = capability.inspect()
    targets: dict[str, Target] = {}
    changed: dict[tuple[str, str], Tensor] = {}
    for component, tensors in structure.components.items():
        targets[component] = Target(source="source", source_component=component)
        for key, tensor in tensors.items():
            if component in COMPONENTS and key in FC1_KEYS:
                changed[component, key] = _declaration(
                    tensor, variant, encodings, component=component, key=key
                )
    wanted = {(component, key) for component in COMPONENTS for key in FC1_KEYS}
    if set(changed) != wanted:
        raise ValueError("repair source does not contain all 104 declared H3 FC1 matrices")
    for component in COMPONENTS:
        additions = {key: row for (owner, key), row in changed.items() if owner == component}
        targets[component] = Target(
            source="source",
            source_component=component,
            drop=tuple(additions),
            add=additions,
        )
    configs = {
        name: Config("copy", source="source", source_config=name) for name in structure.configs
    }
    order = tuple(
        (component, key) for component, rows in structure.components.items() for key in rows
    )
    source_bytes, replayed_parts = 0, 0
    with ctx.output("checkpoint").open(
        Derivation(
            sources={"source": structure.source},
            targets=targets,
            configs=configs,
            order=order,
        )
    ) as transaction:
        replayed = transaction.receipt is not None
        if transaction.receipt is not None:
            receipt = transaction.receipt
            assert receipt is not None
        else:
            completed = transaction.completed_parts()
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
    artifact = ctx.adopt_model(receipt)
    tel.log(
        "H3 checkpoint repaired",
        source_checkpoint=source.checkpoint_ref,
        checkpoint=artifact.manifest.digest,
        repaired_tensors=len(changed),
        source_bytes_read_this_run=source_bytes,
        replayed_parts=replayed_parts,
        replayed=replayed,
    )
    return artifact


def main(*, source: Model[object], ctx: Context, tel: Telemetry) -> ModelArtifact:
    return repair(source, ctx, tel, ENCODINGS)
