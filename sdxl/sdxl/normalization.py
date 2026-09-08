"""Reviewed SDXL single-file names and layouts become the family's constructor input."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import msgspec
import numpy as np
from cozy_runtime import canonical_json
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    WeightsConfig,
    WeightsPart,
    WeightsPartSource,
    WeightsSink,
    WeightsSource,
    WeightsTarget,
    WeightsTensor,
    WeightsTransaction,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource

MAX_NEW_BYTES = 1 << 30
MAX_PART_BYTES = 16 << 20


class SourceTensor(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    dtype: str
    shape: tuple[int, ...]


class TensorRoute(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    component: str
    key: str
    source_key: str
    shape: tuple[int, ...]
    kind: Literal["graft", "read", "transpose"]
    offset: int


class NormalizationPlan(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    plain: str
    source: dict[str, dict[str, SourceTensor]]
    targets: tuple[TensorRoute, ...]
    configs: dict[str, dict[str, Any]]


PLAN = msgspec.json.decode(
    Path(__file__).with_name("normalization.json").read_bytes(), type=NormalizationPlan
)


def _validate(source: WeightsSource, plan: NormalizationPlan) -> None:
    observed = {(tensor.component, tensor.key): tensor for tensor in source.tensors}
    expected = {
        (component, key): spec
        for component, rows in plan.source.items()
        for key, spec in rows.items()
    }
    if source.configs or set(observed) != set(expected):
        raise UnsupportedInput("normalization requires the reviewed raw SDXL component key sets")
    for key, spec in expected.items():
        tensor = observed[key]
        if (
            tensor.logical_dtype != spec.dtype
            or tensor.shape != spec.shape
            or tensor.encoding != plan.plain
            or len(tensor.parts) != 1
            or tensor.parts[0].name != "value"
            or tensor.parts[0].dtype != spec.dtype
            or tensor.parts[0].shape != spec.shape
        ):
            raise UnsupportedInput(f"normalization source geometry or encoding differs at {key}")


def _bytes(transaction: WeightsTransaction, route: TensorRoute, spec: SourceTensor) -> bytes:
    size = math.prod(route.shape) * 2
    if spec.dtype != "f16" or not 0 < size <= MAX_PART_BYTES:
        raise UnsupportedInput("SDXL normalization role exceeds the reviewed fp16 byte bound")
    if route.kind == "read":
        raw = bytearray(size)
        transaction.source_read_into(
            "source", route.component, route.source_key, "value", route.offset * 2, memoryview(raw)
        )
        return bytes(raw)
    source_size = math.prod(spec.shape) * 2
    if (
        route.kind != "transpose"
        or len(spec.shape) != 2
        or source_size > MAX_PART_BYTES
        or route.shape != spec.shape[::-1]
        or route.offset
    ):
        raise UnsupportedInput("SDXL normalization has an unsupported transpose")
    raw = bytearray(source_size)
    transaction.source_read_into(
        "source", route.component, route.source_key, "value", 0, memoryview(raw)
    )
    # uint16 preserves every bit, including signed zero and NaN payloads: this is a
    # permutation, never float arithmetic or an independently implemented quantizer.
    return np.frombuffer(raw, dtype="<u2").reshape(spec.shape).T.copy().tobytes()


def _normalize(
    source: QuantizationSource,
    weights: WeightsSink,
    ctx: Context,
    tel: Telemetry,
    plan: NormalizationPlan,
) -> ModelArtifact:
    _validate(weights.structure(source), plan)
    targets: dict[str, dict[str, WeightsTensor]] = {}
    for route in plan.targets:
        graft = None
        if route.kind == "graft":
            graft = WeightsPartSource("source", route.component, route.source_key, "value")
        targets.setdefault(route.component, {})[route.key] = WeightsTensor(
            "f16",
            route.shape,
            plan.plain,
            {"value": WeightsPart("f16", route.shape, source=graft)},
        )
    configs = {name: canonical_json.encode(value) for name, value in plan.configs.items()}
    with weights.open(
        "model",
        sources={"source": source},
        targets={
            name: WeightsTarget(
                source="source", source_component=name, drop=tuple(plan.source[name]), add=rows
            )
            for name, rows in targets.items()
        },
        configs={name: WeightsConfig(data=data) for name, data in configs.items()},
        order=tuple((row.component, row.key) for row in plan.targets),
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return receipt.artifact
        # The stock CLIP position IDs are reconstructed by the constructor. A checkpoint
        # that changed their values is outside this mapping even if its shapes still match.
        position = plan.source.get("text_encoder", {}).get("text_model.embeddings.position_ids")
        if position is not None:
            count = math.prod(position.shape)
            if position.dtype != "i64" or not 0 < count <= 77:
                raise UnsupportedInput("unexpected SDXL position-ID geometry")
            raw = bytearray(count * 8)
            transaction.source_read_into(
                "source",
                "text_encoder",
                "text_model.embeddings.position_ids",
                "value",
                0,
                memoryview(raw),
            )
            if not np.array_equal(np.frombuffer(raw, dtype="<i8"), np.arange(count, dtype="<i8")):
                raise UnsupportedInput("SDXL source changed the constructor's position IDs")
        completed = transaction.completed_parts
        for index, route in enumerate(plan.targets):
            ctx.raise_if_cancelled()
            if route.kind == "graft" or (route.component, route.key, "value") in completed:
                continue
            value = _bytes(transaction, route, plan.source[route.component][route.source_key])
            transaction.add_part(route.component, route.key, "value", value)
            transaction.checkpoint()
            tel.progress((index + 1) / len(plan.targets), stage="normalize-sdxl")
        completed_configs = transaction.completed_configs
        for name, data in configs.items():
            if name not in completed_configs:
                transaction.add_config(name, data)
                transaction.checkpoint()
        return transaction.commit().artifact


@invocable(memoize=True)
async def normalize(
    ctx: Context, *, source: QuantizationSource, weights: WeightsSink, tel: Telemetry
) -> ModelArtifact:
    """Normalize one reviewed unquantized SDXL source using bounded native custody."""
    return _normalize(source, weights, ctx, tel, PLAN)
