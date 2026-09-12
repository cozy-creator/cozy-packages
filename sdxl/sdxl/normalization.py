"""Reviewed SDXL single-file names and layouts become the family's constructor input."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import msgspec
import numpy as np
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
    canonical_json,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource

MAX_NEW_BYTES = 1 << 30
MAX_PART_BYTES = 16 << 20

Component = Literal["text_encoder", "text_encoder_2", "unet", "vae"]
_COMPONENTS: tuple[Component, ...] = ("text_encoder", "text_encoder_2", "unet", "vae")


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


def _validate_position_ids(raw: bytes, dtype: np.dtype[Any], count: int) -> None:
    values = np.frombuffer(raw, dtype=dtype)
    # Exact comparison also rejects fractions, infinities and NaNs. The reviewed
    # single-file checkpoint stores this integer sequence losslessly as F16.
    if not np.array_equal(values, np.arange(count, dtype="<i8")):
        raise UnsupportedInput("SDXL source changed the constructor's position IDs")


def _targets(plan: NormalizationPlan) -> dict[str, WeightsTarget]:
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
    # Explicit destination rows include every graft source; no inherited base or
    # drop-all roster is needed. Keep receipts below their base64-wrapped limit.
    return {name: WeightsTarget(add=rows) for name, rows in targets.items()}


def _normalize(
    source: QuantizationSource,
    weights: WeightsSink,
    ctx: Context,
    tel: Telemetry,
    plan: NormalizationPlan,
) -> ModelArtifact:
    _validate(weights.structure(source), plan)
    configs = {name: canonical_json.encode(value) for name, value in plan.configs.items()}
    with weights.open(
        "model",
        sources={"source": source},
        targets=_targets(plan),
        configs={name: WeightsConfig(data=data) for name, data in configs.items()},
        order=tuple((row.component, row.key) for row in plan.targets),
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return receipt.artifact
        # The stock CLIP position IDs are reconstructed by the constructor. A checkpoint
        # that changed their values is outside this mapping even if its shapes still match.
        position = (
            plan.source.get("text_encoder", {}).get("text_model.embeddings.position_ids")
            if any(route.component == "text_encoder" for route in plan.targets)
            else None
        )
        if position is not None:
            count = math.prod(position.shape)
            if position.dtype not in {"f16", "i64"} or not 0 < count <= 77:
                raise UnsupportedInput("unexpected SDXL position-ID geometry")
            dtype = np.dtype("<f2" if position.dtype == "f16" else "<i8")
            raw = bytearray(count * dtype.itemsize)
            transaction.source_read_into(
                "source",
                "text_encoder",
                "text_model.embeddings.position_ids",
                "value",
                0,
                memoryview(raw),
            )
            _validate_position_ids(bytes(raw), dtype, count)
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


def _component_plan(plan: NormalizationPlan, component: Component) -> NormalizationPlan:
    if component not in _COMPONENTS or component not in plan.source:
        raise UnsupportedInput("unknown SDXL normalization component")
    return NormalizationPlan(
        plain=plan.plain,
        source=plan.source,
        targets=tuple(route for route in plan.targets if route.component == component),
        configs={
            name: value
            for name, value in plan.configs.items()
            if name == component or (component == "unet" and name not in _COMPONENTS)
        },
    )


def _assemble_normalized(
    sources: Mapping[str, QuantizationSource],
    weights: WeightsSink,
    ctx: Context,
    plan: NormalizationPlan,
) -> ModelArtifact:
    if set(sources) != set(_COMPONENTS):
        raise UnsupportedInput("normalized SDXL assembly requires all four components")
    configs: dict[str, WeightsConfig] = {}
    for component in _COMPONENTS:
        ctx.raise_if_cancelled()
        source = sources[component]
        selected = _component_plan(plan, component)
        observed = weights.structure(source)
        expected_order = tuple((row.component, row.key) for row in selected.targets)
        if (
            tuple((row.component, row.key) for row in observed.tensors) != expected_order
            or set(observed.configs) != set(selected.configs)
        ):
            raise UnsupportedInput(f"normalized {component} key, order or config set differs")
        for actual, expected in zip(observed.tensors, selected.targets, strict=True):
            if (
                actual.logical_dtype != "f16"
                or actual.shape != expected.shape
                or actual.encoding != plan.plain
                or len(actual.parts) != 1
                or actual.parts[0].name != "value"
                or actual.parts[0].dtype != "f16"
                or actual.parts[0].shape != expected.shape
            ):
                raise UnsupportedInput(f"normalized {component}.{expected.key} geometry differs")
        for name, value in selected.configs.items():
            if weights.config(source, name) != canonical_json.encode(value):
                raise UnsupportedInput(f"normalized {component} config {name} differs")
            configs[name] = WeightsConfig(source=component, source_config=name)
    with weights.open(
        "model",
        sources=sources,
        targets={
            name: WeightsTarget(source=name, source_component=name) for name in _COMPONENTS
        },
        configs=configs,
        order=tuple((row.component, row.key) for row in plan.targets),
    ) as transaction:
        return transaction.commit().artifact


@invocable(memoize=True)
async def normalize_component(
    ctx: Context,
    *,
    source: QuantizationSource,
    component: Component,
    weights: WeightsSink,
    tel: Telemetry,
) -> ModelArtifact:
    """Normalize one SDXL component with the original bounded byte transformations."""
    return _normalize(source, weights, ctx, tel, _component_plan(PLAN, component))


@invocable(memoize=True)
async def assemble_normalized(
    ctx: Context,
    *,
    text_encoder: QuantizationSource,
    text_encoder_2: QuantizationSource,
    unet: QuantizationSource,
    vae: QuantizationSource,
    weights: WeightsSink,
) -> ModelArtifact:
    """Validate and graft normalized components into the exact SDXL construction order."""
    return _assemble_normalized(
        {"text_encoder": text_encoder, "text_encoder_2": text_encoder_2, "unet": unet, "vae": vae},
        weights,
        ctx,
        PLAN,
    )


async def normalize(*, source: ModelArtifact) -> ModelArtifact:
    """Compose reusable normalized components in the caller without nested jobs."""
    component_call = cast(Callable[..., Awaitable[ModelArtifact]], normalize_component)
    assembly_call = cast(Callable[..., Awaitable[ModelArtifact]], assemble_normalized)
    components = {
        component: await component_call(source=source, component=component)
        for component in _COMPONENTS
    }
    return await assembly_call(**components)
