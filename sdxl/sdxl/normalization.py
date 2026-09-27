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
    MemoDistribution,
    MemoResource,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    canonical_json,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource
from tensorfs.derived import (
    Config,
    Derivation,
    DerivedTransaction,
    Part,
    PartSource,
    SourceInspection,
    Target,
    Tensor,
    derive,
)

MAX_NEW_BYTES = 1 << 30

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


#: Constructor-derived CLIP position IDs. Civitai checkpoints carry either, both or neither;
#: the constructor rebuilds them, so they are validated when present and never emitted.
OPTIONAL_BUFFERS = frozenset(
    {
        ("text_encoder", "text_model.embeddings.position_ids"),
        ("text_encoder_2", "transformer.text_model.embeddings.position_ids"),
    }
)
#: Source float widths a checkpoint may store; the normalized lane is always f16.
FLOATS = frozenset({"f16", "bf16", "f32"})
_NUMPY = {"f16": "<f2", "f32": "<f4", "i64": "<i8"}
_WIDTH = {"f16": 2, "bf16": 2, "f32": 4, "i64": 8}
MAX_CONVERTED_BYTES = 512 << 20
READ_CHUNK = 32 << 20
POSITIONS = 77


def _validate(source: SourceInspection, plan: NormalizationPlan) -> dict[tuple[str, str], Tensor]:
    """Check the reviewed SDXL key set and geometry; return each source tensor's dtype.

    A tensor may be stored at f16, bf16 or f32 (f16 grafts unchanged; wider or bf16 rows are
    rounded to f16). Only the optional position-ID buffers may be absent or extra.
    """
    observed = {
        (component, key): tensor
        for component, rows in source.components.items()
        for key, tensor in rows.items()
    }
    expected = {
        (component, key): spec
        for component, rows in plan.source.items()
        for key, spec in rows.items()
    }
    missing = sorted(set(expected) - set(observed) - OPTIONAL_BUFFERS)
    extra = sorted(set(observed) - set(expected) - OPTIONAL_BUFFERS)
    if source.configs or missing or extra or set(source.components) != set(plan.source):
        raise UnsupportedInput(
            "not the reviewed SDXL single-file layout: "
            f"missing={[f'{c}.{k}' for c, k in missing[:4]]} "
            f"unexpected={[f'{c}.{k}' for c, k in extra[:4]]} "
            f"components={sorted(source.components)} configs={sorted(source.configs)}",
            code="sdxl_source_layout",
        )
    for name, tensor in observed.items():
        value = tensor.parts.get("value")
        spec = expected.get(name)
        buffer = name in OPTIONAL_BUFFERS
        admitted = FLOATS | ({"i64"} if buffer else set())
        if (
            tensor.logical_dtype not in admitted
            or (spec is not None and tensor.shape != spec.shape)
            or tensor.encoding != plan.plain
            or len(tensor.parts) != 1
            or value is None
            or value.dtype != tensor.logical_dtype
            or tuple(value.shape) != tuple(tensor.shape)
        ):
            raise UnsupportedInput(
                f"SDXL source tensor {name[0]}.{name[1]} is {tensor.logical_dtype} "
                f"{tuple(tensor.shape)}; expected a plain f16/bf16/f32 value"
                + ("" if spec is None else f" of shape {spec.shape}"),
                code="sdxl_source_layout",
            )
        if buffer and (not 0 < math.prod(tensor.shape) <= POSITIONS or len(tensor.shape) > 2):
            raise UnsupportedInput(f"unexpected SDXL position-ID geometry at {name[0]}.{name[1]}")
    return observed


def _read(
    transaction: DerivedTransaction, component: str, key: str, offset: int, raw: bytearray
) -> None:
    view = memoryview(raw)
    for start in range(0, len(raw), READ_CHUNK):
        transaction.source_read_into(
            "source", component, key, "value", offset + start, view[start : start + READ_CHUNK]
        )


def _to_f16(raw: bytearray, dtype: str, where: str) -> np.ndarray[Any, np.dtype[np.float16]]:
    """Round a bf16/f32 row to f16 (round to nearest even); refuse any overflow."""
    if dtype == "bf16":
        wide = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        wide = np.frombuffer(raw, dtype="<f4")
    with np.errstate(over="ignore"):
        narrow = wide.astype("<f2")
    if not np.array_equal(np.isfinite(wide), np.isfinite(narrow)):
        raise UnsupportedInput(
            f"SDXL tensor {where} exceeds the f16 range", code="sdxl_f16_overflow"
        )
    return narrow


def _bytes(
    transaction: DerivedTransaction, route: TensorRoute, spec: SourceTensor, dtype: str
) -> bytes:
    """One normalized f16 value: an f16 source is permuted bit-exactly, others are rounded."""
    where = f"{route.component}.{route.source_key}"
    if route.kind == "transpose":
        if len(spec.shape) != 2 or route.shape != spec.shape[::-1] or route.offset:
            raise UnsupportedInput("SDXL normalization has an unsupported transpose")
        count = math.prod(spec.shape)
    elif route.kind in {"read", "graft"}:
        count = math.prod(route.shape)
    else:
        raise UnsupportedInput(f"unknown SDXL normalization route {route.kind}")
    size = count * _WIDTH[dtype]
    if dtype not in FLOATS or not 0 < size <= MAX_CONVERTED_BYTES:
        raise UnsupportedInput(f"SDXL tensor {where} exceeds the reviewed byte bound")
    raw = bytearray(size)
    offset = 0 if route.kind == "transpose" else route.offset * _WIDTH[dtype]
    _read(transaction, route.component, route.source_key, offset, raw)
    values = np.frombuffer(raw, dtype="<u2") if dtype == "f16" else _to_f16(raw, dtype, where)
    if route.kind == "transpose":
        # uint16 preserves every f16 bit, including signed zero and NaN payloads: this is
        # a permutation, never float arithmetic.
        return values.view("<u2").reshape(spec.shape).T.copy().tobytes()
    return values.tobytes()


def _validate_position_ids(raw: bytes, dtype: str, count: int) -> None:
    if dtype == "bf16":
        values = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        values = np.frombuffer(raw, dtype=_NUMPY[dtype])
    # Exact comparison also rejects fractions, infinities and NaNs: the stock sequence is
    # stored losslessly at every admitted width.
    if not np.array_equal(values, np.arange(count, dtype="<i8")):
        raise UnsupportedInput("SDXL source changed the constructor's position IDs")


def _targets(plan: NormalizationPlan, dtypes: Mapping[tuple[str, str], str]) -> dict[str, Target]:
    """f16 grafts inherit source objects; every other route is written as f16 bytes."""
    targets: dict[str, dict[str, Tensor]] = {}
    for route in plan.targets:
        graft = None
        if route.kind == "graft" and dtypes[route.component, route.source_key] == "f16":
            graft = PartSource("source", route.component, route.source_key, "value")
        targets.setdefault(route.component, {})[route.key] = Tensor(
            "f16",
            route.shape,
            plan.plain,
            {"value": Part("f16", route.shape, source=graft)},
        )
    # Grafts already authorize their source component. Keep a base only when a
    # component has no graft, so its read/transpose routes retain that authority.
    return {
        name: Target(add=rows)
        if any(
            part.source is not None for tensor in rows.values() for part in tensor.parts.values()
        )
        else Target(
            source="source",
            source_component=name,
            drop=tuple(sorted(key for component, key in dtypes if component == name)),
            add=rows,
        )
        for name, rows in targets.items()
    }


def _normalize(
    source: QuantizationSource,
    ctx: Context,
    tel: Telemetry,
    plan: NormalizationPlan,
) -> ModelArtifact:
    source_capability = ctx.tensorfs_source(source)
    tensors = _validate(source_capability.inspect(), plan)
    dtypes = {name: tensor.logical_dtype for name, tensor in tensors.items()}
    configs = {name: canonical_json.encode(value) for name, value in plan.configs.items()}
    definition = Derivation(
        sources={"source": source_capability},
        targets=_targets(plan, dtypes),
        configs={name: Config("add") for name in configs},
        order=tuple((row.component, row.key) for row in plan.targets),
    )
    components = {route.component for route in plan.targets}
    with derive(ctx.output("model"), definition) as transaction:
        if transaction.receipt is not None:
            receipt = transaction.receipt
            assert receipt is not None
            return ctx.adopt_model(receipt)
        # The stock CLIP position IDs are reconstructed by the constructor. A checkpoint
        # that changed their values is outside this mapping even if its shapes still match.
        for component, key in sorted(OPTIONAL_BUFFERS):
            buffer = tensors.get((component, key))
            if component not in components or buffer is None:
                continue
            count = math.prod(buffer.shape)
            raw = bytearray(count * _WIDTH[buffer.logical_dtype])
            transaction.source_read_into("source", component, key, "value", 0, memoryview(raw))
            _validate_position_ids(bytes(raw), buffer.logical_dtype, count)
        completed = set(transaction.completed_parts())
        reused = written = 0
        for index, route in enumerate(plan.targets):
            ctx.raise_if_cancelled()
            dtype = dtypes[route.component, route.source_key]
            if route.kind == "graft" and dtype == "f16":
                continue
            if (route.component, route.key, "value") in completed:
                reused += 1
                continue
            value = _bytes(transaction, route, plan.source[route.component][route.source_key], dtype)
            transaction.add_part(route.component, route.key, "value", value)
            transaction.checkpoint()
            written += 1
            tel.progress((index + 1) / len(plan.targets), stage="normalize-sdxl")
        tel.metric("sdxl.reused_tensors", float(reused))
        tel.metric("sdxl.computed_tensors", float(written))
        completed_configs = set(transaction.completed_configs())
        for name, data in configs.items():
            if name not in completed_configs:
                transaction.add_config(name, data)
                transaction.checkpoint()
        return ctx.adopt_model(transaction.receipt or transaction.commit())


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
    ctx: Context,
    plan: NormalizationPlan,
) -> ModelArtifact:
    if set(sources) != set(_COMPONENTS):
        raise UnsupportedInput("normalized SDXL assembly requires all four components")
    configs: dict[str, Config] = {}
    capabilities = {name: ctx.tensorfs_source(source) for name, source in sources.items()}
    for component in _COMPONENTS:
        ctx.raise_if_cancelled()
        selected = _component_plan(plan, component)
        observed = capabilities[component].inspect()
        observed_tensors = [
            (name, key, tensor)
            for name, rows in observed.components.items()
            for key, tensor in rows.items()
        ]
        expected_order = tuple((row.component, row.key) for row in selected.targets)
        if tuple((name, key) for name, key, _ in observed_tensors) != expected_order or set(
            observed.configs
        ) != set(selected.configs):
            raise UnsupportedInput(f"normalized {component} key, order or config set differs")
        for (_, _, actual), expected in zip(observed_tensors, selected.targets, strict=True):
            value = actual.parts.get("value")
            if (
                actual.logical_dtype != "f16"
                or actual.shape != expected.shape
                or actual.encoding != plan.plain
                or len(actual.parts) != 1
                or value is None
                or value.dtype != "f16"
                or tuple(value.shape) != expected.shape
            ):
                raise UnsupportedInput(f"normalized {component}.{expected.key} geometry differs")
        for name, config_value in selected.configs.items():
            if observed.configs[name] != canonical_json.encode(config_value):
                raise UnsupportedInput(f"normalized {component} config {name} differs")
            configs[name] = Config("copy", source=component, source_config=name)
    definition = Derivation(
        sources=capabilities,
        targets={name: Target(source=name, source_component=name) for name in _COMPONENTS},
        configs=configs,
        order=tuple((row.component, row.key) for row in plan.targets),
    )
    with derive(ctx.output("model"), definition) as transaction:
        return ctx.adopt_model(transaction.receipt or transaction.commit())


_NORMALIZATION_MEMO = (
    SourceTensor, TensorRoute, NormalizationPlan,
    "tensorfs.derived", MemoResource("sdxl", "normalization.json"),
    MemoDistribution("tensorfs"), MemoDistribution("numpy"), MemoDistribution("msgspec"),
)


_NORMALIZE_HELPERS = (
    _normalize, _validate, _read, _to_f16, _bytes, _validate_position_ids, _targets,
)


@invocable(memoize=True, memo_dependencies=(
    *_NORMALIZATION_MEMO, *_NORMALIZE_HELPERS, _component_plan,
))
async def normalize_component(
    ctx: Context,
    *,
    source: QuantizationSource,
    component: Component,
    tel: Telemetry,
) -> ModelArtifact:
    """Normalize one SDXL component with the original bounded byte transformations."""
    return _normalize(source, ctx, tel, _component_plan(PLAN, component))


@invocable(memoize=True, memo_dependencies=(
    *_NORMALIZATION_MEMO, _assemble_normalized, _component_plan,
))
async def assemble_normalized(
    ctx: Context,
    *,
    text_encoder: QuantizationSource,
    text_encoder_2: QuantizationSource,
    unet: QuantizationSource,
    vae: QuantizationSource,
) -> ModelArtifact:
    """Validate and graft normalized components into the exact SDXL construction order."""
    return _assemble_normalized(
        {"text_encoder": text_encoder, "text_encoder_2": text_encoder_2, "unet": unet, "vae": vae},
        ctx,
        PLAN,
    )


async def normalize(*, source: ModelArtifact) -> ModelArtifact:
    """Convert one Civitai SDXL single-file checkpoint into the served f16 Diffusers model.

    ``source`` is converted with ``civitai/sdxl/single-file/1`` at f16, bf16 or f32. Four
    component normalizations and one assembly keep every native declaration under the
    1 MiB protocol bound; the whole-model declaration alone exceeds it.
    """
    component_call = cast(Callable[..., Awaitable[ModelArtifact]], normalize_component)
    assembly_call = cast(Callable[..., Awaitable[ModelArtifact]], assemble_normalized)
    components = {
        component: await component_call(source=source, component=component)
        for component in _COMPONENTS
    }
    return await assembly_call(**components)
