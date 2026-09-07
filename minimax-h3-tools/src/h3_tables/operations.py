"""Managed H3 transformations for ordinary client scripts."""

from __future__ import annotations

from typing import Literal

import msgspec
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    WeightsConfig,
    WeightsSink,
    WeightsSource,
    WeightsTarget,
    invocable,
)
from cozy_runtime.derive.quantization import (
    ArtifactQuantizationPlan,
    ArtifactQuantizationRequest,
    h3_quantization_plan,
    prepare_quantization,
    quantization_additions,
    quantize_component_into,
)

from .job import TARGET_COMPONENT, H3FullTransformer


def _quantization_plan(structure: WeightsSource) -> ArtifactQuantizationPlan:
    """Use the reviewed H3 keys, preserving every unselected tensor at source precision."""
    plan = prepare_quantization(h3_quantization_plan())
    tensors = {(tensor.component, tensor.key): tensor for tensor in structure.tensors}
    for component in TARGET_COMPONENT.values():
        for expected in plan.tensors:
            actual = tensors.get((component, expected.key))
            if actual is None or (
                actual.logical_dtype != expected.logical_dtype
                or actual.shape != expected.shape
                or len(actual.parts) != 1
                or actual.parts[0].role != "value"
                or actual.parts[0].dtype != expected.logical_dtype
                or actual.parts[0].shape != expected.shape
            ):
                raise UnsupportedInput(
                    f"{component}.{expected.key} must match the reviewed plain BF16 H3 source",
                    code="quantization_source",
                )
    return plan


@invocable(memoize=True)
async def quantize_lane(
    ctx: Context,
    *,
    source: H3FullTransformer,
    encoding: Literal["fp8-rowwise/1", "mxfp8/1"],
    max_relative_frobenius: float | None = None,
    weights: WeightsSink,
    tel: Telemetry,
) -> ModelArtifact:
    """Encode the two BF16 DiT bodies independently of shared weights and AdaLN tables.

    Both encodings take the same BF16 full or pruned checkpoint. An already encoded
    source refuses; all unselected parts, configs and construction order are inherited.
    """
    structure = weights.structure(source)
    plan = _quantization_plan(structure)
    additions = quantization_additions(encoding, plan, "dit")
    targets = {
        component: WeightsTarget(source="source", source_component=component)
        for component in dict.fromkeys(tensor.component for tensor in structure.tensors)
    }
    for component in TARGET_COMPONENT.values():
        targets[component] = WeightsTarget(
            source="source",
            source_component=component,
            drop=tuple(sorted(additions)),
            add=additions,
        )
    payload = ArtifactQuantizationRequest(max_relative_frobenius=max_relative_frobenius)
    with weights.open(
        "model",
        sources={"source": source},
        targets=targets,
        configs={
            name: WeightsConfig(source="source", source_config=name) for name in structure.configs
        },
        order=tuple((tensor.component, tensor.key) for tensor in structure.tensors),
    ) as transaction:
        if transaction.replayed:
            assert transaction.receipt is not None
            return transaction.receipt.artifact
        for component in TARGET_COMPONENT.values():
            stats = quantize_component_into(
                transaction,
                ctx,
                payload,
                tel,
                encoding=encoding,
                plan=plan,
                component="dit",
                source="source",
                source_component=component,
                target_component=component,
            )
            tel.log("quantization measurement", component=component, **msgspec.to_builtins(stats))
        return transaction.commit().artifact
