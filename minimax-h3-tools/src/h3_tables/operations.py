"""Managed H3 transformations for ordinary client scripts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib.resources import files
from typing import Literal, cast

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

from .model_config import dual_full_config, parse_production_config
from .order import current_order, full_order
from .source import TARGET_COMPONENT, H3FullTransformer, select_full_targets


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
                or actual.parts[0].name != "value"
                or actual.parts[0].dtype != expected.logical_dtype
                or actual.parts[0].shape != expected.shape
            ):
                raise UnsupportedInput(
                    f"{component}.{expected.key} must match the reviewed plain BF16 H3 source",
                    code="quantization_source",
                )
    return plan


def _quantization_order(structure: WeightsSource) -> tuple[tuple[str, str], ...]:
    """Accept only the existing complete H3 construction contracts, in their exact order."""
    assets = files(__package__).joinpath("assets")
    pruned = current_order(assets.joinpath("whole-order.json").read_bytes()).rows
    sections = parse_production_config(assets.joinpath("model-config.json").read_bytes())
    actual = {(tensor.component, tensor.key) for tensor in structure.tensors}
    for order in (pruned, full_order(sections, pruned)):
        if actual == set(order):
            return order
    raise UnsupportedInput(
        "H3 quantization requires one assembled full or AdaLN-pruned checkpoint",
        code="quantization_source",
    )


@invocable(memoize=True)
async def quantize(
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
    order = _quantization_order(structure)
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
        order=order,
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


@invocable(memoize=True)
async def precompute_adaln(
    ctx: Context,
    *,
    model: ModelArtifact,
    timesteps: int = 50,
    generating_model: ModelArtifact | None = None,
) -> ModelArtifact:
    """Return the pruned H3 model using shared tables for approved 30/40/50-step schedules.

    This orchestration job has inert artifact inputs and owns no GPU while children run.
    Retabling an already pruned model requires its full generating model. The current
    approved union bank supports all three schedules and is shared across body encodings.
    """
    from .adaln_operations import (
        Selection,
        _plan,
        apply_adaln,
        compute_adaln_tables,
        retable_adaln,
        select_adaln_weights,
    )

    # The generated interfaces remove injected services and replace bound Model
    # parameters with ModelArtifact. These remain the same admitted call proxies.
    select = cast(Callable[..., Awaitable[Selection]], select_adaln_weights)
    compute = cast(Callable[..., Awaitable[ModelArtifact]], compute_adaln_tables)
    attach = cast(
        Callable[..., Awaitable[ModelArtifact]],
        retable_adaln if generating_model is not None else apply_adaln,
    )

    ctx.raise_if_cancelled()
    if (
        type(timesteps) is not int
        or timesteps not in _plan("fl2va").steps
        or timesteps not in _plan("ref2va").steps
    ):
        raise UnsupportedInput(
            "H3 currently supports approved 30, 40 and 50 step schedules", code="adaln_plan"
        )
    # Managed proxies carry exact ModelArtifact values; the owner injects the bound
    # model only into each child implementation. Generated caller interfaces express this.
    generator = generating_model or model
    selected_fl = await select(source=generator, task="fl2va")
    selected_ref = await select(source=generator, task="ref2va")
    if selected_fl.ready and selected_ref.ready:
        if generating_model is not None:
            raise UnsupportedInput(
                "generating_model must contain the original dynamic AdaLN weights",
                code="adaln_generating_weights",
            )
        return model
    if selected_fl.projection is None or selected_ref.projection is None:
        raise UnsupportedInput(
            "H3 source does not contain both tasks' generating weights",
            code="adaln_generating_weights",
        )
    fl = await compute(
        source=selected_fl.projection,
        task="fl2va",
        plan_digest=_plan("fl2va").digest,
    )
    ref = await compute(
        source=selected_ref.projection,
        task="ref2va",
        plan_digest=_plan("ref2va").digest,
    )
    return await attach(
        source=model,
        fl2va=fl,
        ref2va=ref,
    )


@invocable(memoize=True)
async def assemble_full(
    ctx: Context,
    *,
    dits: H3FullTransformer,
    shared: H3FullTransformer,
    weights: WeightsSink,
) -> ModelArtifact:
    """Assemble converted DiTs and shared weights using the existing native H3 order."""
    ctx.raise_if_cancelled()
    assets = files(__package__).joinpath("assets")
    sections = parse_production_config(assets.joinpath("model-config.json").read_bytes())
    current = current_order(assets.joinpath("whole-order.json").read_bytes()).rows
    sources = {"dits": dits, "shared": shared}
    return weights.derive(
        "model",
        sources=sources,
        targets=select_full_targets(weights, sources),
        configs={"model": WeightsConfig(data=dual_full_config(sections))},
        order=full_order(sections, current),
    ).artifact
