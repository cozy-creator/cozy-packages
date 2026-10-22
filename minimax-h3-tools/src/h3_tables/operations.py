"""Managed H3 transformations for ordinary client scripts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib.resources import files
from typing import cast

from cozy_runtime.author import (
    ModelArtifact,
    UnsupportedInput,
    canonical_json,
)
from cozy_runtime.derive.operations import QuantizationPlan
from cozy_runtime.derive.operations import quantize as runtime_quantize

from .adaln_operations import (
    Selection,
    apply_adaln,
    compute_adaln_tables,
    retable_adaln,
    select_adaln_weights,
)
from .lanes import Encoding
from .model_config import DIT_COMPONENT, parse_full_config
from .order import current_order, full_order
from .quantization import dit_plan


def quantization_plan() -> QuantizationPlan:
    """Bind reviewed H3 geometry and canonical order without loading a source model."""
    plan = dit_plan()
    components = tuple(DIT_COMPONENT.values())
    selected = [
        [component, key, "bf16", list(shape)] for component in components for key, shape in plan
    ]
    assets = files(__package__).joinpath("assets")
    pruned = current_order(assets.joinpath("whole-order.json").read_bytes())
    sections = parse_full_config(assets.joinpath("model-config.json").read_bytes())
    full = full_order(sections, pruned.rows)
    return QuantizationPlan(
        components=components,
        keys=tuple(key for key, _ in plan),
        selected_schema_digest=canonical_json.digest(selected),
        source_order_digests=(
            pruned.digest,
            canonical_json.digest([[component, key] for component, key in full]),
        ),
        output_precision="preserve",
    )


async def quantize(
    *,
    source: ModelArtifact,
    encoding: Encoding,
    max_relative_frobenius: float | None = None,
) -> ModelArtifact:
    """Quantize both reviewed DiTs through the shared Runtime operation.

    The exact assembled full/pruned source order is required. Unselected tensors and
    configs retain their source precision; an encoded or malformed source refuses.
    """
    call = cast(Callable[..., Awaitable[ModelArtifact]], runtime_quantize)
    return await call(
        source=source,
        plan=quantization_plan(),
        encoding=encoding,
        max_relative_frobenius=max_relative_frobenius,
    )


async def precompute_adaln(
    *,
    model: ModelArtifact,
    timesteps: int = 50,
    generating_model: ModelArtifact | None = None,
) -> ModelArtifact:
    """Return the pruned H3 model using shared tables for approved 30/40/50-step schedules.

    This ordinary Python helper runs in the calling script; each leaf's committed derivation replays.
    Retabling an already pruned model requires its full generating model. The current
    approved union bank supports all three schedules and is shared across body encodings.
    """
    # The generated interfaces remove injected services and replace bound Model
    # parameters with ModelArtifact. These remain the same admitted call proxies.
    select = cast(Callable[..., Awaitable[Selection]], select_adaln_weights)
    compute = cast(Callable[..., Awaitable[ModelArtifact]], compute_adaln_tables)
    attach = cast(
        Callable[..., Awaitable[ModelArtifact]],
        retable_adaln if generating_model is not None else apply_adaln,
    )

    # Managed proxies carry exact ModelArtifact values; the owner injects the bound
    # model only into each child implementation. Generated caller interfaces express this.
    generator = generating_model or model
    selected_fl = await select(source=generator, task="fl2va")
    selected_ref = await select(source=generator, task="ref2va")
    if (
        type(timesteps) is not int
        or timesteps not in selected_fl.steps
        or timesteps not in selected_ref.steps
    ):
        raise UnsupportedInput(
            "H3 timestep count is not supported by both source model plans", code="adaln_plan"
        )
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
        plan_digest=selected_fl.plan_digest,
    )
    ref = await compute(
        source=selected_ref.projection,
        task="ref2va",
        plan_digest=selected_ref.plan_digest,
    )
    return await attach(
        source=model,
        fl2va=fl,
        ref2va=ref,
    )
