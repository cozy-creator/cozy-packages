"""Managed SDXL transformations for ordinary client scripts."""

from __future__ import annotations

from typing import Literal

import cozy_runtime.derive as derive
from cozy_runtime.author import Context, ModelArtifact, Telemetry, WeightsSink, invocable

from . import SdxlModel


@invocable(memoize=True)
async def quantize_lane(
    ctx: Context,
    *,
    source: SdxlModel,
    encoding: Literal["fp8-rowwise/1", "mxfp8/1"],
    max_relative_frobenius: float | None = None,
    weights: WeightsSink,
    tel: Telemetry,
) -> ModelArtifact:
    """Derive one lane from the canonical source; reuse complete tensor groups."""
    return derive.quantize_artifact(
        source,
        derive.plan(("unet",), encoding, max_relative_frobenius=max_relative_frobenius),
        sink=weights,
        ctx=ctx,
        tel=tel,
    )
