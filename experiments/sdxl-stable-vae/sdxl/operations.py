"""SDXL policy for the shared Runtime quantization operation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, cast

from cozy_runtime.author import ModelArtifact
from cozy_runtime.derive.operations import QuantizationPlan
from cozy_runtime.derive.operations import quantize as runtime_quantize


def quantization_plan() -> QuantizationPlan:
    """Select unet GEMM weights with the family's existing F32 normalization."""
    return QuantizationPlan(components=("unet",))


async def quantize(
    *,
    source: ModelArtifact,
    encoding: Literal["fp8-rowwise/1", "mxfp8/1"],
    max_relative_frobenius: float | None = None,
) -> ModelArtifact:
    """Compose family policy with the single Runtime-owned memoized operation."""
    call = cast(Callable[..., Awaitable[ModelArtifact]], runtime_quantize)
    return await call(
        source=source,
        plan=quantization_plan(),
        encoding=encoding,
        max_relative_frobenius=max_relative_frobenius,
    )
