"""Measure the exact Runtime adapters, without copying their kernels or quantizers.

This private diagnostic intentionally depends on a pinned Runtime source artifact.
It is not a public author API. Construction includes Runtime's small native smoke;
the timed closure includes all preprocessing for the captured full-size operands.
"""

from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
from typing import Any

import torch
from cozy_runtime.author import AttentionLayout, attention_scope
from cozy_runtime.internal import attention, attention_sol
from cozy_runtime.internal.encoding import measure_device
from diffusers.models.attention_dispatch import dispatch_attention_fn


def build_runtime_backend(
    name: str,
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    layout: AttentionLayout | None = None,
    module_path: str = "",
) -> tuple[Any, dict[str, Any]]:
    device = measure_device(torch, q.device.index or 0)
    entry = attention.pinned(name, device)[0]
    if name == "sol-attn" and (layout is None or not module_path):
        raise ValueError("Sol comparison requires the captured model layout and module path")
    provenance: dict[str, Any] = {
        "backend": name,
        "runtime_backend": entry.candidate.backend,
        "baked_location": entry.location,
        "device": asdict(device),
        "input_dtype": str(q.dtype),
        "shape": list(q.shape),
        "input_strides": [list(x.stride()) for x in (q, k, v)],
        "scale": scale,
        "layout": asdict(layout) if layout is not None else None,
        "module_path": module_path,
        "construction_includes": "Runtime resolution and small native smoke, including its JIT",
        "timing_includes": (
            "Runtime adapter dispatch, layouts, quantization, native kernel and contiguous NHD output"
        ),
    }
    if entry.candidate.distribution:
        distribution = metadata.distribution(entry.candidate.distribution)
        provenance["distribution"] = distribution.metadata["Name"]
        provenance["version"] = distribution.version
        provenance["direct_url"] = distribution.read_text("direct_url.json")

    if name != "sol-attn":
        return (
            lambda: dispatch_attention_fn(q, k, v, scale=scale, backend=entry.member).contiguous(),
            provenance,
        )

    assert layout is not None

    class Call(torch.nn.Module):  # type: ignore[misc]
        def forward(self) -> Any:
            return dispatch_attention_fn(q, k, v, scale=scale, backend=entry.member)

    call = Call()
    attention_sol.install_site(call, "fl2va_dit", module_path)
    provenance["dense_reference"] = attention_sol.dense_identity()

    def invoke() -> Any:
        with attention_scope(layout), attention_sol.observing() as counts:
            result = call()
        provenance["last_effective_calls"] = dict(counts)
        return result.contiguous()

    return invoke, provenance
