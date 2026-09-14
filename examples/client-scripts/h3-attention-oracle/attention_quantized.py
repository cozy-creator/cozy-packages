"""Complete BF16-input quantized attention calls for the private H3 oracle.

Quantization, smoothing, padding and layout work remain inside the timed call.
Neither backend promises BF16-equivalent output, and neither falls back to a
different attention implementation when its requested kernel is unavailable.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _source(function: Any, distribution: str) -> dict[str, Any]:
    filename = inspect.getsourcefile(function)
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "distribution": distribution,
        "version": version,
        "function": f"{function.__module__}.{function.__qualname__}",
        "source_file": filename,
        "source_sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if filename
        else None,
    }


def build_quantized(
    name: str, q: Any, k: Any, v: Any, *, scale: float
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Bind one dense, noncausal, SM90 attention operation in NHD layout.

    Construction resolves the requested kernel without running GPU work. The
    returned call performs fresh quantization from the original BF16 Q/K/V on
    every invocation, including any preprocessing allocations.
    """
    import torch
    from attention_backends import OptionalBackendUnavailable

    if name not in {"sage2_sm90", "fa3_fp8"}:
        raise ValueError(f"Unknown quantized attention backend: {name}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be finite and positive")
    if any(x.dtype != torch.bfloat16 or not x.is_cuda for x in (q, k, v)):
        raise ValueError("The quantized attention oracle requires BF16 CUDA Q/K/V")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("The H3 oracle requires matching [batch, sequence, heads, 128] Q/K/V")
    if any(x.device != q.device or x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("Q/K/V must share a device and have contiguous head dimensions")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise OptionalBackendUnavailable(f"{name} in this oracle requires an SM90 GPU")

    common = {
        "backend": name,
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "layout": "NHD",
        "causal": False,
        "scale": scale,
        "timing_scope": "BF16 Q/K/V through preprocessing and attention to BF16 output",
        "prequantized_inputs": False,
        "approximate": True,
    }
    if name == "sage2_sm90":
        try:
            sage = importlib.import_module("sageattention")
        except (ImportError, OSError, RuntimeError) as error:
            raise OptionalBackendUnavailable(
                f"SageAttention cannot be imported: {error}"
            ) from error
        sage_function = getattr(sage, "sageattn_qk_int8_pv_fp8_cuda_sm90", None)
        if not callable(sage_function):
            raise OptionalBackendUnavailable(
                "SageAttention's SM90 INT8/FP8 entrypoint is unavailable"
            )

        def sage_call() -> Any:
            return sage_function(
                q,
                k,
                v,
                tensor_layout="NHD",
                is_causal=False,
                sm_scale=scale,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp32+fp32",
                smooth_k=True,
                return_lse=False,
            )

        return sage_call, {
            **common,
            **_source(sage_function, "sageattention"),
            "qk_quantization": "INT8 per-thread",
            "pv_quantization": "FP8 E4M3",
            "pv_accumulation": "fp32+fp32",
            "smooth_k": True,
            "v_padding_tokens": (-q.shape[1]) % 128,
            "preprocessing": "SageAttention SM90 public wrapper, included in each call",
        }

    dispatch = importlib.import_module("diffusers.models.attention_dispatch")
    backend_name = "_flash_3_hub_fp8"
    try:
        backend = dispatch.AttentionBackendName(backend_name)
    except ValueError as error:
        raise OptionalBackendUnavailable(
            "The worker has not registered its production FA3 FP8 backend in Diffusers; "
            "the oracle will not substitute an eager quantizer"
        ) from error
    function = dispatch._AttentionBackendRegistry._backends.get(backend)
    if not callable(function):
        raise OptionalBackendUnavailable(
            "The production FA3 FP8 backend has no registered implementation"
        )

    def fp8_call() -> Any:
        return dispatch.dispatch_attention_fn(
            q, k, v, backend=backend, scale=scale, is_causal=False
        )

    return fp8_call, {
        **common,
        **_source(function, "cozy-runtime"),
        "diffusers_backend": backend_name,
        "qkv_quantization": "Runtime production per-(batch, head) FP8 E4M3",
        "preprocessing": "Registered production wrapper, included in each call",
        "quantizer_path": "production wrapper selects fused or eager from actual input strides",
        "input_contiguous": [x.is_contiguous() for x in (q, k, v)],
    }
