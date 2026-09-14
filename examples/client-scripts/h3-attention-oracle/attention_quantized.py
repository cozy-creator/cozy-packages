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

import cozy_runtime.internal.attention_fp8 as production_fp8
import torch
from diffusers.models import attention_dispatch as dispatch


class OptionalBackendUnavailable(RuntimeError):
    """The requested backend is unavailable in this worker configuration."""


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
    if name not in {
        "sage2_sm90",
        "fa3_fp8",
        "fa3_qkv_roundtrip",
        "fa3_fp8_splits2",
        "fa3_fp8_splits4",
        "fa3_twolevel_fp8",
    }:
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

    backend_name = "_flash_3_hub_fp8"
    kernel = dispatch._HUB_KERNELS_REGISTRY[dispatch.AttentionBackendName._FLASH_3_HUB].kernel_fn
    if not callable(kernel):
        raise OptionalBackendUnavailable("The measured production FA3 kernel is not loaded")
    if name == "fa3_twolevel_fp8":
        module = importlib.import_module("h3_fa3_fp8_e4ad1ed05262")
        candidate = module.flash_attn_func

        def twolevel_call() -> Any:
            (q8, qs), (k8, ks), (v8, vs) = [production_fp8.quantise(x) for x in (q, k, v)]
            return candidate(
                q8,
                k8,
                v8,
                softmax_scale=scale,
                causal=False,
                num_splits=1,
                q_descale=qs,
                k_descale=ks,
                v_descale=vs,
            )

        return twolevel_call, {
            **common,
            **_source(candidate, "cozy-h3-fa3-fp8-candidate"),
            "experimental": True,
            "candidate_source": "e4ad1ed052626bdee606345371cb9f4e376c786e",
            "pv_accumulation": "completed tensor-core tile plus separate FP32 running sum",
            "preprocessing": "Runtime production per-head quantization inside each call",
        }
    if name in {"fa3_fp8_splits2", "fa3_fp8_splits4"}:
        splits = int(name[-1])
        free, _ = torch.cuda.mem_get_info(q.device)
        reusable = torch.cuda.memory_reserved(q.device) - torch.cuda.memory_allocated(q.device)
        required = q.numel() * (4 * splits + 3 + 2) + q.shape[0] * q.shape[1] * q.shape[2] * 4 * (
            splits + 1
        )
        if required + (1 << 30) > free + reusable:
            raise OptionalBackendUnavailable(
                "split-KV scratch exceeds available memory with a 1 GiB margin"
            )

        def split_call() -> Any:
            (q8, qs), (k8, ks), (v8, vs) = [production_fp8.quantise(x) for x in (q, k, v)]
            return kernel(
                q8,
                k8,
                v8,
                softmax_scale=scale,
                causal=False,
                num_splits=splits,
                q_descale=qs,
                k_descale=ks,
                v_descale=vs,
            )

        return split_call, {
            **common,
            "experimental": True,
            "num_splits": splits,
            "estimated_extra_bytes": required,
            "available_and_reusable_bytes": free + reusable,
            "preprocessing": "Runtime production per-head quantization inside each call",
        }
    if name == "fa3_qkv_roundtrip":

        def roundtrip_call() -> Any:
            decoded = []
            for value in (q, k, v):
                codes, descale = production_fp8.quantise(value)
                decoded.append((codes.float() * descale[:, None, :, None]).to(value.dtype))
            return kernel(*decoded, softmax_scale=scale, causal=False, num_splits=1)

        return roundtrip_call, {
            **common,
            "diagnostic_control": True,
            "attention_compute": "BF16 after production FP8 Q/K/V quantize-dequantize",
            "purpose": "separate QKV quantization error from FP8 attention arithmetic",
            **_source(production_fp8.quantise, "cozy-runtime"),
        }
    # Explicit diagnostic coupling: measure the real quantizer on the same inputs.
    production_fp8.bind(kernel)
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
        "registration": "diagnostic binds production quantizer to the loaded FA3 kernel",
        "input_contiguous": [x.is_contiguous() for x in (q, k, v)],
    }
