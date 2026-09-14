"""Explicit attention candidates for the private H3 oracle; never select a fallback.

Inputs and outputs use [batch, sequence, heads, head_dim]. Every returned call
starts from BF16 Q/K/V and includes its required conversions. Optional kernels
must already be installed or loaded by the worker; this module downloads nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import math
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from attention_quantized import OptionalBackendUnavailable, build_quantized, candidate_kernel
from diffusers.models import attention_dispatch as ad
from torch.nn.attention import SDPBackend, sdpa_kernel


def _function_provenance(fn: Callable[..., Any]) -> dict[str, Any]:
    module = inspect.getmodule(fn)
    path = getattr(module, "__file__", None)
    return {
        "module": getattr(fn, "__module__", None),
        "function": getattr(fn, "__qualname__", None),
        "source": path,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
    }


def load_fa3() -> tuple[Callable[..., Any], dict[str, Any]]:
    """Reuse the exact FA3 function already loaded for Diffusers serving."""
    config = ad._HUB_KERNELS_REGISTRY[ad.AttentionBackendName._FLASH_3_HUB]
    fn = config.kernel_fn
    if not callable(fn):
        raise OptionalBackendUnavailable("FA3 is not already loaded in the Diffusers hub registry")
    return fn, {"load": "diffusers_loaded_flash_3_hub", **_function_provenance(fn)}


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> None:
    if q.ndim != 4 or any(x.shape != q.shape for x in (k, v)) or min(q.shape) < 1:
        raise ValueError("H3 oracle requires equal nonempty Q/K/V shapes [B,N,H,D]")
    if any(x.device != q.device or x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("H3 oracle requires BF16 Q/K/V on the same device")
    if q.device.type != "cuda":
        raise OptionalBackendUnavailable("attention candidates require CUDA")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be positive and finite")


def _flash_call(
    fn: Callable[..., Any], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    # Flash kernels accept strided tensors with contiguous head dimensions.
    inputs = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]
    result = fn(*inputs, softmax_scale=scale, causal=False, num_splits=1)
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError("Flash attention returned no output tensor")
    return result.to(torch.bfloat16).contiguous()


def build_backend(
    name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    """Build one named backend; caller times construction, first call, and warm calls."""
    _validate(q, k, v, scale)
    metadata: dict[str, Any] = {
        "backend": name,
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "layout": "BNHD",
        "shape": list(q.shape),
        "input_strides": [list(x.stride()) for x in (q, k, v)],
        "scale": scale,
        "causal": False,
        "dropout_p": 0.0,
        "num_splits": 1 if name.startswith("fa") else None,
        "torch": torch.__version__,
        "timing_includes": "required input layout conversion and output BF16 contiguous conversion",
    }
    if name == "fa3_bf16":
        if torch.cuda.get_device_capability(q.device)[0] != 9:
            raise OptionalBackendUnavailable("the baked FA3 candidate requires Hopper SM90")
        fn, provenance = load_fa3()
        return lambda: _flash_call(fn, q, k, v, scale), {**metadata, **provenance}
    if name == "fa4_bf16":
        try:
            module = importlib.import_module("flash_attn.cute")
            fn = module.flash_attn_func
        except (ImportError, OSError, AttributeError) as exc:
            raise OptionalBackendUnavailable(f"FA4 is not importable: {exc}") from exc
        try:
            version = importlib.metadata.version("flash-attn-4")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return lambda: _flash_call(fn, q, k, v, scale), {
            **metadata,
            **_function_provenance(fn),
            "flash_attn_4": version,
        }
    if name in {"fa3_rebuilt_bf16", "fa3_tile128_bf16"}:
        fn, provenance = candidate_kernel("twolevel" if name == "fa3_rebuilt_bf16" else "tile128")
        return lambda: _flash_call(fn, q, k, v, scale), {
            **metadata,
            **provenance,
            "diagnostic_control": True,
        }
    if name == "cudnn_bf16":
        params = torch.backends.cuda.SDPAParams(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, 0.0, False, False
        )
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            if not torch.backends.cuda.can_use_cudnn_attention(params, debug=True):
                raise OptionalBackendUnavailable("cuDNN BF16 rejects the captured shape or layout")

        def call() -> torch.Tensor:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                )
            return out.transpose(1, 2).contiguous()

        read_cudnn_version: Callable[[], int | None] = torch.backends.cudnn.version
        return call, {
            **metadata,
            "cudnn_backend": read_cudnn_version(),
            "allowed_sdpa_backends": ["CUDNN_ATTENTION"],
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
    if name == "cudnn_fp8":
        call, provenance = _build_cudnn_fp8(q, k, v, scale)
        return call, {**metadata, **provenance}
    if name in (
        "sage2_sm90",
        "fa3_fp8",
        "fa3_qkv_roundtrip",
        "fa3_q_roundtrip",
        "fa3_k_roundtrip",
        "fa3_v_roundtrip",
        "fa3_k_center_bf16",
        "fa3_k_center_roundtrip",
        "fa3_k_center_tile128_fp8",
        "fa3_fp8_splits2",
        "fa3_fp8_splits4",
        "fa3_twolevel_fp8",
        "fa3_tile128_fp8",
    ):
        call, provenance = build_quantized(name, q, k, v, scale=scale)
        return call, {**metadata, **provenance}
    raise ValueError(f"unknown attention backend: {name}")


def _quantize_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Power-of-two scale as recommended by cuDNN. FP32 arithmetic precedes the
    # saturating E4M3 cast; scale/descale remain device scalars, with no .item().
    amax = x.abs().amax().float().clamp_min(1e-8)
    multiplier = torch.exp2(torch.floor(torch.log2(448.0 / amax)))
    quantized = (x.float() * multiplier).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.contiguous(), multiplier.reciprocal().reshape(1, 1, 1, 1)


def _build_cudnn_fp8(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    try:
        cudnn = importlib.import_module("cudnn")
    except (ImportError, OSError) as exc:
        raise OptionalBackendUnavailable(f"cuDNN frontend is not importable: {exc}") from exc
    architecture = torch.cuda.get_device_capability(q.device)[0]
    if architecture not in (9, 10):
        raise OptionalBackendUnavailable("cuDNN FP8 oracle supports SM90 and SM100")
    if q.shape[-1] % 16:
        raise OptionalBackendUnavailable("cuDNN FP8 requires head dimension divisible by 16")
    handle = cudnn.create_handle()
    try:
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(q.device).cuda_stream)
        e4m3, fp32 = cudnn.data_type.FP8_E4M3, cudnn.data_type.FLOAT
        graph = cudnn.pygraph(
            io_data_type=e4m3,
            intermediate_data_type=fp32,
            compute_data_type=fp32,
            handle=handle,
        )
        batch, sequence, heads, dimension = q.shape
        shape = (batch, heads, sequence, dimension)
        strides = (sequence * heads * dimension, dimension, heads * dimension, 1)
        tq, tk, tv = [
            graph.tensor(name=n, dim=shape, stride=strides, data_type=e4m3) for n in ("q", "k", "v")
        ]
        scalar = dict(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=fp32)
        dq, dk, dv, ds, ss, so = [
            graph.tensor(name=n, **scalar) for n in ("dq", "dk", "dv", "ds", "ss", "so")
        ]
        out, _, amax_s, amax_o = graph.sdpa_fp8(
            q=tq,
            k=tk,
            v=tv,
            descale_q=dq,
            descale_k=dk,
            descale_v=dv,
            descale_s=ds,
            scale_s=ss,
            scale_o=so,
            attn_scale=scale,
            use_causal_mask=False,
            generate_stats=False,
        )
        # cuDNN 9.13+ supports BF16 output on Blackwell; Hopper requires FP8 O.
        bf16_output = architecture == 10
        out.set_output(True).set_dim(shape).set_stride(strides).set_data_type(
            cudnn.data_type.BFLOAT16 if bf16_output else e4m3
        )
        for amax in (amax_s, amax_o):
            amax.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(fp32)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans()
        workspace_bytes = graph.get_workspace_size()
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
        softmax_scale = torch.full((1, 1, 1, 1), 256.0, dtype=torch.float32, device=q.device)
        softmax_descale = softmax_scale.reciprocal()
        output_one = torch.ones_like(softmax_scale)
    except Exception as exc:
        cudnn.destroy_handle(handle)
        raise OptionalBackendUnavailable(f"cuDNN FP8 plan construction failed: {exc}") from exc

    def call() -> torch.Tensor:
        (q8, qs), (k8, ks), (v8, vs) = [_quantize_tensor(x) for x in (q, k, v)]
        # Half the V scale leaves headroom for rounding in quantized P/V. This
        # bound is conservative, not a calibrated optimal output scale.
        output_scale = output_one if bf16_output else vs.reciprocal() * 0.5
        output = torch.empty_like(q8, dtype=torch.bfloat16 if bf16_output else torch.float8_e4m3fn)
        pack = {
            tq: q8,
            tk: k8,
            tv: v8,
            dq: qs,
            dk: ks,
            dv: vs,
            ds: softmax_descale,
            ss: softmax_scale,
            so: output_scale,
            out: output,
            amax_s: torch.empty_like(output_one),
            amax_o: torch.empty_like(output_one),
        }
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(q.device).cuda_stream)
        graph.execute(pack, workspace, handle=handle)
        return output if bf16_output else (output.float() / output_scale).to(torch.bfloat16)

    weakref.finalize(call, cudnn.destroy_handle, handle)
    return call, {
        "cudnn_frontend": getattr(cudnn, "__version__", None),
        "cudnn_backend": cudnn.backend_version_string(),
        "workspace_bytes": workspace_bytes,
        "native_output_dtype": "bfloat16" if bf16_output else "float8_e4m3fn",
        "qkv_quantization": (
            "per-tensor dynamic power-of-two amax scaling; FP32 multiply and saturating E4M3 cast"
        ),
        "softmax_scale": 256.0,
        "softmax_scale_policy": "conservative fixed bound P<=1; uncalibrated for H3",
        "output_scale_policy": (
            "1" if bf16_output else "half the dynamically computed V quantization scale"
        ),
        "timing_includes": (
            "Q/K/V amax, scaling, quantization, cuDNN execution, "
            "output allocation and BF16 dequantization"
        ),
    }
