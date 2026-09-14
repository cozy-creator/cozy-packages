"""Compare complete attention operations on Q/K/V from the unchanged H3 graph."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import statistics
import tarfile
import tempfile
import time
import traceback
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import msgspec
import torch
from attention_backends import build_backend
from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    Context,
    FileAsset,
    Image,
    Outputs,
    Telemetry,
    uses_components,
)
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.internal import attention_sol
from cozy_runtime.models.minimax_h3 import H3Model
from diffusers.models import attention_dispatch as dispatch

from h3 import FirstLastFrameToVideoInput, fl2va

app = App()
KeyframeAssets = Annotated[Assets[Image], AssetLimits(images=2)]
_ACTIVE: ContextVar[Capture | None] = ContextVar("h3_attention_capture", default=None)


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    seed: int = 7101
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)] = 15
    capture_step: Annotated[int, msgspec.Meta(ge=0, le=29)] = 0
    capture_block: Annotated[int, msgspec.Meta(ge=0, le=49)] = 0
    backends: tuple[str, ...] = ("fa3_bf16", "cudnn_bf16")
    repeats: Annotated[int, msgspec.Meta(ge=3, le=10)] = 5
    profile: bool = True
    sequence_limit: Annotated[int, msgspec.Meta(ge=0, le=109104)] = 0
    reference_backend: str = "fa3_bf16"


class Result(msgspec.Struct):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))
    ]
    traces: Annotated[FileAsset, AssetBound(max_bytes=32 << 20, media_types=("application/gzip",))]
    status: str


class Captured(Exception):
    pass


class Capture:
    def __init__(self, payload: Input) -> None:
        self.payload = payload
        self.step = 0
        self.tensors: tuple[Any, Any, Any] | None = None
        self.scale = 0.0
        self.layout: Any = None
        self.module_path = ""

    def sample(self, root: Any, invoke: Any, on_step: Any) -> Any:
        references = {
            "fa3_bf16": "_flash_3_hub",
            "cudnn_bf16": "_native_cudnn",
            "sdpa": "native",
        }
        if self.payload.reference_backend not in references:
            raise ValueError("capture reference must be fa3_bf16, cudnn_bf16 or sdpa")
        member = dispatch.AttentionBackendName(references[self.payload.reference_backend])
        original = dispatch._AttentionBackendRegistry._backends[member]
        processors = [
            (path, module, module.processor, module.processor._attention_backend)
            for path, module in root.named_modules()
            if hasattr(module, "processor") and hasattr(module.processor, "_attention_backend")
        ]

        @functools.wraps(original)
        def capture(**kwargs: Any) -> Any:
            q, k, v = (kwargs[key] for key in ("query", "key", "value"))
            # Match the actual module path; prompt length cannot turn a text-refiner
            # call into a DiT block or shift the diagnostic block counter.
            site = attention_sol._SITE.get()
            parts = site.path.split(".") if site is not None else []
            if len(parts) > 1 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                block = int(parts[1])
                if self.step == self.payload.capture_step and block == self.payload.capture_block:
                    if kwargs.get("attn_mask") is not None or kwargs.get("is_causal", False):
                        raise ValueError("expected H3's dense noncausal attention")
                    if kwargs.get("_parallel_config") is not None:
                        raise ValueError("capture requires unsharded attention")
                    self.tensors = (q.detach(), k.detach(), v.detach())
                    self.scale = kwargs.get("scale") or 1 / math.sqrt(q.shape[-1])
                    self.layout = _ACTIVE_LAYOUT.get()
                    self.module_path = site.path if site is not None else ""
                    raise Captured()
            return original(**kwargs)

        def step(index: int) -> None:
            on_step(index)
            self.step = index + 1

        temporary_sites: list[tuple[str, Any]] = []
        # An explicit preparation pin can make an optional kernel available.
        # Capture always follows the same BF16 trajectory before comparing it.
        try:
            for path, module, processor, _ in processors:
                if getattr(module, "_cozy_sol_site", None) is None:
                    attention_sol.install_site(module, "fl2va_dit", path)
                    temporary_sites.append((path, module))
                processor._attention_backend = member
            dispatch._AttentionBackendRegistry._backends[member] = capture
            return invoke(step)
        finally:
            dispatch._AttentionBackendRegistry._backends[member] = original
            for _, _, processor, backend in processors:
                processor._attention_backend = backend
            for path, module in temporary_sites:
                attention_sol.remove_site(module, "fl2va_dit", path)


class OracleModel(H3Model, encoded_leaves="accept", fusion="accept"):
    @uses_components("fl2va_dit")
    def sample_fl2va(self, state: Any, *, on_step: Any, cancel: Any, checks: Any) -> Any:
        capture = _ACTIVE.get()
        if capture is None:
            raise RuntimeError("attention capture context is absent")
        root = self.pipe.components["fl2va_dit"]
        checks.component("fl2va_dit", root)
        with checks.forwards(root, "fl2va_dit"):
            return capture.sample(
                root,
                lambda step: self.pipe.denoise(
                    "fl2va", state, on_step=step, cancel=cancel, checks=checks
                ),
                on_step,
            )


def fingerprint(value: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    # Bounded host copies preserve original element order without a full CPU clone.
    for batch in value:
        for rows in batch.split(512):
            digest.update(rows.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "sha256": digest.hexdigest(),
    }


def reference_rows(q: Any, k: Any, v: Any, scale: float) -> tuple[Any, Any]:
    """FP32 reference on 16 evenly spread query rows, retaining all keys/values."""
    rows = torch.linspace(0, q.shape[1] - 1, 16, device=q.device).long().unique()
    outputs = []
    for head in range(q.shape[2]):
        query = q[:, rows, head].float()
        keys, values = k[:, :, head].float(), v[:, :, head].float()
        probabilities = torch.softmax((query @ keys.transpose(-2, -1)) * scale, dim=-1)
        outputs.append(probabilities @ values)
    return rows, torch.stack(outputs, dim=2)


def delta(actual: Any, expected: Any) -> dict[str, Any]:
    # Row chunks avoid allocating several full-size FP32 activation copies.
    error_square = total_square = actual_square = dot = absmax = 0.0
    finite = True
    equal = True
    for start in range(0, actual.shape[1], 256):
        a, b = actual[:, start : start + 256].float(), expected[:, start : start + 256].float()
        finite = finite and bool(torch.isfinite(a).all())
        if not finite:
            return {"finite": False, "equal_values": False, "max_abs": None, "relative_l2": None}
        equal = equal and torch.equal(a, b)
        error = a - b
        absmax = max(absmax, float(error.abs().max()))
        error_square += float(error.double().square().sum())
        total_square += float(b.double().square().sum())
        actual_square += float(a.double().square().sum())
        dot += float((a.double() * b.double()).sum())
    return {
        "finite": finite,
        "equal_values": equal,
        "max_abs": absmax,
        "relative_l2": math.sqrt(error_square / max(total_square, 1e-300)),
        "norm_ratio": math.sqrt(actual_square / max(total_square, 1e-300)),
        "cosine": dot / math.sqrt(max(actual_square * total_square, 1e-300)),
    }


def timed(call: Any) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    result = call()
    end.record()
    end.synchronize()
    return result, time.perf_counter() - started, begin.elapsed_time(end)


@app.entrypoint
def probe(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: OracleModel,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    if torch.cuda.device_count() != 1:
        raise ValueError("the attention oracle requires one physical CUDA GPU")
    capture = Capture(payload)
    token = _ACTIVE.set(capture)
    started = time.perf_counter()
    try:
        with contextlib.suppress(Captured):
            fl2va(
                ctx,
                FirstLastFrameToVideoInput(
                    prompt=payload.prompt,
                    seed=payload.seed,
                    duration_s=payload.duration_s,
                    steps=30,
                ),
                assets,
                model,
                out,
                tel,
            )
    finally:
        _ACTIVE.reset(token)
    if capture.tensors is None:
        raise RuntimeError("the requested real H3 attention site was not captured")
    q, k, v = capture.tensors
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    model_qkv_shape = list(q.shape)
    if payload.sequence_limit:
        if "sol-attn" in payload.backends:
            raise ValueError("Sol comparison requires the full captured document layout")
        if payload.sequence_limit > q.shape[1]:
            raise ValueError("sequence_limit exceeds the captured model sequence")
        q, k, v = (x[:, : payload.sequence_limit] for x in (q, k, v))
    originals = [fingerprint(x) for x in (q, k, v)]
    reference_call, reference_provenance = build_backend(
        payload.reference_backend, q, k, v, scale=capture.scale
    )
    reference = reference_call()
    rows, fp32 = reference_rows(q, k, v, capture.scale)
    paths: list[Path] = []
    results: list[dict[str, Any]] = []
    for name in payload.backends:
        record: dict[str, Any] = {"name": name}
        try:
            began = time.perf_counter()
            call, provenance = build_backend(
                name,
                q,
                k,
                v,
                scale=capture.scale,
                layout=capture.layout,
                module_path=capture.module_path,
            )
            record.update(provenance=provenance, construction_seconds=time.perf_counter() - began)
            result, first_wall, first_cuda = timed(call)
            record.update(first_call_seconds=first_wall, first_call_cuda_ms=first_cuda)
            # The second call heats the device and fills per-operation caches.
            result, _, _ = timed(call)
            times = []
            for _ in range(payload.repeats):
                result, wall, cuda_ms = timed(call)
                times.append({"wall_seconds": wall, "cuda_ms": cuda_ms})
            record.update(
                times=times,
                median_wall_seconds=statistics.median(x["wall_seconds"] for x in times),
                median_cuda_ms=statistics.median(x["cuda_ms"] for x in times),
            )
            if result.shape != q.shape or result.dtype != torch.bfloat16:
                raise ValueError("attention did not return the expected BF16 NHD tensor")
            record["against_reference"] = delta(result, reference)
            record["against_fp32_sampled_rows"] = delta(result[:, rows], fp32)
            repeated = call()
            record["repeatability"] = delta(repeated, result)
            del repeated, result
            if payload.profile:
                path = Path(tempfile.mkdtemp(prefix="h3-attention-oracle-")) / f"{name}.json"
                with (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler,
                    torch.profiler.record_function(name),
                ):
                    call()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(path))
                paths.append(path)
            record["status"] = "completed"
        except Exception:
            record.update(status="error", error=traceback.format_exc())
        results.append(record)
        tel.log(
            "attention oracle arm",
            backend=name,
            status=record["status"],
            median_cuda_ms=record.get("median_cuda_ms"),
        )
    after = [fingerprint(x) for x in (q, k, v)]
    if originals != after:
        raise RuntimeError("an attention candidate modified the captured Q/K/V inputs")
    registry = dispatch._AttentionBackendRegistry._backends
    document = {
        "format": "h3.attention-oracle/2",
        "request_id": ctx.request_id,
        "input": msgspec.to_builtins(payload),
        "checkpoint": model.checkpoint_ref,
        "pid": os.getpid(),
        "gpu": torch.cuda.get_device_name(),
        "capture_seconds": capture_seconds,
        "model_qkv_shape": model_qkv_shape,
        "sequence_scope": "captured prefix ablation"
        if payload.sequence_limit
        else "complete model sequence",
        "capture_backend": payload.reference_backend,
        "captured_attention_layout": asdict(capture.layout) if capture.layout is not None else None,
        "captured_module_path": capture.module_path,
        "qkv": originals,
        "scale": capture.scale,
        "reference": reference_provenance,
        "reference_rows": rows.cpu().tolist(),
        "reference_vs_fp32": delta(reference[:, rows], fp32),
        "registry": {
            str(key): {
                "function": f"{fn.__module__}.{fn.__qualname__}",
                "source_file": inspect.getsourcefile(fn),
            }
            for key, fn in registry.items()
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "cozy-runtime", "tensorfs", "triton")
        },
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "results": results,
    }
    raw = json.dumps(document, indent=2, allow_nan=False).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.name)
    return Result(
        measurements=out.save_bytes(raw, media_type="application/json"),
        traces=out.save_bytes(buffer.getvalue(), media_type="application/gzip"),
        status="completed" if all(x["status"] == "completed" for x in results) else "partial",
    )
