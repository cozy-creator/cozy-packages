"""Private matched-video comparison; the published H3 generation function is unchanged."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal

import msgspec
import torch
from attention_backends import build_backend, load_fa3
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    Outputs,
    Telemetry,
    uses_components,
)
from diffusers.models import attention_dispatch as dispatch

from h3 import FirstLastFrameToVideoInput, H3Model, H3VideoOutput, KeyframeAssets, fl2va

app = App()
_ACTIVE: ContextVar[Selection | None] = ContextVar("h3_quality_selection", default=None)
_SITE: ContextVar[tuple[Selection, int] | None] = ContextVar("h3_quality_site", default=None)
_LOCK = threading.Lock()


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int = 7101
    duration_s: Literal[5, 15] = 5
    backend: Literal["fa3_bf16", "fa3_tile128_fp8", "sage2_sm90"] = "fa3_bf16"


class Result(H3VideoOutput):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))
    ]


class Selection:
    def __init__(self, payload: Input, cancel: Callable[[], None]) -> None:
        self.payload, self.cancel = payload, cancel
        self.calls = [0] * 50
        self.provenance: dict[str, Any] = {}
        self.denoise_seconds = 0.0
        self.denoise_cuda_ms = 0.0
        self.shape: list[int] = []

    def attention(self, original: Any, block: int, kwargs: dict[str, Any]) -> Any:
        self.cancel()
        q, k, v = (kwargs[key] for key in ("query", "key", "value"))
        if (
            kwargs.get("attn_mask") is not None
            or kwargs.get("is_causal", False)
            or kwargs.get("dropout_p", 0.0) != 0.0
            or kwargs.get("_parallel_config") is not None
        ):
            raise ValueError("quality comparison requires unsharded dense noncausal attention")
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-2:] != (56, 128):
            raise ValueError("unexpected H3 main-block attention geometry")
        if any(x.dtype != torch.bfloat16 or not x.is_cuda for x in (q, k, v)):
            raise ValueError("quality comparison requires BF16 CUDA Q/K/V")
        if self.payload.backend == "fa3_bf16":
            # Preserve the exact serving backend call, including its normal dispatch arguments.
            result = original(**kwargs)
            if not self.provenance:
                source = inspect.getsourcefile(original)
                self.provenance = {
                    "backend": "fa3_bf16",
                    "function": f"{original.__module__}.{original.__qualname__}",
                    "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
                    if source
                    else None,
                    "kernel": load_fa3()[1],
                }
        else:
            call, facts = build_backend(
                self.payload.backend, q, k, v, scale=kwargs.get("scale") or 1 / math.sqrt(128)
            )
            result = call()
            if not self.provenance:
                self.provenance = facts
        if (
            not isinstance(result, torch.Tensor)
            or result.shape != q.shape
            or result.dtype != q.dtype
        ):
            raise ValueError("selected attention did not return H3's BF16 output geometry")
        self.calls[block] += 1
        self.shape = list(q.shape)
        return result

    @contextmanager
    def scope(self, root: Any) -> Iterator[None]:
        if torch.cuda.device_count() != 1:
            raise ValueError("this quality comparison requires one physical CUDA GPU")
        modules = [block.attn for block in root.transformer_blocks]
        if len(modules) != 50 or len({id(module.processor) for module in modules}) != 50:
            raise ValueError("expected 50 independently owned H3 main attention processors")
        member = dispatch.AttentionBackendName._FLASH_3_HUB
        original = dispatch._AttentionBackendRegistry._backends[member]
        # The CLI kernel pin applies to every component, including H3's FP32/D256
        # audio VAE. Scope BF16 to this DiT instead; its two tiny refiner sites
        # remain BF16 while only the marked main blocks enter the candidate.
        processors = {
            id(module.processor): (module.processor, module.processor._attention_backend)
            for module in root.modules()
            if hasattr(module, "processor") and hasattr(module.processor, "_attention_backend")
        }
        if not _LOCK.acquire(blocking=False):
            raise RuntimeError("another private attention override is already active")
        handles: list[Any] = []
        token = _SITE.set(None)

        @functools.wraps(original)
        def selected(**kwargs: Any) -> Any:
            site = _SITE.get()
            if site is None or site[0] is not self:
                return original(**kwargs)
            return self.attention(original, site[1], kwargs)

        def before(index: int, _module: Any, _args: Any) -> None:
            if _SITE.get() is not None:
                raise RuntimeError("nested main-block attention is unsupported")
            _SITE.set((self, index))

        def after(_module: Any, _args: Any, _output: Any) -> None:
            _SITE.set(None)

        try:
            for processor, _ in processors.values():
                processor._attention_backend = member
            for index, module in enumerate(modules):
                handles.append(module.register_forward_pre_hook(functools.partial(before, index)))
                handles.append(module.register_forward_hook(after, always_call=True))
            dispatch._AttentionBackendRegistry._backends[member] = selected
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            start.record()
            try:
                yield
            finally:
                self.denoise_seconds = time.perf_counter() - began
                end.record()
                end.synchronize()
                self.denoise_cuda_ms = start.elapsed_time(end)
        finally:
            dispatch._AttentionBackendRegistry._backends[member] = original
            for handle in handles:
                handle.remove()
            for processor, backend in processors.values():
                processor._attention_backend = backend
            _SITE.reset(token)
            _LOCK.release()


class QualityModel(H3Model, encoded_leaves="accept", fusion="accept"):
    @uses_components("fl2va_dit")
    def sample_fl2va(self, state: Any, *, on_step: Any, cancel: Any, checks: Any) -> Any:
        selection = _ACTIVE.get()
        if selection is None:
            raise RuntimeError("quality request context is absent")
        root = self.pipe.components["fl2va_dit"]
        # This override already owns the public component-use scope. Calling the
        # decorated parent method would reenter it; retain its exact sample body.
        with selection.scope(root):
            checks.component("fl2va_dit", root)
            with checks.forwards(root, "fl2va_dit"):
                return self.pipe.denoise(
                    "fl2va", state, on_step=on_step, cancel=cancel, checks=checks
                )


@app.entrypoint
def generate(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: QualityModel,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    selection = Selection(payload, ctx.raise_if_cancelled)
    token = _ACTIVE.set(selection)
    began = time.perf_counter()
    completed = False
    try:
        video = fl2va(
            ctx,
            FirstLastFrameToVideoInput(
                prompt=payload.prompt, seed=payload.seed, duration_s=payload.duration_s, steps=30
            ),
            assets,
            model,
            out,
            tel,
        )
        if selection.calls != [30] * 50:
            raise RuntimeError(f"expected 30 selected calls per main block: {selection.calls}")
        completed = True
    finally:
        _ACTIVE.reset(token)
        tel.log(
            "h3 attention quality execution",
            backend=payload.backend,
            applied_calls=sum(selection.calls),
            completed=completed,
            generation_seconds=time.perf_counter() - began,
        )
    report = {
        "input": msgspec.to_builtins(payload),
        "steps": 30,
        "selected_component": "fl2va_dit",
        "selected_sites": "transformer_blocks.0..49.attn; token refiner remains BF16",
        "applied_calls": sum(selection.calls),
        "calls_per_block": selection.calls,
        "shape": selection.shape,
        "backend": selection.provenance,
        "sources": json.loads(Path(__file__).with_name("quality_sources.json").read_text()),
        "generation_seconds": time.perf_counter() - began,
        "denoise_seconds": selection.denoise_seconds,
        "denoise_cuda_ms": selection.denoise_cuda_ms,
        "timing_scope": "ordinary generation; denoise includes checks and per-call backend setup",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    measurements = out.save_bytes(
        json.dumps(report, sort_keys=True).encode(), media_type="application/json"
    )
    return Result(video.video, video.continuation_frame, video.warnings, measurements)
