"""Private one-step profiler for ordinary H3 and generic Runtime model-slot LoRAs."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    Outputs,
    Telemetry,
    uses_components,
)

from h3 import FirstLastFrameToVideoInput, H3Model, KeyframeAssets, fl2va

app = App()
_ACTIVE: ContextVar[Capture | None] = ContextVar("h3_lora_profile", default=None)


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int = 7381
    duration_s: Literal[5, 15] = 15
    profile: bool = True


class Result(msgspec.Struct):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))
    ]
    trace: Annotated[FileAsset, AssetBound(max_bytes=64 << 20, media_types=("application/gzip",))]


class ProfileComplete(Exception):
    pass


class Capture:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.profiler: Any = None
        self.running = False
        self.trace = gzip.compress(b"{}")
        self.operators: list[dict[str, Any]] = []

    def after_step(self, index: int) -> None:
        # Start after step0's one-time warmup; stop immediately after step1.
        if not self.enabled:
            return
        if index == 0:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self.profiler.start()
            self.running = True
        elif index == 1:
            self.stop()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        profiler, self.profiler = self.profiler, None
        profiler.stop()
        self.operators = [
            {
                "operator": event.key,
                "shapes": event.input_shapes,
                "count": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "self_device_us": event.self_device_time_total,
                "cpu_us": event.cpu_time_total,
                "device_us": event.device_time_total,
            }
            for event in profiler.key_averages(group_by_input_shape=True)
        ]
        self.operators.sort(key=lambda row: row["self_device_us"], reverse=True)
        with tempfile.TemporaryDirectory(prefix="h3-lora-profile-") as directory:
            path = Path(directory) / "trace.json"
            profiler.export_chrome_trace(str(path))
            self.trace = gzip.compress(path.read_bytes())
        if len(self.trace) > 64 << 20:
            raise ValueError("one-step trace exceeds its retained output bound")


class ProfileModel(H3Model, encoded_leaves="accept", fusion="accept"):
    @uses_components("fl2va_dit")
    def sample_fl2va(self, state: Any, *, on_step: Any, cancel: Any, checks: Any) -> Any:
        capture = _ACTIVE.get()
        if capture is None:
            raise RuntimeError("profile context is absent")
        if torch.cuda.device_count() != 1:
            raise ValueError("this private profile requires one physical CUDA GPU")

        def step(index: int) -> None:
            on_step(index)
            capture.after_step(index)
            if index == 1:
                checks.settle()
                raise ProfileComplete()

        root = self.pipe.components["fl2va_dit"]
        checks.component("fl2va_dit", root)
        try:
            with checks.forwards(root, "fl2va_dit"):
                return self.pipe.denoise("fl2va", state, on_step=step, cancel=cancel, checks=checks)
        finally:
            capture.stop()


@app.entrypoint
def probe(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: ProfileModel,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    capture = Capture(payload.profile)
    token = _ACTIVE.set(capture)
    started = time.perf_counter()
    try:
        try:
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
        except ProfileComplete:
            pass
        else:
            raise RuntimeError("profile unexpectedly generated a full video")
    finally:
        _ACTIVE.reset(token)
        capture.stop()
    report = {
        "input": msgspec.to_builtins(payload),
        "profiled_step": 1 if payload.profile else None,
        "two_step_seconds": time.perf_counter() - started,
        "executed_steps": 2,
        "timing_scope": (
            "conditioning and two denoise steps with one-step profiler overhead; no full video"
        ),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "operators": capture.operators,
        "attention_override": "none; ordinary per-component Runtime selection",
        "sources": json.loads(Path(__file__).with_name("profile_sources.json").read_text()),
    }
    return Result(
        out.save_bytes(json.dumps(report, sort_keys=True).encode(), media_type="application/json"),
        out.save_bytes(capture.trace, media_type="application/gzip"),
    )
