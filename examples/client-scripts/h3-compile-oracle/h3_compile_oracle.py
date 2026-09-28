"""Private single-GPU compiler experiment over the unchanged H3 serving source."""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
import tarfile
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Callable
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
from torch._dynamo.utils import compile_times, counters

from h3 import FirstLastFrameToVideoInput, H3Model, H3VideoOutput, KeyframeAssets, fl2va

app = App()
_ACTIVE: ContextVar[RunState | None] = ContextVar("h3_compile_oracle", default=None)


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    duration_s: int
    seed: int = 7101
    mode: Literal["eager", "blocks", "dit"] = "eager"
    autotune: bool = False
    cold: bool = False
    profile: bool = False
    warmup_steps: Annotated[int, msgspec.Meta(ge=0, le=2)] = 0


class ProbeResult(msgspec.Struct):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))
    ]
    compiler_artifacts: Annotated[
        FileAsset, AssetBound(max_bytes=128 << 20, media_types=("application/gzip",))
    ]
    status: str


class GenerationResult(H3VideoOutput):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))
    ]
    compiler_artifacts: Annotated[
        FileAsset, AssetBound(max_bytes=128 << 20, media_types=("application/gzip",))
    ]


class DenoisingComplete(Exception):
    pass


class CompilerState:
    def __init__(self, root: Any, payload: Input) -> None:
        self.key = (payload.mode, payload.autotune)
        self.root = Path(tempfile.mkdtemp(prefix="h3-compile-oracle-"))
        self.graphs: list[dict[str, Any]] = []
        self.modules = list(root.transformer_blocks) if payload.mode == "blocks" else [root]
        self.original = [module.forward for module in self.modules]
        self.compiled: list[Any] = []
        # Runtime owns and seals these per-content paths before CUDA starts.
        # Observe them; changing them inside a model correctly breaks the seal.
        self.cache_roots = {
            name: Path(os.environ[key])
            for name, key in (
                ("inductor", "TORCHINDUCTOR_CACHE_DIR"),
                ("triton", "TRITON_CACHE_DIR"),
            )
            if os.environ.get(key)
        }
        self.cache_files_before = {
            name: [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
            for name, root in self.cache_roots.items()
        }
        if payload.mode != "eager":
            if "inductor" not in self.cache_roots:
                raise RuntimeError("compiler cache must be imposed before executor startup")
            if payload.cold and self.cache_files_before["inductor"]:
                raise RuntimeError("cold compile requires a fresh content-scoped Inductor cache")
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()

            def backend(graph: Any, inputs: list[Any]) -> Any:
                index = len(self.graphs)
                record = {
                    "index": index,
                    "nodes": len(list(graph.graph.nodes)),
                    "targets": dict(
                        Counter(
                            str(node.target)
                            for node in graph.graph.nodes
                            if node.op == "call_function"
                        )
                    ),
                }
                self.graphs.append(record)
                (self.root / f"fx_graph_{index:04d}.py").write_text(graph.code)
                started = time.perf_counter()
                try:
                    return torch._inductor.compile(
                        graph,
                        inputs,
                        options={"max_autotune": payload.autotune, "triton.cudagraphs": False},
                    )
                finally:
                    record["backend_compile_seconds"] = time.perf_counter() - started

            self.compiled = [
                torch.compile(forward, backend=backend, fullgraph=False, dynamic=False)
                for forward in self.original
            ]

    def install(self, compiled: bool) -> None:
        forwards = self.compiled if compiled else self.original
        for module, forward in zip(self.modules, forwards, strict=True):
            module.forward = forward


class RunState:
    def __init__(self, payload: Input, stop_after_steps: int) -> None:
        self.payload = payload
        self.stop_after_steps = stop_after_steps
        self.steps: list[float] = []
        self.compiler: CompilerState | None = None
        self.graphs_before = 0
        self.profile: Any = None
        self.profile_path: Path | None = None
        self.sample_seconds = 0.0
        self.wall_seconds = 0.0

    def generate(self, invoke: Callable[[], H3VideoOutput]) -> H3VideoOutput | None:
        token = _ACTIVE.set(self)
        started = time.perf_counter()
        try:
            return invoke()
        except DenoisingComplete:
            return None
        finally:
            self.wall_seconds = time.perf_counter() - started
            _ACTIVE.reset(token)

    def sample(self, root: Any, invoke: Any, on_step: Any) -> Any:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("this experiment requires one physical CUDA GPU")
        state = getattr(root, "_compile_oracle", None)
        if state is not None:
            state.install(False)
        if self.payload.mode != "eager":
            if (
                state is None
                or state.key != (self.payload.mode, self.payload.autotune)
                or self.payload.cold
            ):
                state = CompilerState(root, self.payload)
                root._compile_oracle = state
            state.install(True)
        elif state is None:
            state = CompilerState(root, self.payload)
        self.compiler = state
        self.graphs_before = len(state.graphs)
        if self.payload.profile:
            self.profile = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
            )
            self.profile.start()
        torch.cuda.synchronize()
        started = last = time.perf_counter()

        def step(index: int) -> None:
            nonlocal last
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.steps.append(now - last)
            if index == 0 and self.profile is not None:
                self.profile.stop()
                self.profile_path = state.root / f"profile-{time.time_ns()}.json"
                self.profile.export_chrome_trace(str(self.profile_path))
                self.profile = None
            on_step(index)
            if self.stop_after_steps and len(self.steps) >= self.stop_after_steps:
                raise DenoisingComplete()
            last = time.perf_counter()

        try:
            with torch.profiler.record_function("h3_oracle_denoise"):
                return invoke(step)
        finally:
            torch.cuda.synchronize()
            self.sample_seconds = time.perf_counter() - started
            if self.profile is not None:
                self.profile.stop()
                self.profile = None


class OracleModel(H3Model, encoded_leaves="accept", fusion="accept"):
    @uses_components("fl2va_dit")
    def sample_fl2va(self, state: Any, *, on_step: Any, cancel: Any, checks: Any) -> Any:
        run = _ACTIVE.get()
        if run is None:
            raise RuntimeError("compiler oracle context is absent")
        root = self.pipe.components["fl2va_dit"]
        checks.component("fl2va_dit", root)
        with checks.forwards(root, "fl2va_dit"):
            return run.sample(
                root,
                lambda step: self.pipe.denoise(
                    "fl2va",
                    state,
                    on_step=step,
                    cancel=cancel,
                    checks=checks,
                ),
                on_step,
            )


def archive(run: RunState, report: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("measurements.json")
        info.size = len(report)
        tar.addfile(info, io.BytesIO(report))
        if run.compiler is not None:
            total = 0
            roots = {"diagnostic": run.compiler.root, **run.compiler.cache_roots}
            for label, root in roots.items():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.suffix in {".py", ".cpp", ".json"}:
                        total += path.stat().st_size
                        if total > 512 << 20:
                            raise RuntimeError(
                                "compiler source/trace archive exceeds diagnostic bound"
                            )
                        tar.add(path, arcname=str(Path(label) / path.relative_to(root)))
    result = buffer.getvalue()
    if len(result) > 128 << 20:
        raise RuntimeError("compressed compiler archive exceeds output bound")
    return result


def execute(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: OracleModel,
    out: Outputs,
    tel: Telemetry,
    *,
    probe_only: bool,
) -> tuple[H3VideoOutput | None, ProbeResult]:
    if not 5 <= payload.duration_s <= 15 or payload.seed < 0:
        raise ValueError("use duration_s 5..15 and a nonnegative seed")
    if not 0 <= payload.warmup_steps <= 2:
        raise ValueError("warmup_steps must be 0..2")

    def invoke() -> H3VideoOutput:
        return fl2va(
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

    # Warmup and measurement share this model/executor. A new seeded fl2va call
    # restarts the actual request; cold applies only before the warmup begins.
    state = RunState(
        msgspec.structs.replace(payload, cold=False) if payload.warmup_steps else payload,
        1 if probe_only else 0,
    )
    warmup = None
    warmup_report = None
    result = None
    status = "completed"
    error = None
    try:
        if payload.warmup_steps:
            warmup = RunState(
                msgspec.structs.replace(payload, profile=False), payload.warmup_steps
            )
            try:
                if warmup.generate(invoke) is not None:
                    raise RuntimeError("warmup reached decoding instead of stopping in denoising")
            finally:
                warmup_report = {
                    "requested_steps": payload.warmup_steps,
                    "steps": warmup.steps,
                    "wall_seconds": warmup.wall_seconds,
                    "denoise_seconds": warmup.sample_seconds,
                    "graphs_before": warmup.graphs_before,
                    "graphs_after": len(warmup.compiler.graphs) if warmup.compiler else 0,
                    "pid": os.getpid(),
                }
        result = state.generate(invoke)
        if result is None:
            status = "first_step_only"
    except Exception:
        status = "error"
        error = traceback.format_exc()
    compiler = state.compiler or (warmup.compiler if warmup else None)
    document = {
        "format": "h3.compile-oracle/1",
        "status": status,
        "error": error,
        "request_id": ctx.request_id,
        "pid": os.getpid(),
        "checkpoint": model.checkpoint_ref,
        "input": msgspec.to_builtins(payload),
        "warmup": warmup_report,
        "steps": state.steps,
        "generation_seconds": state.wall_seconds,
        "denoise_seconds": state.sample_seconds,
        "compiler_key": compiler.key if compiler else None,
        "compiler_cache": {key: str(value) for key, value in compiler.cache_roots.items()}
        if compiler
        else {},
        "cache_files_before": compiler.cache_files_before if compiler else {},
        "profile": str(state.profile_path) if state.profile_path else None,
        "graphs_before": state.graphs_before,
        "graphs_after": len(compiler.graphs) if compiler else 0,
        "graphs": compiler.graphs if compiler else [],
        "compile_times": compile_times(),
        "compiler_counters": {str(key): dict(value) for key, value in counters.items()},
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "diffusers",
                "cozy-runtime",
                "tensorfs",
                "triton",
            )
        },
        "gpu": torch.cuda.get_device_name(),
        "torch_settings": {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudagraphs": False,
        },
    }
    raw = json.dumps(document, default=str, allow_nan=False, indent=2).encode()
    tel.log(
        "compiler oracle",
        status=status,
        mode=payload.mode,
        generation_seconds=state.wall_seconds,
        denoise_seconds=state.sample_seconds,
        graphs=document["graphs_after"],
    )
    return result, ProbeResult(
        measurements=out.save_bytes(raw, media_type="application/json"),
        compiler_artifacts=out.save_bytes(
            archive(state if state.compiler else warmup or state, raw),
            media_type="application/gzip",
        ),
        status=status,
    )


@app.entrypoint
def run(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: OracleModel,
    out: Outputs,
    tel: Telemetry,
) -> GenerationResult:
    result, diagnostic = execute(ctx, payload, assets, model, out, tel, probe_only=False)
    if result is None:
        raise RuntimeError(
            "compiler benchmark failed; inspect the diagnostic probe before retrying"
        )
    return GenerationResult(
        video=result.video,
        warnings=result.warnings,
        measurements=diagnostic.measurements,
        compiler_artifacts=diagnostic.compiler_artifacts,
    )


@app.entrypoint
def probe(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: OracleModel,
    out: Outputs,
    tel: Telemetry,
) -> ProbeResult:
    _, diagnostic = execute(ctx, payload, assets, model, out, tel, probe_only=True)
    return diagnostic
