"""One complete H3 Turbo denoising-forward profile through ordinary Creator serving."""

from __future__ import annotations

import importlib.metadata
import json
import tarfile
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import msgspec
import torch.distributed as dist
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.author import (
    App,
    AssetBound,
    AttentionContext,
    Context,
    FileAsset,
    Loader,
    Outputs,
    Telemetry,
    sequence_parallel,
)
from cozy_runtime.models.minimax_h3 import H3TurboBase, H3TurboLoRA
from cozy_runtime.models.minimax_h3.official import NumericalChecks, frames_for
from profile_trace import MARKER, install_profile_hooks

from h3 import H3Model as WorkflowModel

app = App()
ATTENTION_BACKEND: Literal["sol-attn", "flash-attn3"] = "flash-attn3"
# The capture driver substitutes another explicitly reviewed source together with
# its exact local wheel. The request cannot relax or change this source guard.
RUNTIME_SOURCE = "d140609bf5fd07f8c67eaf7e87de433f5177f94f"
MAX_ARCHIVE_BYTES = 256 << 20


@sequence_parallel(degrees=(2, 4))
class ProfiledTurbo(H3TurboBase, encoded_leaves="accept", fusion="accept"):
    def choose_attention(self, context: AttentionContext) -> str | None:
        return ATTENTION_BACKEND if context.component.rsplit("/", 1)[-1] == "fl2va_dit" else None

    def load(self, loader: Loader) -> None:
        if COMMIT != RUNTIME_SOURCE:
            raise RuntimeError(f"profile Runtime {COMMIT} differs from reviewed {RUNTIME_SOURCE}")
        super().load(loader)
        install_profile_hooks(self.pipe.components["fl2va_dit"])


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int = 7101
    duration_s: Literal[15] = 15
    capture_step: Annotated[int, msgspec.Meta(ge=0, le=7)] = 1
    sol_dense_steps: Literal[3, 4, 8] = 8
    expected_degree: Literal[1, 2, 4] = 4


class Result(msgspec.Struct):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/json",))
    ]
    traces: Annotated[
        FileAsset, AssetBound(max_bytes=MAX_ARCHIVE_BYTES, media_types=("application/gzip",))
    ]
    status: str


class ProfileComplete(Exception):
    """The leader stops after every rank has completed the selected forward."""


@app.entrypoint
def main(
    ctx: Context,
    payload: Input,
    base_model: ProfiledTurbo,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    if not isinstance(base_model, WorkflowModel):
        raise RuntimeError("profile requires the migrated H3 workflow and Runtime models")
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world != payload.expected_degree:
        raise RuntimeError(f"expected {payload.expected_degree} GPUs, received {world}")
    config = base_model.pipe.components["fl2va_dit"].config
    if config.num_attention_heads != 56 or config.attention_head_dim != 128:
        raise RuntimeError("this profile requires the real H3 56-head, width-128 DiT")
    view = base_model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, base_model.pipe.resident)
    with tel.stage("prepare"):
        state = base_model.pipe.start_fl2va(
            prompt=payload.prompt,
            first_frame=None,
            last_frame=None,
            generator=base_model.pipe.generator(view.generator),
            steps=8,
            frames=frames_for(payload.duration_s),
            task="fl2va_turbo",
        )
    with tel.stage("condition_text"):
        base_model.condition_text("fl2va_turbo", state, checks=checks)
    prefix = out.temporary_file(".h3-scaling")
    state.set(
        "attention_kwargs",
        {
            **(state.get("attention_kwargs") or {}),
            MARKER: {"prefix": str(prefix), "step": payload.capture_step},
        },
    )
    elapsed: list[dict[str, Any]] = []
    started = time.perf_counter()
    boundary = started

    def on_step(index: int) -> None:
        nonlocal boundary
        now = time.perf_counter()
        elapsed.append({"step": index, "leader_step_wall_ms": (now - boundary) * 1000})
        boundary = now
        tel.progress((index + 1) / (payload.capture_step + 1), stage="profile")
        if index == payload.capture_step:
            # This callback follows the completed component call and follower ACKs.
            # The sentinel never interrupts an in-flight collective or follower.
            raise ProfileComplete

    with tel.stage("profile"):
        try:
            base_model.sample_fl2va_turbo(
                state,
                turbo_lora=turbo_lora,
                sol_dense_steps=payload.sol_dense_steps,
                on_step=on_step,
                cancel=ctx.raise_if_cancelled,
                checks=checks,
            )
        except ProfileComplete:
            pass
        else:
            raise RuntimeError("selected full forward did not produce a profile")
    total_wall_ms = (time.perf_counter() - started) * 1000
    ranks = []
    trace_paths = []
    for rank in range(world):
        metadata = Path(f"{prefix}.rank-{rank}.json")
        report = json.loads(metadata.read_text())
        if (report["rank"], report["world"], report["step"]) != (rank, world, payload.capture_step):
            raise RuntimeError("rank profile identity differs from the completed step")
        if not report["forward_succeeded"]:
            raise RuntimeError("a rank did not complete the profiled forward")
        ranks.append(report)
        if report["trace_within_bound"]:
            trace_paths.append(Path(f"{prefix}.rank-{rank}.trace.json"))
    document = {
        "request_id": ctx.request_id,
        "runtime_source": COMMIT,
        "backend": ATTENTION_BACKEND,
        "payload": msgspec.to_builtins(payload),
        "world": world,
        "ranks": ranks,
        "steps": elapsed,
        "leader_denoising_wall_ms": total_wall_ms,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "tensorfs", "cozy-runtime")
        },
        "notes": [
            "One full forward includes 50 main blocks, refiner, PDD LoRA and final projection.",
            "Profiler stop may synchronize once. Wrappers add no per-operation CUDA sync.",
            "Leader wall time includes argument transport and trace export; it is diagnostic.",
            "Rank wall time excludes export; CPU waits overlap GPU kernels. Inspect the trace.",
            "Size-gather backend distinguishes CPU/Gloo from accelerator/NCCL; inspect it.",
        ],
    }
    archive_path = out.temporary_file(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in trace_paths:
            archive.add(path, arcname=path.name, recursive=False)
    document["trace_archive_bytes"] = archive_path.stat().st_size
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        document["trace_archive_omitted"] = "compressed trace exceeds declared asset bound"
        archive_path = out.temporary_file(".summary-only.tar.gz")
        with tarfile.open(archive_path, "w:gz"):
            pass
        trace_paths.clear()
    return Result(
        measurements=out.save_bytes(
            json.dumps(document, indent=2, allow_nan=False).encode(), media_type="application/json"
        ),
        traces=out.save_file(archive_path, media_type="application/gzip"),
        status="complete" if len(trace_paths) == world else "trace_size_limited",
    )
