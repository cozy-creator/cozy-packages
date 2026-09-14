"""Check an assigned GPU and real attention kernels without any checkpoint download."""

from __future__ import annotations

import importlib.metadata
import time
import uuid

import diffusers
import msgspec
import torch
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.author import App, Bound, Context
from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts

app = App()


class Input(msgspec.Struct, forbid_unknown_fields=True):
    expected_runtime_source: str
    expected_gpu_uuid: str
    expected_sm: int
    expected_visible_devices: int = 1
    backends: tuple[str, ...] = ("flash-attn3", "sageattention", "kitchen-int8", "sol-attn")


class Kernel(msgspec.Struct):
    name: str
    backend: str
    location: str
    preparation_seconds: float


class Result(msgspec.Struct):
    status: str
    runtime_source: str
    gpu_uuid: str
    gpu_name: str
    sm: int
    visible_devices: int
    total_memory_bytes: int
    torch_version: str
    cuda_version: str
    diffusers_version: str
    diffusers_path: str
    kernels: list[Kernel]


@app.entrypoint(
    demand=Bound(
        reason="Bound small synthetic attention scratch and imports; no model weights are loaded.",
        vram_bytes=1 << 30,
        host_ram_bytes=4 << 30,
    )
)
def probe(ctx: Context, payload: Input) -> Result:
    ctx.raise_if_cancelled()
    if payload.expected_runtime_source != COMMIT:
        raise ValueError(
            f"Runtime source differs: expected {payload.expected_runtime_source}, got {COMMIT}"
        )
    if ctx.device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "This smoke request requires an assigned CUDA device; CPU is not a fallback"
        )
    if not payload.expected_gpu_uuid:
        raise ValueError("Use the UUID from this owned rental's verified bootstrap receipt")
    count = torch.cuda.device_count()
    if count != payload.expected_visible_devices:
        raise ValueError(
            f"Expected {payload.expected_visible_devices} visible devices, observed {count}"
        )
    index = ctx.device.index if ctx.device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    gpu_uuid = str(getattr(properties, "uuid", ""))
    sm = properties.major * 10 + properties.minor
    observed_uuid = uuid.UUID(gpu_uuid.removeprefix("GPU-"))
    expected_uuid = uuid.UUID(payload.expected_gpu_uuid.removeprefix("GPU-"))
    if observed_uuid != expected_uuid or sm != payload.expected_sm:
        raise ValueError(f"Assigned GPU identity differs: {gpu_uuid}, SM{sm}")
    if importlib.metadata.version("diffusers") != "0.40.0" or diffusers.__version__ != "0.40.0":
        raise ValueError(f"Diffusers metadata/import mismatch at {diffusers.__file__}")
    if not payload.backends or len(payload.backends) != len(set(payload.backends)):
        raise ValueError("Request a nonempty set of distinct explicit backends")

    device = DeviceFacts(
        kind="cuda", name=properties.name, sm=sm, driver="", configuration="", index=index
    )
    kernels: list[Kernel] = []
    for name in payload.backends:
        ctx.raise_if_cancelled()
        started = time.perf_counter()
        # Runtime resolves this exact backend and executes its existing real-GPU
        # numerical check, including explicit synthetic layout for Sol. No fallback.
        ready = attention.pinned(name, device)
        torch.cuda.synchronize(index)
        if len(ready) != 1 or ready[0].name != name:
            raise ValueError("Runtime did not resolve the exact requested backend")
        entry = ready[0]
        kernels.append(
            Kernel(name, str(entry.member.value), entry.location, time.perf_counter() - started)
        )
    return Result(
        "kernel_smoke_passed_not_video_quality",
        COMMIT,
        gpu_uuid,
        properties.name,
        sm,
        count,
        properties.total_memory,
        str(torch.__version__),
        str(torch.version.cuda),
        diffusers.__version__,
        str(diffusers.__file__),
        kernels,
    )
