"""Observe one normal H3 turbo denoise step; never claims a generated video.

The leader's final projection outputs have already been gathered by Ulysses.
Intermediate prefix rows belong to rank zero at every supported GPU count.
Hooks copy bounded chunks for hashing and never replace tensors or modify RNG.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any

import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    Context,
    FileAsset,
    Image,
    InvalidRequest,
    Outputs,
    Telemetry,
    sequence_parallel,
)
from safetensors.torch import save

from h3 import FirstLastFrameToVideoTurboInput, H3TurboBase, H3TurboLoRA, fl2va_turbo

app = App()
_ACTIVE: ContextVar[FirstStepTrace | None] = ContextVar("h3_first_step", default=None)
_PREFIX_ROWS = 8
_HASH_CHUNK_BYTES = 1 << 20


class FirstStepCaptured(Exception):
    """The first scheduler update completed; the decoder was never reached."""


def tensor_bytes(value: torch.Tensor) -> Iterator[bytes]:
    if value.is_contiguous() or value.numel() * value.element_size() <= _HASH_CHUNK_BYTES:
        flat = value.contiguous().reshape(-1).view(torch.uint8)
        for chunk in flat.split(_HASH_CHUNK_BYTES):
            yield chunk.cpu().numpy().tobytes()
    else:
        for piece in value.unbind(0):
            yield from tensor_bytes(piece)


def tensor_record(value: torch.Tensor, *, prefix: bool = False) -> dict[str, Any]:
    """Hash original-dtype bytes; retain a small numeric sample for error measurement."""
    original_shape = list(value.shape)
    original_stride = list(value.stride())
    original_storage_offset = value.storage_offset()
    alignment = value.data_ptr() % 256
    if prefix:
        if value.ndim != 3 or value.shape[1] < _PREFIX_ROWS:
            raise ValueError("prefix observation requires at least eight packed rows")
        value = value[:, :_PREFIX_ROWS]
    value = value.detach()
    digest = hashlib.sha256()
    count = 0
    for raw in tensor_bytes(value):
        count += len(raw)
        digest.update(raw)
    indices = torch.linspace(
        0,
        max(0, value.numel() - 1),
        min(64, value.numel()),
        device=value.device,
        dtype=torch.long,
    )
    sample = value[torch.unravel_index(indices, value.shape)] if value.ndim else value.reshape(1)
    return {
        "shape": original_shape,
        "original_stride": original_stride,
        "original_storage_offset": original_storage_offset,
        "data_pointer_mod_256": alignment,
        "observed_shape": list(value.shape),
        "dtype": str(value.dtype),
        "coverage": "global packed rows 0..7 on rank zero" if prefix else "whole tensor",
        "sha256": digest.hexdigest(),
        "bytes": count,
        "sample_flat_indices": indices.cpu().tolist(),
        "sample_values": sample.float().cpu().tolist(),
    }


class FirstStepTrace:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.retained: dict[str, torch.Tensor] = {}
        self.completed_steps = 0
        self.forward = -1

    def record(self, name: str, value: Any, *, prefix: bool = False) -> None:
        if isinstance(value, torch.Tensor):
            if name in (
                "token_refiner.input",
                "token_refiner.output",
                "token_refiner.refiner_blocks.0.attn.input",
                "token_refiner.refiner_blocks.0.attn.output",
                "token_refiner.refiner_blocks.0.attn.to_q.output",
                "token_refiner.refiner_blocks.0.attn.to_k.output",
                "token_refiner.refiner_blocks.0.attn.to_v.output",
                "token_refiner.refiner_blocks.0.attn.to_out.0.input",
            ):
                if value.numel() * value.element_size() > 2 << 20:
                    raise ValueError("refiner observation exceeds two MiB per tensor")
                self.retained[name] = value.detach().to(device="cpu", copy=True).contiguous()
            self.records.append(
                {
                    "name": name,
                    "forward": self.forward,
                    **tensor_record(value, prefix=prefix),
                }
            )
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                self.record(f"{name}.{index}", child, prefix=prefix)

    @contextmanager
    def observe(self, root: Any) -> Iterator[None]:
        signature = inspect.signature(root.forward)
        handles: list[Any] = []

        def before(_module: Any, args: Any, kwargs: Any) -> None:
            self.forward += 1
            if self.forward >= 4:
                raise ValueError("more than four DiT evaluations before the first callback")
            for name, value in signature.bind(*args, **kwargs).arguments.items():
                self.record(f"dit.input.{name}", value)

        def input_hook(name: str, prefix: bool) -> Any:
            def observed(_module: Any, args: Any, kwargs: Any) -> None:
                value = args[0] if args else kwargs["hidden_states"]
                self.record(name + ".input", value, prefix=prefix)

            return observed

        def output_hook(name: str, prefix: bool) -> Any:
            def observed(_module: Any, _args: Any, value: Any) -> None:
                self.record(name + ".output", value, prefix=prefix)

            return observed

        try:
            handles.append(root.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(root.register_forward_hook(output_hook("dit", False)))
            for name, prefix_input, prefix_output in (
                ("proj_in", False, False),
                ("audio_proj_in", False, False),
                ("context_embedder", False, False),
                ("token_refiner", False, False),
                ("token_refiner.refiner_blocks.0", False, False),
                ("token_refiner.refiner_blocks.0.attn", False, False),
                ("token_refiner.refiner_blocks.0.attn.to_q", False, False),
                ("token_refiner.refiner_blocks.0.attn.to_k", False, False),
                ("token_refiner.refiner_blocks.0.attn.to_v", False, False),
                ("token_refiner.refiner_blocks.0.attn.to_out.0", False, False),
                ("token_refiner.refiner_blocks.0.ff", False, False),
                ("transformer_blocks.0", False, True),
                ("norm_out", True, True),
                ("proj_out", True, False),
                ("audio_proj_out", True, False),
            ):
                module = root.get_submodule(name)
                handles.append(
                    module.register_forward_pre_hook(
                        input_hook(name, prefix_input),
                        with_kwargs=True,
                    )
                )
                handles.append(module.register_forward_hook(output_hook(name, prefix_output)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def step_callback(self, original: Any) -> Any:
        def finished(index: int) -> None:
            original(index)
            if index != 0:
                raise ValueError("first-step probe received a nonzero first callback")
            self.completed_steps = 1
            raise FirstStepCaptured

        return finished


@sequence_parallel(degrees=(2, 4))
class TraceLoRA(H3TurboLoRA):
    def _sample(
        self,
        base: Any,
        trunk: Any,
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        checks: Any,
    ) -> Any:
        trace = _ACTIVE.get()
        if trace is None:
            raise RuntimeError("first-step capture is not active")
        with trace.observe(base.pipe.components[f"{trunk}_dit"]):
            return super()._sample(
                base,
                trunk,
                state,
                on_step=trace.step_callback(on_step),
                cancel=cancel,
                checks=checks,
            )


class ProbeInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)] = 5


class ProbeOutput(msgspec.Struct):
    refiner_tensors: Annotated[
        FileAsset, AssetBound(max_bytes=9 << 20, media_types=("application/octet-stream",))
    ]
    trace: Annotated[FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))]
    completed_steps: int


@app.entrypoint
def probe(
    ctx: Context,
    payload: ProbeInput,
    assets: Annotated[Assets[Image], AssetLimits(images=2)],
    base_model: H3TurboBase,
    turbo_lora: TraceLoRA,
    out: Outputs,
    tel: Telemetry,
) -> ProbeOutput:
    if payload.seed < 0:
        raise InvalidRequest("seed must be nonnegative", fields=["seed"])
    trace = FirstStepTrace()
    token = _ACTIVE.set(trace)
    try:
        fl2va_turbo(
            ctx,
            FirstLastFrameToVideoTurboInput(
                prompt=payload.prompt,
                seed=payload.seed,
                duration_s=payload.duration_s,
            ),
            assets,
            base_model,
            turbo_lora,
            out,
            tel,
        )
    except FirstStepCaptured:
        pass
    else:
        raise RuntimeError("probe reached video completion instead of the first callback")
    finally:
        _ACTIVE.reset(token)
    document = {
        "format": "h3.first-step-boundaries/1",
        "request_id": ctx.request_id,
        "completed_steps": trace.completed_steps,
        "generated_video": False,
        "base_checkpoint": base_model.checkpoint_ref,
        "adapter_checkpoint": turbo_lora.checkpoint_ref,
        "prompt": payload.prompt,
        "seed": payload.seed,
        "duration_s": payload.duration_s,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("h3-first-step", "torch", "diffusers", "cozy-runtime", "tensorfs")
        },
        "retained_tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in trace.retained.items()
        },
        "torch_settings": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "matmul_allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "distributed_initialized": torch.distributed.is_initialized(),
            "world_size": torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1,
        },
        "coverage": (
            "Whole DiT inputs, block-zero inputs, gathered projection and DiT outputs; "
            "rank-zero eight-row prefixes at block-zero output and projection inputs. "
            "Middle transformer blocks are not traced; first differing boundary is observed, "
            "not necessarily the first arithmetic operation that differs."
        ),
        "records": trace.records,
    }
    raw = json.dumps(document, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > 1 << 20:
        raise ValueError("first-step trace exceeds one MiB")
    return ProbeOutput(
        refiner_tensors=out.save_bytes(save(trace.retained), media_type="application/octet-stream"),
        trace=out.save_bytes(raw, media_type="application/json"),
        completed_steps=1,
    )
