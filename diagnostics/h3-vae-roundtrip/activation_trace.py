"""Bounded observations of the unchanged H3 DiT; no tensor or RNG is modified."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

ROWS_PER_MODALITY = 8
CHANNELS = 16
MAX_BATCHES = 2
MAX_CALLS_PER_STEP = 4
MAX_LATENT_BYTES = 96 << 20
TRACE_FORMAT = "h3.activation-samples/1"


class FirstStepCaptured(Exception):
    """The explicit first-step diagnostic completed; no video or final latents exist."""


ACTIVE_TRACE: ContextVar[ActivationTrace | None] = ContextVar("h3_activation_trace", default=None)


def evenly_spaced(length: int, maximum: int) -> list[int]:
    count = min(length, maximum)
    if count < 2:
        return list(range(count))
    return [index * (length - 1) // (count - 1) for index in range(count)]


class ActivationTrace:
    def __init__(self, *, evaluations: int, first_step: bool) -> None:
        if not 1 <= evaluations <= 128:
            raise ValueError("unsupported diagnostic evaluation count")
        self.evaluations = evaluations
        self.first_step = first_step
        self.selected_steps = [0] if first_step else sorted({0, evaluations // 2, evaluations - 1})
        self.step = 0
        self.completed_steps = 0
        self.calls = 0
        self.block_indices: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.forward: dict[str, Any] | None = None
        self.latents: dict[str, Any] = {}

    @contextmanager
    def active(self) -> Iterator[None]:
        token = ACTIVE_TRACE.set(self)
        try:
            yield
        finally:
            ACTIVE_TRACE.reset(token)

    def step_callback(self, original: Callable[[int], None]) -> Callable[[int], None]:
        def finished(index: int) -> None:
            original(index)
            if index != self.step:
                raise ValueError("nonsequential denoise callback")
            self.completed_steps = index + 1
            self.step = index + 1
            self.calls = 0
            if self.first_step:
                raise FirstStepCaptured

        return finished

    @contextmanager
    def observe(self, module: Any) -> Iterator[None]:
        blocks = module.transformer_blocks
        if not 1 <= len(blocks) <= 128:
            raise ValueError("unsupported diagnostic block count")
        self.block_indices = sorted({0, (len(blocks) - 1) // 2, len(blocks) - 1})
        signature = inspect.signature(module.forward)
        handles: list[Any] = []

        def before(_module: Any, args: Any, kwargs: Any) -> None:
            if ACTIVE_TRACE.get() is not self:
                return
            self.forward = None
            ordinal = self.calls
            self.calls += 1
            if self.step not in self.selected_steps:
                return
            if ordinal >= MAX_CALLS_PER_STEP:
                raise ValueError("too many DiT forwards in one captured step")
            values = signature.bind(*args, **kwargs).arguments
            positions = values["position_ids"]
            rows: list[int] = []
            modalities: list[str] = []
            for name in ("video", "text", "audio"):
                source = values[name + "_indices"]
                offsets = torch.tensor(
                    evenly_spaced(source.numel(), ROWS_PER_MODALITY),
                    device=source.device,
                    dtype=torch.long,
                )
                selected = source.index_select(0, offsets).detach().cpu().tolist()
                rows.extend(selected)
                modalities.extend([name] * len(selected))
            indices = torch.tensor(rows, device=positions.device, dtype=torch.long)
            timestep = values["timestep"]
            if timestep.numel() > 8:
                raise ValueError("too many distinct H3 timesteps")
            self.forward = {
                "step": self.step,
                "forward_in_step": ordinal,
                "packed_rows": rows,
                "modalities": modalities,
                "position_ids": positions.index_select(0, indices).detach().cpu().tolist(),
                "timestep_indices": values["timestep_indices"]
                .index_select(0, indices)
                .detach()
                .cpu()
                .tolist(),
                "token_tags": values["token_tags"].index_select(0, indices).detach().cpu().tolist(),
                "timesteps": timestep.detach().float().cpu().reshape(-1).tolist(),
                "sequence_length": int(positions.shape[0]),
            }

        def block_hook(index: int) -> Callable[..., None]:
            def after(_module: Any, _args: Any, output: Any) -> None:
                if ACTIVE_TRACE.get() is not self or self.forward is None:
                    return
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise ValueError("H3 block output must be batch/packed-row/channel")
                if (
                    output.shape[0] > MAX_BATCHES
                    or output.shape[1] != self.forward["sequence_length"]
                ):
                    raise ValueError("unexpected H3 activation geometry")
                channels = evenly_spaced(output.shape[2], CHANNELS)
                rows = torch.tensor(
                    self.forward["packed_rows"], device=output.device, dtype=torch.long
                )
                columns = torch.tensor(channels, device=output.device, dtype=torch.long)
                # Gather on the device before the small CPU copy; never clone the sequence.
                sample = (
                    output.detach().index_select(1, rows).index_select(2, columns).float().cpu()
                )
                self.records.append(
                    {
                        **self.forward,
                        "module": f"transformer_blocks.{index}",
                        "output_shape": list(output.shape),
                        "output_dtype": str(output.dtype),
                        "device": str(output.device),
                        "batch_indices": list(range(output.shape[0])),
                        "channel_indices": channels,
                        "values": sample.tolist(),
                    }
                )

            return after

        try:
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            for index in self.block_indices:
                handles.append(blocks[index].register_forward_hook(block_hook(index)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.forward = None

    def final_latents(self, state: Any) -> None:
        tensors = {"video_latents": state.latents, "audio_latents": state.audio_latents}
        if self.first_step or self.completed_steps != self.evaluations:
            raise ValueError("final latents require every denoise step to complete")
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise ValueError("final H3 latents are absent")
        if (
            sum(value.numel() * value.element_size() for value in tensors.values())
            > MAX_LATENT_BYTES
        ):
            raise ValueError("final latents exceed the 96 MiB diagnostic limit")
        self.latents = {
            name: value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
            for name, value in tensors.items()
        }

    def document(self) -> dict[str, Any]:
        return {
            "format": TRACE_FORMAT,
            "mode": "first_step" if self.first_step else "complete_video",
            "component": "ref2va_dit",
            "selected_steps": self.selected_steps,
            "selected_blocks": self.block_indices,
            "completed_steps": self.completed_steps,
            "final_latents_present": bool(self.latents),
            "coordinates": "zero-based packed rows by modality; values[batch][row][channel]",
            "coverage": "deterministic samples, not whole-activation fidelity",
            "samples": self.records,
            "final_latents": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.latents.items()
            },
        }
