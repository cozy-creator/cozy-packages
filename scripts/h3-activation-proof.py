"""CPU proof that diagnostic hooks observe without changing H3 calls or RNG.

Run with the diagnostic project's locked interpreter; no weights or GPU needed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics/h3-vae-roundtrip"))
from activation_trace import ACTIVE_TRACE, ActivationTrace, FirstStepCaptured  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.fail = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("injected forward failure")
        return value * (self.index + 1) + 0.125


class TinyDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block(i) for i in range(5)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, timestep_indices, token_tags, position_ids
        del video_indices, audio_indices, text_indices
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def main() -> None:
    torch.set_num_threads(1)
    model = TinyDiT()
    value = torch.arange(2 * 33 * 35, dtype=torch.float32).reshape(2, 33, 35) / 4096
    arguments: dict[str, Any] = {
        "hidden_states": value,
        "timestep": torch.tensor([0.0, 0.25]),
        "timestep_indices": torch.arange(33) % 2,
        "token_tags": torch.tensor([0] * 17 + [1] * 11 + [2] * 5),
        "position_ids": torch.arange(99).reshape(33, 3),
        "video_indices": torch.arange(17),
        "text_indices": torch.arange(17, 28),
        "audio_indices": torch.arange(28, 33),
    }
    original_rng = torch.get_rng_state().clone()
    expected = model(**arguments)
    outputs = []
    documents = []
    for _ in range(2):
        trace = ActivationTrace(evaluations=3, first_step=False)
        calls: list[int] = []
        callback = trace.step_callback(calls.append)
        with trace.active(), trace.observe(model):
            for index in range(3):
                outputs.append(model(**arguments))
                callback(index)
        state = SimpleNamespace(latents=value.to(torch.bfloat16), audio_latents=value[:, :2])
        trace.final_latents(state)
        assert trace.latents["video_latents"].dtype == torch.bfloat16
        assert trace.latents["video_latents"].data_ptr() != state.latents.data_ptr()
        assert torch.equal(trace.latents["video_latents"], state.latents)
        assert calls == [0, 1, 2]
        assert len(trace.records) == 9
        assert trace.block_indices == [0, 2, 4]
        first = trace.records[0]
        # Coordinates are independently asserted, including the short audio modality.
        assert first["packed_rows"] == [
            0,
            2,
            4,
            6,
            9,
            11,
            13,
            16,
            17,
            18,
            19,
            21,
            22,
            24,
            25,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
        assert first["channel_indices"] == [
            0,
            2,
            4,
            6,
            9,
            11,
            13,
            15,
            18,
            20,
            22,
            24,
            27,
            29,
            31,
            34,
        ]
        for batch in range(2):
            for row_offset, row in enumerate(first["packed_rows"]):
                for channel_offset, channel in enumerate(first["channel_indices"]):
                    assert first["values"][batch][row_offset][channel_offset] == float(
                        value[batch, row, channel] + 0.125
                    )
        documents.append(trace.document())
    assert documents[0] == documents[1]
    assert all(torch.equal(output, expected) for output in outputs)
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert ACTIVE_TRACE.get() is None

    probe = ActivationTrace(evaluations=29, first_step=True)
    calls = []
    try:
        with probe.active(), probe.observe(model):
            model(**arguments)
            probe.step_callback(calls.append)(0)
    except FirstStepCaptured:
        pass
    else:
        raise AssertionError("first step did not stop")
    assert calls == [0] and probe.completed_steps == 1 and len(probe.records) == 3
    assert not probe.document()["final_latents_present"]
    try:
        probe.final_latents(state)
    except ValueError:
        pass
    else:
        raise AssertionError("a partial run claimed final latents")

    failing = model.transformer_blocks[2]
    assert isinstance(failing, Block)
    failing.fail = True
    try:
        with probe.active(), probe.observe(model):
            model(**arguments)
    except RuntimeError as exc:
        assert str(exc) == "injected forward failure"
    else:
        raise AssertionError("forward failure was swallowed")
    assert ACTIVE_TRACE.get() is None
    assert not model._forward_pre_hooks
    assert all(not block._forward_hooks for block in model.transformer_blocks)
    assert torch.equal(torch.get_rng_state(), original_rng)
    failing.fail = False

    # Exercise the diagnostic override and unchanged H3 parent method.
    import vae_diagnostic
    from safetensors.torch import load, save

    import h3
    from official import OfficialH3Pipeline

    assert vae_diagnostic.ref2va is h3.ref2va
    assert vae_diagnostic.TraceModel.sample_ref2va is h3.H3Model.sample_ref2va
    seen: list[int] = []

    def denoise(
        self: Any, task: str, state: Any, *, on_step: Any, cancel: Any, checks: Any = None
    ) -> Any:
        assert task == "ref2va"
        del checks
        for index in range(3):
            cancel()
            state.latents = model(**arguments)
            state.audio_latents = state.latents[:, :2]
            seen.append(index)
            on_step(index)
        return "unchanged parent result"

    @contextmanager
    def synthetic_steps() -> Iterator[None]:
        original = OfficialH3Pipeline.denoise
        OfficialH3Pipeline.denoise = denoise  # type: ignore[method-assign]
        try:
            yield
        finally:
            OfficialH3Pipeline.denoise = original  # type: ignore[method-assign]

    pipe = object.__new__(vae_diagnostic.TracePipeline)
    pipe.components = {"ref2va_dit": model}
    observed_model: Any = vae_diagnostic.TraceModel.for_test(pipe=pipe)
    checks = SimpleNamespace(component=lambda *_: None, forwards=lambda *_: nullcontext())
    full = ActivationTrace(evaluations=3, first_step=False)
    with synthetic_steps(), full.active():
        result = observed_model.sample_ref2va(
            SimpleNamespace(), on_step=lambda _: None, cancel=lambda: None, checks=checks
        )
    assert result == "unchanged parent result" and seen == [0, 1, 2]
    restored = load(save(full.latents))
    assert all(torch.equal(restored[key], value) for key, value in full.latents.items())
    seen.clear()
    partial = ActivationTrace(evaluations=3, first_step=True)
    try:
        with synthetic_steps(), partial.active():
            observed_model.sample_ref2va(
                SimpleNamespace(), on_step=lambda _: None, cancel=lambda: None, checks=checks
            )
    except FirstStepCaptured:
        pass
    else:
        raise AssertionError("component wrapper swallowed the explicit probe stop")
    assert seen == [0] and not partial.latents and ACTIVE_TRACE.get() is None
    assert not model._forward_pre_hooks
    assert all(not block._forward_hooks for block in model.transformer_blocks)
    print(
        json.dumps(
            {
                "output_bitwise_unchanged": True,
                "rng_unchanged": True,
                "repeat_coordinates_and_values_identical": True,
                "first_step_only": True,
                "partial_not_final": True,
                "failure_cleanup": True,
                "full_sample_records": 9,
                "probe_sample_records": 3,
                "actual_h3_parent_delegation": True,
                "component_scope_preserves_sentinel": True,
                "safetensors_roundtrip": True,
            }
        )
    )


if __name__ == "__main__":
    main()
