"""Verify diagnostic neutrality on a tiny real H3 forward, not an inference qualification."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3"))
sys.path.insert(0, str(ROOT / "examples/client-scripts/h3-first-step"))

from h3_first_step import FirstStepCaptured, FirstStepTrace, tensor_record  # noqa: E402

from cozy_runtime.models.minimax_h3.adaln_pruned import AdaLNPrunedMiniMaxH3Transformer  # noqa: E402
from cozy_runtime.models.minimax_h3.official import canonical_timestep_plan  # noqa: E402
from cozy_runtime.models.minimax_h3.turbo import (  # noqa: E402
    ATTENTION_KWARG,
    OVERLAY_KWARG,
    TURBO_BANK,
    TurboOverlay,
    TurboSchedule,
)


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(712)
    config = {
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "hidden_size": 8,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 16,
        "in_channels": 2,
        "audio_in_channels": 2,
        "patch_size": (1, 1, 1),
        "text_dim": 8,
        "freq_dim": 4,
        "time_embed_hidden_dim": 8,
        "time_embed_dim": 4,
        "rope_freq_dim": 1,
    }
    times, keys = canonical_timestep_plan("fl2va").table_layout()
    dit = AdaLNPrunedMiniMaxH3Transformer.from_official_config(
        config,
        table_timesteps=times,
        table_block_keys=keys,
    ).eval()
    plan = canonical_timestep_plan("fl2va_turbo")
    times, keys = plan.table_layout()
    schedule = TurboSchedule(plan.schedules[0].video_timesteps, plan.schedules[0].audio_timesteps)
    overlay = TurboOverlay.from_official_config(
        config,
        rank=4,
        alpha=4,
        schedule=schedule,
        table_timesteps=times,
        table_block_keys=keys,
        block_table_dtype=torch.float32,
        final_table_dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in (*dit.parameters(), *overlay.parameters()):
            parameter.normal_(0, 0.02)
    dit.install_lora_consumers()
    dit.set_attention_backend("native")
    distinct = sorted({schedule.video[0], schedule.audio[0]})
    inputs = {
        "hidden_states": torch.randn(1, 8, 2),
        "audio_hidden_states": torch.randn(1, 4, 2),
        "encoder_hidden_states": torch.randn(1, 4, 8),
        "timestep": torch.tensor(distinct),
        "timestep_indices": torch.tensor(
            [distinct.index(schedule.video[0])] * 12 + [distinct.index(schedule.audio[0])] * 4,
        ),
        "token_tags": torch.tensor([1] * 4 + [0] * 8 + [2] * 4),
        "position_ids": torch.zeros(16, 3),
        "text_indices": torch.arange(4),
        "video_indices": torch.arange(4, 12),
        "audio_indices": torch.arange(12, 16),
        "return_dict": False,
        "attention_kwargs": {ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: overlay},
    }
    with torch.no_grad():
        expected = dit(**inputs)
        random_state = torch.get_rng_state()
        trace = FirstStepTrace()
        original_hooks = {
            name: (len(module._forward_hooks), len(module._forward_pre_hooks))
            for name, module in dit.named_modules()
        }
        with trace.observe(dit):
            actual = dit(**inputs)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
        assert torch.equal(random_state, torch.get_rng_state())
        assert original_hooks == {
            name: (len(module._forward_hooks), len(module._forward_pre_hooks))
            for name, module in dit.named_modules()
        }
        # A real changed projection must first appear at the observed head output,
        # while noise, conditioning, trunk and head inputs remain exactly equal.
        overlay.proj_out.bias.add_(0.125)
        changed = FirstStepTrace()
        with changed.observe(dit):
            dit(**inputs)
        differences = [
            a["name"]
            for a, b in zip(trace.records, changed.records, strict=True)
            if a["sha256"] != b["sha256"]
        ]
        assert differences == ["proj_out.output", "dit.output.0"], differences
    names = {record["name"] for record in trace.records}
    assert {
        "dit.input.hidden_states",
        "transformer_blocks.0.input",
        "norm_out.input",
        "proj_out.output",
        "audio_proj_out.output",
        "dit.output.0",
    } <= names
    for value in (
        torch.tensor(2.0),
        torch.arange(48).reshape(2, 3, 8).transpose(1, 2),
        torch.arange(400000).reshape(400, 1000).T,
    ):
        expected_hash = hashlib.sha256(
            value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),
        ).hexdigest()
        assert tensor_record(value)["sha256"] == expected_hash
    callbacks: list[int] = []
    try:
        trace.step_callback(callbacks.append)(0)
    except FirstStepCaptured:
        pass
    else:
        raise AssertionError("did not stop at first callback")
    assert callbacks == [0] and trace.completed_steps == 1
    print(
        "PASS: real tiny H3 output/RNG neutral, hooks restored, exact original-dtype hashes, "
        "changed head localized to projection output, first callback stops before decoding"
    )


if __name__ == "__main__":
    main()
