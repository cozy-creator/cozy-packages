#!/usr/bin/env python3
"""Compare staged Anima with the original Diffusers workflow on a tiny CPU model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "anima"))

import anima  # noqa: E402


def main() -> None:
    spec = importlib.util.spec_from_file_location("warm_fixture", ROOT / "scripts/warm-conform.py")
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    torch.set_num_threads(1)
    pipe = anima.build_pipeline(fixture.anima_config())
    fixture.initialize_fixture(pipe)
    model = anima.AnimaModel.for_test(pipe=pipe)
    previous: dict[int, torch.Tensor] = {}

    for seed in (123, 456, 123):
        baseline = anima._text2image_pipeline(torch.device("cpu"), anima._Phases())
        baseline.register_components(
            **pipe.components,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler_config),
            tokenizer=pipe.tokenizer,
            t5_tokenizer=pipe.t5_tokenizer,
        )
        baseline.guider.guidance_scale = 4.5
        baseline.guider._start, baseline.guider._stop = 0.0, 1.0
        expected = baseline(
            prompt="a red fox",
            negative_prompt="blurry",
            width=64,
            height=64,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(seed),
            output="images",
            output_type="pt",
        )
        count = len(model.harness.calls)
        actual = anima._render(
            model, "a red fox", "blurry", 64, 64, 2, 4.5, (0.0, 1.0), 0.0, seed,
            anima._Phases(),
        )
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected), (actual - expected).abs().max().item()
        calls = model.harness.calls[count:]
        assert [(call.method, call.components) for call in calls] == [
            ("encode", ("text_encoder",)),
            ("denoise", ("text_conditioner", "transformer")),
            ("decode", ("vae",)),
        ]
        if seed in previous:
            assert torch.equal(actual, previous[seed]), "resident state leaked between requests"
        previous[seed] = actual
        print(f"PASS seed {seed}: exact Diffusers pixels and disjoint component leases")


if __name__ == "__main__":
    main()
