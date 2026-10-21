#!/usr/bin/env python
"""CPU capture of actual SDXL/Qwen Context.plan calls; no weights or model work."""
from __future__ import annotations

import sys
from collections.abc import Callable, Collection, Sequence
from functools import partial
from pathlib import Path

from cozy_runtime.author import Context, ImageAsset
from cozy_runtime.author.fakes import fake_adjustments, fake_attempt, fake_outputs, fake_telemetry

ROOT = Path(__file__).resolve().parent.parent


class Planned(Exception):
    pass


def capture(render: Callable[[Context], object], expected: tuple[str | frozenset[str], ...]) -> None:
    captured: list[tuple[str | frozenset[str], ...]] = []

    def planned(steps: Sequence[str | Collection[str]]) -> None:
        captured.append(tuple(step if isinstance(step, str) else frozenset(step) for step in steps))
        raise Planned

    try:
        render(Context("model-memory-plan", float("inf"), _plan=planned))
    except Planned:
        pass
    else:
        raise AssertionError("model work started without its complete memory plan")
    assert captured == [expected], (captured, expected)


def sdxl() -> None:
    sys.path.insert(0, str(ROOT / "sdxl"))
    import sdxl as package

    # The first actual declaration ends execution before construction, tokenizing,
    # scheduling, CUDA or media access. Uninitialized models own no weights.
    model = object.__new__(package.SdxlModel)
    attempt = fake_attempt("sdxl-plan")
    out, tel = fake_outputs(attempt), fake_telemetry(attempt)
    for steps in (1, 20, 30, 50):
        payload = package.Txt2ImgInput(prompt="plan", seed=62, steps=steps)
        capture(
            partial(package.generate, payload=payload, model=model, out=out, tel=tel),
            (frozenset({"text_encoder", "text_encoder_2"}), *(["unet"] * steps), "vae"),
        )
    print("PASS SDXL memory plan: every requested UNet evaluation and unchanged encoder/decode phases")


def qwen() -> None:
    sys.path.insert(0, str(ROOT / "qwen-image-2"))
    import qwen_image_2 as package
    from qwen_image_2.model import QwenImage21Graph, QwenImage21Model

    model = object.__new__(QwenImage21Model)
    model.graph = object.__new__(QwenImage21Graph)
    attempt = fake_attempt("qwen-plan")
    out, tel = fake_outputs(attempt), fake_telemetry(attempt)
    for references in (False, True):
        for requested, fixed, expected in ((None, None, 40), (7, None, 7),
                                           (None, 8, 8), (40, 8, 8)):
            model.graph.sample_sigmas = [1.0 - index / fixed for index in range(fixed)] if fixed is not None else None
            payload = package.GenerateInput(
                "plan", seed=62, steps=requested,
                reference_images=[ImageAsset("reference")] if references else [],
            )
            adj = fake_adjustments(attempt)
            before = len(adj.rows)
            capture(
                partial(package._generate, payload=payload, model=model, out=out, tel=tel, adj=adj),
                (*(["vae"] if references else []), "text_encoder",
                 *(["transformer"] * expected), "vae"),
            )
            changed = fixed is not None and requested not in (None, expected)
            assert len(adj.rows) - before == int(changed)
            if changed:
                row = adj.rows[-1]
                assert (row.field, row.requested, row.applied) == ("steps", requested, expected)
    print("PASS Qwen memory plan: resolved checkpoint/default/requested schedule and reference-VAE prefix")


if __name__ == "__main__":
    if sys.argv[1:] == ["sdxl"]:
        sdxl()
    elif sys.argv[1:] == ["qwen-image-2"]:
        qwen()
    else:
        raise SystemExit("use sdxl or qwen-image-2")
