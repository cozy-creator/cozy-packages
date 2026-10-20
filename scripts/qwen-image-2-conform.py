#!/usr/bin/env python
"""Check the qwen-image-2 request and PNG contract in its installed dependency cohort."""

from __future__ import annotations

import math
import secrets
import sys
from pathlib import Path

import msgspec
import torch
from cozy_runtime.author import canonical_json, describe

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen-image-2"))

from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (  # noqa: E402
    calculate_dimensions,
)
from qwen_image_2 import (  # noqa: E402
    _DEMAND,
    GenerateInput,
    Megapixels,
    app,
    output_size,
    rgb_image,
)

payload = msgspec.json.decode(b'{"prompt":"An explorer in a blue coat"}', type=GenerateInput)
assert payload.seed is None and payload.steps == 40
assert payload.shape() == (Megapixels.MP4, "1:1")
assert output_size(Megapixels.MP4, "1:1") == (2048, 2048, False)
assert payload.shape((1200, 800)) == (Megapixels.MP1, "1200:800")
assert GenerateInput("x", "16:9", Megapixels.MP2).shape((4000, 3000)) == (Megapixels.MP2, "16:9")
assert _DEMAND[None] == _DEMAND[Megapixels.MP4]
# Every W:H from 1:10^4 to 10^4:1 stays inside the rotary range, 32 px sides and the demand
# admission declared. Up to 13:1 that is upstream's own size; past the range it is held.
for tier in Megapixels:
    area = tier * 1024 * 1024
    for step in range(-1000, 1001):
        w, h = round(10**4 * 10 ** (step / 250)), 10**4
        width, height, held = output_size(tier, f"{w}:{h}")
        assert width % 32 == height % 32 == 0 and 32 <= min(width, height)
        assert max(width, height) <= 32768 and width * height * 3 <= 20 << 20
        assert width * height <= _DEMAND[tier][0] * _DEMAND[tier][1], (tier, w, h)
        if 1 / 13 <= w / h <= 13:
            assert (width, height) == calculate_dimensions(area, w / h)[:2] and not held
        assert output_size(tier, f"{h}:{w}")[:2] == (height, width)
assert output_size(Megapixels.MP1, "5000:1") == (32768, 32, True)
# Exercise the actual selector at maximal entropy, then cross the real Runtime
# result serializer. This caught a default that inferred successfully but could
# not deliver its seed in the result document. No model or inference is invoked.
random_bits = secrets.randbits
try:
    secrets.randbits = lambda k: (1 << k) - 1
    seed = payload.resolved_seed()
    assert canonical_json.decode(canonical_json.encode({"seed": seed}))["seed"] == seed
finally:
    secrets.randbits = random_bits
for seed in (0, 9301, 9007199254740991):
    explicit = msgspec.json.decode(
        canonical_json.encode({"prompt": "subject", "seed": seed}), type=GenerateInput
    )
    assert explicit.resolved_seed() == seed
# Upstream encodes an empty prompt as a space; unknown fields are Runtime's to warn and drop.
assert msgspec.json.decode(b'{"prompt":"","width":1024}', type=GenerateInput).prompt == ""
pixels = torch.tensor([[[[1.0]], [[-1.0]], [[-1.0]], [[-1.0]]]])
assert rgb_image(pixels).getpixel((0, 0)) == (255, 255, 255)
pixels[0, 3, 0, 0] = 1
assert rgb_image(pixels).getpixel((0, 0)) == (255, 0, 0)
surfaces = {surface.name: surface for surface in describe(app)}
assert set(surfaces) == {"generate_image", "int8"}, sorted(surfaces)
surface = surfaces["generate_image"]
assert not surface.weights_outputs and surface.accelerator
assert [(b.param, b.weights) for b in surface.model_bindings] == [("model", False)]
quantizer = surfaces["int8"]
assert [o.name for o in quantizer.weights_outputs] == ["int8"] and not quantizer.accelerator
assert [(b.param, b.weights) for b in quantizer.model_bindings] == [("source", True)]
assert quantizer.memoize is None
print("qwen-image-2: any W:H inside the demand, seeds and PNG pixels passed")
