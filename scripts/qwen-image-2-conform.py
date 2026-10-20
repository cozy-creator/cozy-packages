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

from qwen_image_2 import (  # noqa: E402
    _DEMAND,
    _MAX_RATIO,
    AspectRatio,
    GenerateInput,
    Megapixels,
    app,
    rgb_image,
)

payload = msgspec.json.decode(b'{"prompt":"An explorer in a blue coat"}', type=GenerateInput)
assert payload.seed is None and payload.steps == 40
assert payload.dimensions() == (1024, 1024)
assert len(AspectRatio) == 9 and {tier.value for tier in Megapixels} == {1, 2, 4}
for aspect in AspectRatio:
    previous_area = 0
    ratio_w, ratio_h = map(int, aspect.value.split(":"))
    for tier in Megapixels:
        request = msgspec.json.decode(
            canonical_json.encode(
                {"prompt": "subject", "aspect_ratio": aspect.value, "megapixels": tier.value}
            ),
            type=GenerateInput,
        )
        width, height = request.dimensions()
        assert width % 32 == height % 32 == 0
        assert previous_area < width * height <= 5_000_000
        assert width * height * 3 <= 20 << 20
        assert abs(width / height / (ratio_w / ratio_h) - 1) < 0.04
        assert width * height <= _DEMAND[tier][0] * _DEMAND[tier][1]
        reverse = GenerateInput(
            "subject", aspect_ratio=AspectRatio(f"{ratio_h}:{ratio_w}"), megapixels=tier
        )
        assert reverse.dimensions() == (height, width)
        previous_area = width * height
assert GenerateInput("subject", AspectRatio.ULTRAWIDE, Megapixels.MP4).dimensions() == (3136, 1344)
# A shape derived from a reference stays inside the demand admission declared. Rounding moves
# each side at most 16 px, so the capped ratio's worst case bounds every reference.
assert _DEMAND[None] == _DEMAND[Megapixels.MP4] == (2400, 1792)
for given in (*Megapixels, None):
    edge, side = math.sqrt((given or Megapixels.MP4) * 1024 * 1024), math.sqrt(_MAX_RATIO)
    worst = (edge * side + 16) * (edge / side + 16)
    assert worst <= _DEMAND[given][0] * _DEMAND[given][1], (given, worst)
    for step in range(-400, 401):
        reference = (round(1000 * 10 ** (step / 200)), 1000)
        width, height = GenerateInput("subject", megapixels=given).dimensions(reference)
        assert width % 32 == height % 32 == 0 and min(width, height) >= 32
        assert width * height <= _DEMAND[given][0] * _DEMAND[given][1], (reference, given)
assert GenerateInput("subject", AspectRatio.WIDE, Megapixels.MP4).dimensions() == (2752, 1536)
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
print("qwen-image-2: buckets, derived shapes in demand, seeds and PNG pixels passed")
