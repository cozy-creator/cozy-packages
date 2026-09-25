#!/usr/bin/env python
"""Check the qwen-image-2 request and PNG contract in its installed dependency cohort."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

import msgspec
import torch
from cozy_runtime.author import canonical_json, describe

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen-image-2"))

from qwen_image_2 import (  # noqa: E402
    _TIER_DEMAND,
    AspectRatio,
    GenerateInput,
    Megapixels,
    app,
    reference_prompt,
    rgb_image,
)

payload = msgspec.json.decode(b'{"prompt":"An explorer in a blue coat"}', type=GenerateInput)
assert payload.seed is None and payload.steps == 40
assert payload.dimensions() == (1024, 1024) and payload.background == "normal"
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
        assert width * height <= _TIER_DEMAND[tier][0] * _TIER_DEMAND[tier][1]
        reverse = GenerateInput(
            "subject", aspect_ratio=AspectRatio(f"{ratio_h}:{ratio_w}"), megapixels=tier
        )
        assert reverse.dimensions() == (height, width)
        previous_area = width * height
assert GenerateInput("subject", AspectRatio.ULTRAWIDE, Megapixels.MP4).dimensions() == (3136, 1344)
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
for document in (
    b'{"prompt":"subject","width":1024}',
    b'{"prompt":"subject","height":1024}',
    b'{"prompt":"subject","aspect_ratio":"32:9"}',
    b'{"prompt":"subject","megapixels":8}',
    b'{"prompt":"subject","megapixels":0}',
    b'{"prompt":"subject","megapixels":1.5}',
    b'{"prompt":"subject","width":2752,"height":2752}',
    b'{"prompt":"subject","steps":0}',
    b'{"prompt":"subject","seed":-1}',
    b'{"prompt":"subject","seed":9007199254740992}',
    b'{"prompt":"subject","background":"transparent"}',
):
    try:
        msgspec.json.decode(document, type=GenerateInput)
    except msgspec.ValidationError:
        pass
    else:
        raise AssertionError(f"invalid request accepted: {document!r}")
assert reference_prompt("a railway platform", "normal") == "a railway platform"
assert "pure white studio background" in reference_prompt("a character", "white")
pixels = torch.tensor([[[[1.0]], [[-1.0]], [[-1.0]], [[-1.0]]]])
assert rgb_image(pixels).getpixel((0, 0)) == (255, 255, 255)
pixels[0, 3, 0, 0] = 1
assert rgb_image(pixels).getpixel((0, 0)) == (255, 0, 0)
(surface,) = describe(app)
assert surface.name == "generate_image" and surface.kind == "entrypoint"
assert {binding.param for binding in surface.model_bindings} == {"model"}
print("qwen-image-2: 27 buckets, defaults, seeds, hardcut refusals and PNG pixels passed")
