#!/usr/bin/env python
"""Check the reference-image request and PNG contract in its installed dependency cohort."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

import msgspec
import torch
from cozy_runtime.author import canonical_json, describe

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference-image"))

from reference_image import GenerateInput, app, reference_prompt, rgb_image  # noqa: E402

payload = msgspec.json.decode(b'{"prompt":"An explorer in a blue coat"}', type=GenerateInput)
assert payload.seed is None and payload.steps == 40
assert (payload.width, payload.height, payload.background) == (1024, 1024, "normal")
# Exercise the actual selector at maximal entropy, then cross the real Runtime
# result serializer. This caught a default that inferred successfully but could
# not deliver its seed in the result document. No model or inference is invoked.
random_bits = secrets.randbits
try:
    secrets.randbits = lambda bits: (1 << bits) - 1
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
    b'{"prompt":"subject","width":1025}',
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
assert surface.name == "generate" and surface.kind == "entrypoint"
assert {binding.param for binding in surface.model_bindings} == {"model"}
print("reference-image: defaults, interoperable seeds, request refusal and PNG pixels passed")
