"""Prepare an ingested BiRefNet checkpoint without rewriting its weight bytes."""

from __future__ import annotations

from importlib.resources import files

import torch
from cozy_runtime.author import Context, ModelArtifact, Weights, invocable
from cozy_runtime.derive import Derivation

from .model import BackgroundRemovalModel
from .upstream.BiRefNet_config import BiRefNetConfig
from .upstream.birefnet import BiRefNet


@invocable
async def normalize(ctx: Context, *, source: Weights[BackgroundRemovalModel]) -> ModelArtifact:
    """Add constructor configuration and order; retain original FP16 weights and buffers."""
    capability = ctx.tensorfs_source(source)
    with torch.device("meta"):
        expected = BiRefNet(config=BiRefNetConfig(bb_pretrained=False)).eval().half().state_dict()
    if set(capability.components) != {"model"}:
        raise ValueError("source components do not match the BiRefNet architecture")
    tensors = capability.components["model"].tensors
    if set(tensors) != set(expected):
        raise ValueError("source tensor keys do not match the BiRefNet architecture")
    dtypes = {torch.float16: "F16", torch.int64: "I64"}
    for key, tensor in expected.items():
        found = tensors[key]
        if tuple(found.shape) != tuple(tensor.shape) or found.dtype != dtypes[tensor.dtype]:
            raise ValueError(f"source tensor geometry differs from BiRefNet: {key}")
    definition = Derivation(capability, order=tuple(("model", key) for key in expected))
    with ctx.output("model", definition) as output:
        if output.done is None:
            ctx.raise_if_cancelled()
            output.component("model", config={"architecture": "BiRefNet", "bb_pretrained": False})
            for name in ("LICENSE", "NOTICE"):
                output.asset("model", name, files("birefnet.upstream").joinpath(name).read_bytes())
        return ctx.adopt_model(output)
