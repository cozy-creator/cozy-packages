"""Prepare an ingested BiRefNet checkpoint without rewriting its weight bytes."""

from __future__ import annotations

from importlib.resources import files

import torch
from cozy_runtime.author import Context, Loader, Model, ModelArtifact, canonical_json, invocable
from tensorfs.derived import Config, Derivation, Target

from .upstream.BiRefNet_config import BiRefNetConfig
from .upstream.birefnet import BiRefNet


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True)
async def normalize(ctx: Context, *, source: Source) -> ModelArtifact:
    """Add constructor configuration and order; retain original FP16 weights and buffers."""
    capability = ctx.tensorfs_source(source)
    inspection = capability.inspect()
    with torch.device("meta"):
        expected = BiRefNet(config=BiRefNetConfig(bb_pretrained=False)).eval().half().state_dict()
    tensors = inspection.components.get("model", {})
    if set(inspection.components) != {"model"} or set(tensors) != set(expected):
        raise ValueError("source tensor keys do not match the BiRefNet architecture")
    dtypes = {torch.float16: "f16", torch.int64: "i64"}
    for key, tensor in expected.items():
        found = tensors[key]
        if tuple(found.shape) != tuple(tensor.shape) or found.logical_dtype != dtypes[tensor.dtype]:
            raise ValueError(f"source tensor geometry differs from BiRefNet: {key}")
    definition = Derivation(
        sources={"source": capability},
        targets={"model": Target(source="source", source_component="model")},
        configs={"model": Config("add")},
        order=tuple(("model", key) for key in expected),
        files={
            "LICENSE": files("birefnet.upstream").joinpath("LICENSE").read_bytes(),
            "NOTICE": files("birefnet.upstream").joinpath("NOTICE").read_bytes(),
        },
    )
    with ctx.output("model").open(definition) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        ctx.raise_if_cancelled()
        if "model" not in set(transaction.completed_configs()):
            transaction.add_config(
                "model", canonical_json.encode({"architecture": "BiRefNet", "bb_pretrained": False})
            )
            transaction.checkpoint()
        return ctx.adopt_model(transaction.commit())
