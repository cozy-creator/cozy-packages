"""Add constructor metadata to native as-is weights, without rewriting tensors."""

from __future__ import annotations

from importlib.resources import files

import torch
from cozy_runtime.author import Context, Loader, Model, ModelArtifact, canonical_json, invocable
from tensorfs.derived import Config, Derivation, Target

from .model import ARCHITECTURE
from .upstream.architecture import LatentResizer3D


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True)
async def normalize(ctx: Context, *, source: Source) -> ModelArtifact:
    """Validate the complete BF16 checkpoint and retain its original tensor objects."""
    capability = ctx.tensorfs_source(source)
    inspection = capability.inspect()
    with torch.device("meta"):
        expected = LatentResizer3D().eval().bfloat16().state_dict()
    tensors = inspection.components.get("model", {})
    if set(inspection.components) != {"model"} or set(tensors) != set(expected):
        raise ValueError("source keys differ from the H3 latent upscaler 3D Conv v1")
    for key, tensor in expected.items():
        found = tensors[key]
        if tuple(found.shape) != tuple(tensor.shape) or found.logical_dtype != "bf16":
            raise ValueError(f"source tensor differs from the BF16 3D Conv v1: {key}")
    attribution = files("h3_latent_upscaler.upstream")
    definition = Derivation(
        sources={"source": capability},
        targets={"model": Target(source="source", source_component="model")},
        configs={"model": Config("add")},
        order=tuple(("model", key) for key in expected),
        files={name: attribution.joinpath(name).read_bytes()
               for name in ("LICENSE", "WEIGHTS-LICENSE", "NOTICE")},
    )
    with ctx.output("model").open(definition) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        ctx.raise_if_cancelled()
        if "model" not in set(transaction.completed_configs()):
            transaction.add_config("model", canonical_json.encode({"architecture": ARCHITECTURE}))
            transaction.checkpoint()
        return ctx.adopt_model(transaction.commit())
