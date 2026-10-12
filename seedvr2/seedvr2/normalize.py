"""Compose native source checkpoints without rewriting their weight bytes."""
from __future__ import annotations

from importlib.resources import files
import torch
from cozy_runtime.author import Context, Loader, Model, ModelArtifact, canonical_json, invocable
from tensorfs.derived import Config, Derivation, Target
from .model import modules


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True)
async def normalize(ctx: Context, *, dit: Source, vae: Source) -> ModelArtifact:
    sources = {"dit": ctx.tensorfs_source(dit), "vae": ctx.tensorfs_source(vae)}
    with torch.device("meta"):
        expected = modules()
    order = []
    for name, module in expected.items():
        found = sources[name].inspect().components
        tensors = found.get("model", {})
        state = module.state_dict()
        if set(found) != {"model"} or set(tensors) != set(state):
            raise ValueError(f"{name} does not match the SeedVR2-3B tensor census")
        for key, tensor in state.items():
            if tuple(tensors[key].shape) != tuple(tensor.shape) or tensors[key].logical_dtype != "f16":
                raise ValueError(f"SeedVR2 geometry or dtype differs: {name}.{key}")
            order.append((name, key))
    upstream = files("seedvr2.upstream")
    definition = Derivation(
        sources=sources,
        targets={name: Target(source=name, source_component="model") for name in sources},
        configs={"model": Config("add")}, order=tuple(order),
        files={name: upstream.joinpath(name).read_bytes() for name in ("LICENSE", "NOTICE")},
    )
    with ctx.output("model").open(definition) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        if "model" not in set(transaction.completed_configs()):
            transaction.add_config("model", canonical_json.encode({"architecture": "SeedVR2-3B"}))
            transaction.checkpoint()
        return ctx.adopt_model(transaction.commit())
