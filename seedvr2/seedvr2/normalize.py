"""Compose native source checkpoints without rewriting their weight bytes."""

from __future__ import annotations

from importlib.resources import files
import hashlib
from typing import Annotated, Any
import torch
from cozy_runtime.author import (
    AssetBound,
    Tree,
    Context,
    ModelArtifact,
    Weights,
    invocable,
)
from cozy_runtime.derive import Derivation
from .model import ARCHITECTURE, modules


@invocable
async def normalize(
    ctx: Context,
    *,
    dit: Weights[Any],
    vae: Weights[Any],
    conditioning: Annotated[Tree, AssetBound(max_bytes=2 << 20)],
) -> ModelArtifact:
    sources = {"dit": ctx.tensorfs_source(dit), "vae": ctx.tensorfs_source(vae)}
    with torch.device("meta"):
        expected = modules()
    order = []
    for name, module in expected.items():
        found = sources[name].components
        state = module.state_dict()
        if set(found) != {"model"}:
            raise ValueError(f"{name} does not have exactly its model component")
        tensors = found["model"].tensors
        if set(tensors) != set(state):
            raise ValueError(f"{name} does not match the {ARCHITECTURE} tensor census")
        for key, tensor in state.items():
            if (
                tuple(tensors[key].shape) != tuple(tensor.shape)
                or tensors[key].dtype != "F16"
            ):
                raise ValueError(f"SeedVR2 geometry or dtype differs: {name}.{key}")
            order.append((name, key))
    constants = {}
    for key, name, shape, digest in (
        (
            "positive",
            "pos_emb.pt",
            (58, 5120),
            "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799",
        ),
        (
            "negative",
            "neg_emb.pt",
            (64, 5120),
            "6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581",
        ),
    ):
        path = conditioning.path / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"upstream conditioning bytes differ: {name}")
        tensor = torch.load(path, weights_only=True, map_location="cpu")
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"upstream conditioning tensor differs: {name}")
        constants[key] = tensor
        order.append(("conditioning", key))
    upstream = files("seedvr2.upstream")
    definition = Derivation(None, sources=tuple(sources.values()), order=tuple(order))
    with ctx.output("model", definition) as output:
        if output.done is not None:
            return ctx.adopt_model(output)
        for name, checkpoint in sources.items():
            original = checkpoint.components["model"]
            writer = output.component(name, cls=original.cls, config=original.config)
            for asset in original.assets:
                writer.asset(asset, original.asset(asset))
            for key in original.tensors:
                output.copy(name, key, checkpoint, "model")
        output.component("conditioning", cls="", config={})
        for key, value in constants.items():
            if not output.has("conditioning", key):
                ctx.raise_if_cancelled()
                output.put("conditioning", key, "BF16", tuple(value.shape),
                           value.contiguous().view(torch.uint16).numpy().tobytes())
        output.component("model", cls="", config={"architecture": ARCHITECTURE})
        for name in ("LICENSE", "NOTICE"):
            output.asset("model", name, upstream.joinpath(name).read_bytes())
        return ctx.adopt_model(output)
