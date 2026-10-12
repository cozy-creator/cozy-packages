"""Compose native source checkpoints without rewriting their weight bytes."""

from __future__ import annotations

from importlib.resources import files
import hashlib
from typing import Annotated
import torch
from cozy_runtime.author import (
    AssetBound,
    Tree,
    Context,
    Loader,
    Model,
    ModelArtifact,
    canonical_json,
    invocable,
)
from tensorfs.derived import Config, Derivation, Target, Tensor, Part
from .model import modules


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True)
async def normalize(
    ctx: Context,
    *,
    dit: Source,
    vae: Source,
    conditioning: Annotated[Tree, AssetBound(max_bytes=2 << 20)],
) -> ModelArtifact:
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
            if (
                tuple(tensors[key].shape) != tuple(tensor.shape)
                or tensors[key].logical_dtype != "f16"
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
    definition = Derivation(
        sources=sources,
        targets={
            **{name: Target(source=name, source_component="model") for name in sources},
            "conditioning": Target(
                add={
                    key: Tensor(
                        "bf16",
                        tuple(value.shape),
                        "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f",
                        {"value": Part("bf16", tuple(value.shape))},
                    )
                    for key, value in constants.items()
                }
            ),
        },
        configs={"model": Config("add")},
        order=tuple(order),
        files={name: upstream.joinpath(name).read_bytes() for name in ("LICENSE", "NOTICE")},
    )
    with ctx.output("model").open(definition) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        done = set(transaction.completed_parts())
        for key, value in constants.items():
            if ("conditioning", key, "value") not in done:
                transaction.add_part(
                    "conditioning", key, "value", value.view(torch.uint16).numpy().tobytes()
                )
                transaction.checkpoint()
        if "model" not in set(transaction.completed_configs()):
            transaction.add_config("model", canonical_json.encode({"architecture": "SeedVR2-3B"}))
            transaction.checkpoint()
        return ctx.adopt_model(transaction.commit())
