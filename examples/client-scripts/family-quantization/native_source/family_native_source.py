"""Small native source and readback jobs; family quantizers live in their packages."""

import json

import msgspec
import numpy as np
import tensorfs
from cozy_runtime.author import (
    App,
    Context,
    ModelArtifact,
    WeightsOutput,
    WeightsPart,
    WeightsReader,
    WeightsSink,
    WeightsTarget,
    WeightsTensor,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource

app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]


@invocable(memoize=True)
async def produce(
    ctx: Context, *, family: str, variant: int = 0, invalid: bool = False, weights: WeightsSink
) -> ModelArtifact:
    component, dtype = ("unet", "f16") if family == "sdxl" else ("transformer", "bf16")
    width = 31 if invalid else 64
    names = ("block_a.weight", "block_b.weight")
    tensors = {
        name: WeightsTensor(dtype, (8, width), PLAIN, {"value": WeightsPart(dtype, (8, width))})
        for name in names
    }
    tensors["block_a.bias"] = WeightsTensor(dtype, (8,), PLAIN, {"value": WeightsPart(dtype, (8,))})
    other = WeightsTensor(dtype, (8, 64), PLAIN, {"value": WeightsPart(dtype, (8, 64))})
    order = (*((component, name) for name in tensors), ("shared", "untouched.weight"))
    with weights.open(
        "model",
        sources={},
        targets={
            component: WeightsTarget(add=tensors),
            "shared": WeightsTarget(add={"untouched.weight": other}),
        },
        order=order,
    ) as output:
        if output.replayed:
            assert output.receipt is not None
            return output.receipt.artifact
        for part_component, key in order:
            ctx.raise_if_cancelled()
            n = 8 if key.endswith(".bias") else 8 * (64 if part_component == "shared" else width)
            array = np.linspace(-1.731 + variant * 0.01, 1.929, n, dtype="<f4")
            raw = (
                array.astype("<f2").tobytes()
                if dtype == "f16"
                else (array.view("<u4") >> 16).astype("<u2").tobytes()
            )
            output.add_part(part_component, key, "value", raw)
        return output.commit().artifact


class Facts(msgspec.Struct):
    summary: str


@invocable(memoize=False)
async def inspect(
    ctx: Context,
    *,
    original: QuantizationSource,
    candidate: QuantizationSource,
    family: str,
    encoding: str,
    reader: WeightsReader,
) -> Facts:
    component, dtype = ("unet", "f16") if family == "sdxl" else ("transformer", "bf16")
    with reader.open(original) as source, reader.open(candidate) as derived:
        for name in ("block_a.weight", "block_b.weight"):
            row = derived.tensor(component, name)
            assert row.logical_dtype == dtype
            assert {part.name for part in row.parts} == {"data", "scale"}
            assert source.identity(component, name) != derived.identity(component, name)
            data = bytearray(512)
            derived.read_part_into(component, name, "data", 0, data)
            assert any(data)
        for part_component, name in ((component, "block_a.bias"), ("shared", "untouched.weight")):
            assert source.identity(part_component, name) == derived.identity(part_component, name)
        return Facts(
            json.dumps(
                {
                    "family": family,
                    "encoding": encoding,
                    "logical_dtype": dtype,
                    "encoded_keys": 2,
                    "inherited_keys": 2,
                },
                sort_keys=True,
            )
        )


app.job(produce, weights=(WeightsOutput("model", max_new_bytes=65536),))

app.job(inspect)
