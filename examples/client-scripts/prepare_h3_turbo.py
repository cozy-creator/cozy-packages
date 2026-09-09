# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.16.1", "minimax-h3-tools==2.11.0", "torch==2.13.0+cu130"]
# [tool.uv.sources]
# minimax-h3-tools = {path = "../../minimax-h3-tools", editable = true}
# torch = {index = "pytorch-cu130"}
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# [tool.cozy.models]
# full = "paul/minimax-h3@1.0.0-h3-audit.1/bf16-full"
# fl2va_adapter = "paul/minimax-h3@1.0.0-h3-audit.1/fl2va-adapter"
# ref2va_adapter = "paul/minimax-h3@1.0.0-h3-audit.1/ref2va-adapter"
# [tool.cozy.weights]
# full_input = 1
# fl2va_input = 1
# ref2va_input = 1
# ///
"""Produce and publish H3 turbo with an ordinary private Python transaction.

Run with `cozy run ./examples/client-scripts/prepare_h3_turbo.py --rental=NAME --await`.
The three input holds inherit native objects without rewriting tensor data.
The reusable producer owns PDD arithmetic; this script owns composition and publication.
"""

from cozy_runtime.author import Model, WeightsConfig, WeightsSink, WeightsTarget
from cozy_runtime.author.publication import publish_release, upload_checkpoint
from h3_tables.turbo import prepare_turbo


async def main(
    *,
    full: Model[object],
    fl2va_adapter: Model[object],
    ref2va_adapter: Model[object],
    weights: WeightsSink,
):
    inputs = {}
    for slot, output, model in (
        ("full", "full_input", full),
        ("fl2va_adapter", "fl2va_input", fl2va_adapter),
        ("ref2va_adapter", "ref2va_input", ref2va_adapter),
    ):
        structure = weights.structure(model)
        inputs[slot] = weights.derive(
            output,
            sources={"input": model},
            targets={
                row.component: WeightsTarget("input", row.component) for row in structure.tensors
            },
            configs={name: WeightsConfig("input", name) for name in structure.configs},
            order=tuple((row.component, row.key) for row in structure.tensors),
        ).artifact

    turbo = await prepare_turbo(**inputs)
    checkpoint = await upload_checkpoint(turbo, destination="paul/minimax-h3-turbo-lora")
    release = await publish_release(
        destination="paul/minimax-h3-turbo-lora",
        release="1.0.0-audit.1",
        lanes={"pdd8": checkpoint},
    )
    return {"checkpoint": checkpoint.checkpoint, "release": release.release}
