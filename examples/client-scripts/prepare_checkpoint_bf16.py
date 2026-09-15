# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.2,<1", "numpy>=1.26"]
# [tool.cozy.models]
# source = "paul/minimax-h3-spatial-physics-lora@1.0.0/fp16"
# [tool.cozy.weights]
# model = 1073741824
# ///
"""Prepare a BF16 checkpoint through the shared producer, before model inference.

Requires the development Runtime containing PR517 on the client and worker.
Override model.source for another checkpoint; adjust the declared output budget
for sources larger than these LoRAs. The result is retained, ready for explicit
upload/publication through the ordinary checkpoint workflow.
"""

from cozy_runtime.author import Context, Model, ModelArtifact, Telemetry
from cozy_runtime.derive import plan, quantize_artifact


def main(ctx: Context, *, source: Model[object], tel: Telemetry) -> ModelArtifact:
    return quantize_artifact(
        source,
        plan(bf16_sources=("f16", "f32")),
        ctx=ctx,
        tel=tel,
        output="model",
    )
