# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime>=0.16.9,<1",
#   "minimax-h3-tools>=2.12.1,<3",
# ]
# [tool.uv]
# default-groups = []
# [tool.uv.sources]
# minimax-h3-tools = {path = "../../minimax-h3-tools", editable = true}
# [tool.cozy.models]
# full = "paul/minimax-h3@1.0.0-h3-audit.1/bf16-full"
# fl2va_adapter = "paul/minimax-h3@1.0.0-h3-audit.1/fl2va-adapter"
# ref2va_adapter = "paul/minimax-h3@1.0.0-h3-audit.1/ref2va-adapter"
# [tool.cozy.weights]
# model = 536870912
# ///
"""Produce and publish the standalone H3 turbo adapter through an ordinary Python main.

Run with `cozy run ./examples/client-scripts/prepare_h3_turbo.py --await`.
Add `--rental=NAME` to use an existing rental.

For local development, build Runtime with its `scripts/build-dev-wheel.py`, using
a verified native donor. Add the reported wheel path to [tool.uv.sources] above:
# cozy-runtime = {path = "/absolute/path/to/the-built-runtime.whl"}
The wheel must satisfy the declared Runtime floor, which includes typed Context
injection. No PyPI publication or global installation is needed.
"""

from cozy_runtime.author import Context, Telemetry, WeightsSink
from cozy_runtime.author.publication import publish_release, upload_checkpoint
from h3_tables.source import H3FullTransformer
from h3_tables.turbo import build_turbo_adapter


async def main(
    ctx: Context,
    *,
    full: H3FullTransformer,
    fl2va_adapter: H3FullTransformer,
    ref2va_adapter: H3FullTransformer,
    weights: WeightsSink,
    tel: Telemetry,
) -> dict[str, str]:
    turbo = build_turbo_adapter(
        ctx,
        full=full,
        fl2va_adapter=fl2va_adapter,
        ref2va_adapter=ref2va_adapter,
        weights=weights,
        tel=tel,
    )
    checkpoint = await upload_checkpoint(turbo, destination="paul/minimax-h3-turbo-lora")
    release = await publish_release(
        destination="paul/minimax-h3-turbo-lora",
        release="1.0.0-audit.1",
        lanes={"pdd8": checkpoint},
    )
    return {"checkpoint": checkpoint.checkpoint, "release": release.release}
