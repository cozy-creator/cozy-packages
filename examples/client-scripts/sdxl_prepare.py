# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime>=0.18.67,<1", "tensorfs>=0.3.74,<0.4",
# ]
# ///
"""Convert one Civitai SDXL checkpoint into its served lanes and upload them.

Set VERSION, DESTINATION and QUANTIZED, then:

    cozy run examples/client-scripts/sdxl_prepare.py --rental=NAME --allow-publish org/model --await

Works for any SDXL single-file checkpoint (f16, bf16 or f32; CLIP position IDs present or
absent): the conversion writes the servable fp16 checkpoint directly. On the rental that
already ingested the version with `cozy model upload`, the download and conversion are memo
hits. The script is the upload path until Runtime-owned jobs accept a
`cozy run <fn> <in> <org/model>` destination; each quantized lane is Runtime's memoized
`quantize` job over the UNet.
"""

from collections.abc import Awaitable, Callable
from typing import cast

from cozy_runtime.author import ModelArtifact, ScriptContext
from cozy_runtime.author.publication import upload_checkpoint
from cozy_runtime.author.sources import convert_cozytensors, download_civitai
from cozy_runtime.derive.operations import QuantizationPlan, quantize

VERSION = 2883731
DESTINATION = "org/model"
QUANTIZED = ("fp8",)  # and/or "mxfp8"
ENCODINGS = {"fp8": "fp8-rowwise/1", "mxfp8": "mxfp8/1"}
PROFILE = "civitai/sdxl/single-file/1"


async def main(ctx: ScriptContext) -> dict[str, str]:
    source = await download_civitai(VERSION)
    converted = await convert_cozytensors(source, profiles=(PROFILE,))
    lanes = {"fp16": converted}
    call = cast(Callable[..., Awaitable[ModelArtifact]], quantize)
    plan = QuantizationPlan(components=("unet",))
    for name in QUANTIZED:
        lanes[name] = await call(source=converted, plan=plan, encoding=ENCODINGS[name])
    result = {}
    for name, model in lanes.items():
        checkpoint = await upload_checkpoint(model, destination=DESTINATION)
        ctx.log(f"{name}: {checkpoint.checkpoint}")
        result[name] = checkpoint.checkpoint
    return result
