# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime>=0.18.0,<1", "tensorfs>=0.3.42,<0.4",
#   "sdxl",
# ]
# [tool.uv.sources]
# sdxl = {path = "../../sdxl", editable = true}
# ///
"""Convert one Civitai SDXL checkpoint into its served lanes and upload them.

Set VERSION, DESTINATION and QUANTIZED, then:

    cozy run examples/client-scripts/sdxl_prepare.py --rental=NAME --allow-publish org/model --await

Works for any SDXL single-file checkpoint (f16, bf16 or f32; CLIP position IDs present or
absent): the conversion writes the servable fp16 checkpoint directly. On the rental that
already ingested the version with `cozy model upload`, the download and conversion are memo
hits. The script is the upload path until Runtime-owned jobs accept a
`cozy run <fn> <in> <org/model>` destination; each lane is the ordinary memoized package
function.
"""

from collections.abc import Awaitable, Callable
from typing import cast

from cozy_runtime.author import ModelArtifact, ScriptContext
from cozy_runtime.author.publication import upload_checkpoint
from cozy_runtime.author.sources import convert_cozytensors, download_civitai

from sdxl import fp8, mxfp8

VERSION = 2883731
DESTINATION = "org/model"
QUANTIZED = ("fp8",)
PROFILE = "civitai/sdxl/single-file/1"


async def main(ctx: ScriptContext) -> dict[str, str]:
    source = await download_civitai(VERSION)
    converted = await convert_cozytensors(source, profiles=(PROFILE,))
    lanes = {"fp16": converted}
    for name, function in (("fp8", fp8), ("mxfp8", mxfp8)):
        if name in QUANTIZED:
            call = cast(Callable[..., Awaitable[ModelArtifact]], function)
            lanes[name] = await call(source=lanes["fp16"])
    result = {}
    for name, model in lanes.items():
        checkpoint = await upload_checkpoint(model, destination=DESTINATION)
        ctx.log(f"{name}: {checkpoint.checkpoint}")
        result[name] = checkpoint.checkpoint
    return result
