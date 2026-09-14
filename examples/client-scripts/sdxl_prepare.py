# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime>=0.18.0,<1", "tensorfs>=0.3.42,<0.4",
#   "sdxl",
# ]
# [tool.uv.sources]
# sdxl = {path = "../../sdxl", editable = true}
# ///
"""Prepare an FP8 SDXL checkpoint; edit constants and rerun to reuse retained work."""

from collections.abc import Awaitable, Callable
from typing import cast

from cozy_runtime.author import ModelArtifact, ScriptContext
from cozy_runtime.author.sources import convert_cozytensors, download_civitai
from cozy_runtime.derive.operations import quantize

from sdxl.normalization import normalize
from sdxl.operations import quantization_plan

SOURCE_VERSION = 128078
SOURCE_FILE = "civitai/files/92696"
ENCODING = "fp8-rowwise/1"


async def main(ctx: ScriptContext) -> ModelArtifact:
    source = await download_civitai(SOURCE_VERSION, file=SOURCE_FILE)
    converted = await convert_cozytensors(source, profile="civitai/sdxl/single-file/1")
    reference = await normalize(source=converted)
    ctx.log(f"Normalized SDXL reference: {reference.manifest.digest}")
    # Managed invocation binds artifacts and supplies the worker's native services.
    quantize_call = cast(Callable[..., Awaitable[ModelArtifact]], quantize)
    candidate = await quantize_call(source=reference, plan=quantization_plan(), encoding=ENCODING)
    ctx.log(f"FP8 candidate: {candidate.manifest.digest}")
    return candidate
