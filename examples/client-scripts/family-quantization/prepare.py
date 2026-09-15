# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime>=0.18.0,<1", "tensorfs>=0.3.42,<0.4",
#   "family-native-source>=0.0.1", "sdxl", "anima",
# ]
# [tool.uv.sources]
# family-native-source = {path = "./native_source", editable = true}
# sdxl = {path = "../../../sdxl", editable = true}
# anima = {path = "../../../anima", editable = true}
# ///
"""Native family quantizer plumbing, not trained-model quality qualification."""

from collections.abc import Awaitable, Callable
from typing import cast

from cozy_runtime.author import ModelArtifact, ScriptContext
from family_native_source import Facts, inspect, produce

from anima.operations import quantize as anima_quantize
from sdxl.operations import quantize as sdxl_quantize

SDXL_THRESHOLD = None


async def main(ctx: ScriptContext) -> None:
    prepare = cast(Callable[..., Awaitable[ModelArtifact]], produce)
    read = cast(Callable[..., Awaitable[Facts]], inspect)
    for family, quantize in (("sdxl", sdxl_quantize), ("anima", anima_quantize)):
        source = await prepare(family=family)
        for encoding in ("fp8-rowwise/1", "mxfp8/1"):
            threshold = SDXL_THRESHOLD if family == "sdxl" and encoding == "fp8-rowwise/1" else None
            candidate = await cast(Callable[..., Awaitable[ModelArtifact]], quantize)(
                source=source, encoding=encoding, max_relative_frobenius=threshold
            )
            facts = await read(
                original=source, candidate=candidate, family=family, encoding=encoding
            )
            ctx.log(facts.summary)
