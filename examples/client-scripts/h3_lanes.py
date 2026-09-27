# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.40,<1", "minimax-h3-tools"]
# [tool.uv]
# default-groups = []
# [tool.uv.sources]
# minimax-h3-tools = {path = "../../minimax-h3-tools", editable = true}
# ///
"""Produce and upload every H3 lane and the PDD-8 turbo LoRA on the ingest rental.

Set BASE_DESTINATION and TURBO_DESTINATION, then on a CUDA rental:

    cozy run examples/client-scripts/h3_lanes.py --rental=NAME \\
      --allow-publish org/minimax-h3 --allow-publish org/minimax-h3-turbo-lora --await

The two downloads and conversions repeat `cozy model upload` of the same pinned sources, so
on the rental that ingested them they are memo hits. Each lane is the ordinary memoized
package function, which resumes a paused or interrupted attempt. The script is the upload
path until Runtime-owned jobs accept a `cozy run <fn> <in> <org/model>` destination.
"""

from collections.abc import Awaitable, Callable
from typing import cast

from cozy_runtime.author import ModelArtifact, ScriptContext
from cozy_runtime.author.publication import upload_checkpoint
from cozy_runtime.author.sources import convert_cozytensors, download_huggingface
from h3_tables.job import bf16_full, bf16_pruned, fp8_pruned, mxfp8_pruned
from h3_tables.turbo import turbo_lora

BASE = ("MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe")
BASE_PROFILES = ("hf/minimax-h3/native-dual-bf16/1", "hf/minimax-h3/shared-bf16/1")
PDD = ("alibaba-pai/MiniMax-H3-Acc-LoRAs", "335001fb9e5455d68a0caa18ec2e319072150328")
PDD_PROFILES = ("hf/minimax-h3/pdd-fl2va-bf16/1", "hf/minimax-h3/pdd-ref2va-bf16/1")
BASE_DESTINATION = "org/minimax-h3"
TURBO_DESTINATION = "org/minimax-h3-turbo-lora"


def _call(function: object) -> Callable[..., Awaitable[ModelArtifact]]:
    return cast(Callable[..., Awaitable[ModelArtifact]], function)


async def _ingest(repository: tuple[str, str], profiles: tuple[str, ...]) -> ModelArtifact:
    source = await download_huggingface(repository[0], revision=repository[1], profiles=profiles)
    return await convert_cozytensors(source, profiles=profiles)


async def main(ctx: ScriptContext) -> dict[str, str]:
    full = await _call(bf16_full)(source=await _ingest(BASE, BASE_PROFILES))
    lanes = {"bf16-full": full}
    for name, function in (
        ("fp8-pruned", fp8_pruned),
        ("bf16-pruned", bf16_pruned),
        ("mxfp8-pruned", mxfp8_pruned),
    ):
        lanes[name] = await _call(function)(source=full)
    pdd8 = await _call(turbo_lora)(adapters=await _ingest(PDD, PDD_PROFILES), base=full)
    result = {}
    for name, model in lanes.items():
        result[name] = (await upload_checkpoint(model, destination=BASE_DESTINATION)).checkpoint
        ctx.log(f"{BASE_DESTINATION} {name}: {result[name]}")
    result["pdd8"] = (await upload_checkpoint(pdd8, destination=TURBO_DESTINATION)).checkpoint
    ctx.log(f"{TURBO_DESTINATION} pdd8: {result['pdd8']}")
    return result
