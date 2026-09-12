# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime==0.16.8", "tensorfs==0.3.38", "numpy==2.5.1",
#   "torch==2.13.0", "torchvision==0.28.0",
#   "sdxl", "sdxl-assessment-client",
# ]
# [tool.uv.sources]
# sdxl = {path = "../../sdxl", editable = true}
# sdxl-assessment-client = {path = "./sdxl-assessment", editable = true}
# ///
"""One explicitly selected SDXL control cell; never publish or claim ratification."""

from collections.abc import Awaitable, Callable
from importlib.resources import as_file, files
from typing import cast

from cozy_eval.jobs.instrument_config import plan
from cozy_eval.jobs.normalize_instruments import prepare_instrument
from cozy_runtime.author import ModelArtifact, Outputs, ScriptContext, Tree
from cozy_runtime.author.sources import (
    convert_cozytensors,
    download_civitai,
    download_huggingface,
    source_files,
)
from cozy_runtime.derive.operations import quantize
from sdxl_assessment_client.calibration import run_control
from sdxl_assessment_client.control_inputs import load_controls, validate_selection
from sdxl_assessment_client.control_jobs import scale_unet_x2

from sdxl import generate
from sdxl.normalization import normalize
from sdxl.operations import quantization_plan

SPLIT = "calibration"  # held_out uses a separately frozen policy; final inputs are disjoint.
CASE = "null"  # null, quantized, scale_x2, wrong_object, wrong_color, blur, flat
FROZEN_HELD_OUT_POLICY = ""
CONDITIONS_FILE = "conditions.absolute.proposed.v2.json"
JUDGE_REPOSITORY = "Qwen/Qwen3-VL-2B-Instruct"
JUDGE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
JUDGE_PROFILE = "hf/qwen/qwen3-vl-2b-instruct/bf16/1"


async def main(ctx: ScriptContext, *, out: Outputs) -> Tree:
    with as_file(files("sdxl_assessment_client")) as directory:
        inputs = load_controls(directory, SPLIT, CONDITIONS_FILE)
    digest = validate_selection(inputs, CASE, FROZEN_HELD_OUT_POLICY)
    ctx.log(f"Authored unreviewed controls {SPLIT}/{CASE}; proposed policy {digest}")
    metadata_source = await download_huggingface(
        JUDGE_REPOSITORY, revision=JUDGE_REVISION, files=tuple(plan("qwen2b").files)
    )
    metadata = await source_files(metadata_source)
    raw_judge = await download_huggingface(
        JUDGE_REPOSITORY, revision=JUDGE_REVISION, profiles=(JUDGE_PROFILE,)
    )
    converted_judge = await convert_cozytensors(raw_judge, profile=JUDGE_PROFILE)
    judge = await prepare_instrument(source=converted_judge, metadata=metadata, variant="qwen2b")
    raw = await download_civitai(128078, file="civitai/files/92696")
    converted = await convert_cozytensors(raw, profile="civitai/sdxl/single-file/1")
    normalize_call = cast(Callable[..., Awaitable[ModelArtifact]], normalize)
    reference = await normalize_call(source=converted)
    candidate = reference
    if CASE == "quantized":
        quantize_call = cast(Callable[..., Awaitable[ModelArtifact]], quantize)
        candidate = await quantize_call(
            source=reference, plan=quantization_plan(), encoding="fp8-rowwise/1"
        )
    elif CASE == "scale_x2":
        scale_call = cast(Callable[..., Awaitable[ModelArtifact]], scale_unet_x2)
        candidate = await scale_call(source=reference)
    return await run_control(
        ctx=ctx, out=out, inputs=inputs, case=CASE, generate=generate,
        reference=reference, candidate=candidate, judge=judge, policy_digest=digest,
    )
