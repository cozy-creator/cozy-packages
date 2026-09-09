# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime==0.14.1", "tensorfs==0.3.34", "numpy==2.5.1",
#   "torch==2.13.0", "torchvision==0.28.0",
#   "sdxl", "sdxl-assessment-client",
# ]
# [tool.uv.sources]
# sdxl = {path = "../../sdxl", editable = true}
# sdxl-assessment-client = {path = "./sdxl-assessment", editable = true}
# ///
"""Private SDXL assessment; proposed policy by default, publication disabled.

No inference is memoized. Only native preparation and completed measurement inputs
may reuse work. Run through the local Creator CLI after the declared public cohort
and explicitly selected owner-held judge checkpoint are available.
"""

from collections.abc import Awaitable, Callable
from importlib.resources import as_file, files
from typing import cast

from cozy_eval.errors import ConfigError
from cozy_eval.jobs.instrument_config import plan
from cozy_eval.jobs.normalize_instruments import prepare_instrument
from cozy_runtime.author import FileAsset, ModelArtifact, Outputs, ScriptContext
from cozy_runtime.author.sources import (
    convert_cozytensors,
    download_civitai,
    download_huggingface,
    source_files,
)
from sdxl_assessment_client import load_policy, require_approved_policy
from sdxl_assessment_client.composition import Assessment, publish, retain_report

from sdxl import generate
from sdxl.normalization import normalize
from sdxl.operations import quantize

SOURCE_VERSION = 128078
SOURCE_FILE = "civitai/files/92696"
ENCODING = "fp8-rowwise/1"
# Exact reviewed source/profile already qualified by the owner on CPU. Preparing it
# on this execution machine establishes a genuine retained ModelArtifact for the judge.
JUDGE_REPOSITORY = "Qwen/Qwen3-VL-2B-Instruct"
JUDGE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
JUDGE_PROFILE = "hf/qwen/qwen3-vl-2b-instruct/bf16/1"
CONDITIONS_FILE = "conditions.absolute.proposed.v2.json"
# Review/calibration comes before ratification. Empty means measurement-only.
APPROVED_CONDITIONS = ""
PUBLISH = False
DESTINATION = ""
RELEASE = ""
LANE = "fp8"
EXPECTED_REVISION = 0


async def main(ctx: ScriptContext, *, out: Outputs) -> list[FileAsset]:
    with as_file(files("sdxl_assessment_client") / "policy") as directory:
        policy = load_policy(directory, conditions_file=CONDITIONS_FILE)
    if PUBLISH:
        require_approved_policy(policy, APPROVED_CONDITIONS)
        if not DESTINATION or not RELEASE:
            raise ConfigError("publication needs an explicit destination and release")
    ctx.log("SDXL eight-prompt assessment; proposed conditions are not admission")
    metadata_source = await download_huggingface(
        JUDGE_REPOSITORY, revision=JUDGE_REVISION, files=tuple(plan("qwen2b").files)
    )
    metadata = await source_files(metadata_source)
    judge_source = await download_huggingface(
        JUDGE_REPOSITORY, revision=JUDGE_REVISION, profiles=(JUDGE_PROFILE,)
    )
    judge_raw = await convert_cozytensors(judge_source, profile=JUDGE_PROFILE)
    judge = await prepare_instrument(source=judge_raw, metadata=metadata, variant="qwen2b")
    ctx.log(f"Prepared retained judge: {judge.manifest.digest}")
    raw = await download_civitai(SOURCE_VERSION, file=SOURCE_FILE)
    converted = await convert_cozytensors(raw, profile="civitai/sdxl/single-file/1")
    # The caller surface omits services injected into the admitted child signature.
    normalize_call = cast(Callable[..., Awaitable[ModelArtifact]], normalize)
    quantize_call = cast(Callable[..., Awaitable[ModelArtifact]], quantize)
    reference = await normalize_call(source=converted)
    candidate = await quantize_call(source=reference, encoding=ENCODING)
    assessment = Assessment(
        generate=generate,
        out=out,
        policy=policy,
        reference=reference,
        candidate=candidate,
        judge=judge,
    )
    report = await assessment.run()
    report_file, workloads_file = await retain_report(report, policy, out)
    ctx.log("Provisional measurement result (not ratification):\n" + report.summary())
    if PUBLISH:
        receipt = await publish(
            report,
            policy,
            candidate,
            report_file=report_file,
            workload_file=workloads_file,
            approved_conditions=APPROVED_CONDITIONS,
            destination=DESTINATION,
            release=RELEASE,
            lane=LANE,
            expected_revision=EXPECTED_REVISION,
        )
        ctx.log(f"Release {receipt.release} revision {receipt.revision}: {receipt.observation}")
    return [report_file, workloads_file]
