# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime==0.14.0", "tensorfs==0.3.34", "numpy==2.5.1",
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

from importlib.resources import as_file, files

from cozy_eval.errors import ConfigError
from cozy_runtime.author import FileAsset, Outputs, ScriptContext
from cozy_runtime.author.sources import convert_cozytensors, download_civitai
from sdxl_assessment_client import (
    Assessment,
    load_policy,
    publish,
    require_approved_policy,
    retain_report,
)

from sdxl import generate
from sdxl.normalization import normalize
from sdxl.operations import quantize

SOURCE_VERSION = 128078
SOURCE_FILE = "civitai/files/92696"
ENCODING = "fp8-rowwise/1"
# The owner selects an exact already-prepared instrument checkpoint; no implicit download.
JUDGE = ""
CONDITIONS_FILE = "conditions.proposed.json"
# Review/calibration comes before ratification. Empty means measurement-only.
APPROVED_CONDITIONS = ""
PUBLISH = False
DESTINATION = ""
RELEASE = ""
LANE = "fp8"
EXPECTED_REVISION = 0


async def main(ctx: ScriptContext, *, out: Outputs) -> list[FileAsset]:
    if not JUDGE:
        raise ConfigError("set JUDGE to the explicit owner-held evaluation instrument checkpoint")
    with as_file(files("sdxl_assessment_client") / "policy") as directory:
        policy = load_policy(directory, out, CONDITIONS_FILE)
    if PUBLISH:
        require_approved_policy(policy, APPROVED_CONDITIONS)
        if not DESTINATION or not RELEASE:
            raise ConfigError("publication needs an explicit destination and release")
    ctx.log("SDXL eight-prompt assessment; proposed conditions are not admission")
    raw = await download_civitai(SOURCE_VERSION, file=SOURCE_FILE)
    converted = await convert_cozytensors(raw, profile="civitai/sdxl/single-file/1")
    reference = await normalize(source=converted)
    candidate = await quantize(source=reference, encoding=ENCODING)
    assessment = Assessment(
        generate=generate,
        out=out,
        policy=policy,
        reference=reference,
        candidate=candidate,
        judge=JUDGE,
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
