"""Execute one explicit control cell and retain observations for later review."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import msgspec
from cozy_eval import contract
from cozy_eval.errors import DataError
from cozy_eval.jobs import compare_media, measure_media, measure_quality
from cozy_eval.jobs.measurements import ImageSource
from cozy_eval.measurement_facts import read_media, read_pair
from cozy_eval.quality_facts import read_quality
from cozy_runtime.author import ChildCallError, FileAsset, ImageAsset, ModelArtifact, Outputs, Tree

from .composition import Assessment, retain_report
from .control_inputs import ControlInputs, policy_bytes
from .control_jobs import BLUR_RADIUS, FLAT_RGB, corrupt_media
from .image_files import retain_image
from .report_bundle import file_suffix, report_bundle


async def run_control(
    *,
    ctx: Any,
    out: Outputs,
    inputs: ControlInputs,
    case: str,
    generate: Any,
    reference: ModelArtifact,
    candidate: ModelArtifact,
    judge: ModelArtifact,
    policy_digest: str,
) -> Tree:
    """No release effect or adoption: even a provisional PASS is just an observation."""
    outputs: list[FileAsset] = []
    evidence: dict[str, Any] = {
        "schema": "sdxl-control-evidence@1",
        "split": inputs.split,
        "case": case,
        "status": "measured-awaiting-review",
        "independent_human_judgments": False,
        "authored_expectations": inputs.expectations["labels"],
        "base_proposed_conditions": inputs.base_conditions,
        "control_conditions": policy_digest,
        "workload": inputs.policy.workload.digest(),
        "reference": msgspec.to_builtins(reference),
        "candidate": msgspec.to_builtins(candidate),
        "judge": msgspec.to_builtins(judge),
        "publication_permitted": False,
    }
    if case in ("null", "quantized", "scale_x2"):
        assessment = Assessment(
            generate=generate, out=out, policy=inputs.policy,
            reference=reference, candidate=candidate, judge=judge,
        )
        try:
            report = await assessment.run()
        except (ChildCallError, DataError) as error:
            # A failed damaged-model call is evidence, never a complete measured FAIL.
            evidence["status"] = "incomplete-control-awaiting-review"
            evidence["assessment"] = None
            evidence["failure"] = {
                "type": type(error).__name__,
                "reason": str(error),
                "request_id": error.child_request_id if isinstance(error, ChildCallError) else None,
            }
            ctx.log(f"Control {inputs.split}/{case} incomplete; failure details are in the bundle")
        else:
            report_file, workload_file = await retain_report(report, inputs.policy, out)
            outputs.extend((report_file, workload_file))
            evidence["assessment"] = {"digest": report_file.digest, "verdict": report.verdict.value}
            ctx.log(
                f"Control {inputs.split}/{case}: provisional {report.verdict}; "
                "review still required"
            )
        # Keep review images as ordinary FileAssets, without altering capture/report joins.
        evidence["review_images"] = []
        for rendered_image in assessment.images.values():
            retained = await retain_image(rendered_image, out)
            outputs.append(retained)
            evidence["review_images"].append(
                {"image": rendered_image.digest, "file": retained.digest}
            )
    else:
        assessment = Assessment(
            generate=generate, out=out, policy=inputs.policy,
            reference=reference, candidate=reference, judge=judge,
        )
        checklists = await out.commit(
            out.save_bytes(inputs.policy.checklist_bytes, media_type="application/json")
        )
        evidence["observations"] = []
        workload = inputs.policy.workload
        for index, prompt in enumerate(workload.prompts):
            baseline = await assessment.render(
                workload, index, arm="reference", model=reference.manifest.digest,
                capture=workload.capture,
            )
            image: ImageAsset = assessment.images[baseline.media]
            original = image
            actual_prompt = prompt
            generated = None
            transformation = None
            if case in ("wrong_object", "wrong_color"):
                actual_prompt = inputs.expectations[f"{case}_prompts"][index]
                payload = {**workload.payloads[index], "prompt": actual_prompt}
                call = generate(**payload, model=reference)
                result = await call
                if not call.request_id or call.observation is None:
                    raise DataError("counterfactual render has no actual request observation")
                if call.request_id in assessment.requests:
                    raise DataError("counterfactual render reused a baseline request")
                if (result.width, result.height, result.steps) != (
                    workload.width, workload.height, workload.steps
                ):
                    raise DataError("counterfactual output geometry differs from the control")
                assessment.requests.add(call.request_id)
                image = result.image
                generated = {
                    "request_id": call.request_id,
                    "observation": msgspec.to_builtins(call.observation),
                }
            else:
                corrupt_call = cast(Callable[..., Any], corrupt_media)
                call = corrupt_call(image=original, kind=case)
                image = (await call).image
                transformation = {
                    "request_id": call.request_id,
                    "kind": case,
                    "blur_radius": BLUR_RADIUS if case == "blur" else None,
                    "flat_rgb": FLAT_RGB if case == "flat" else None,
                }
            single = await measure_media(media=ImageSource(image))
            pair = await compare_media(
                reference=ImageSource(original), candidate=ImageSource(image)
            )
            quality = await measure_quality(
                media=ImageSource(image), prompt=prompt, seed=workload.seed(index),
                checklist_id=workload.checklist_ids[index], checklists=checklists,
                metrics=("element_recall",), judge=judge,
            )
            review = await retain_image(image, out)
            baseline_review = await retain_image(original, out)
            outputs.extend((review, baseline_review, single.facts, pair.facts, quality.facts))
            evidence["observations"].append({
                "index": index,
                "requested_prompt": prompt,
                "generated_prompt": actual_prompt,
                "baseline": {
                    "request_id": baseline.request_id,
                    "checkpoint": baseline.checkpoint,
                    "environment": msgspec.to_builtins(baseline.environment),
                },
                "baseline_image": original.digest,
                "control_image": image.digest,
                "review_file": review.digest,
                "generated": generated,
                "transformation": transformation,
                "media": msgspec.to_builtins(read_media(single.facts.read_bytes())),
                "pair": msgspec.to_builtins(read_pair(pair.facts.read_bytes())),
                "quality": msgspec.to_builtins(read_quality(quality.facts.read_bytes())),
                "assessment_verdict": None,
                "reason": "a deliberate prompt/pixel control is not a paired checkpoint assessment",
            })
            ctx.log(f"Retained unreviewed {inputs.split}/{case} control {index + 1}/8")
    frozen = await out.commit(
        out.save_bytes(policy_bytes(inputs.policy.conditions), media_type="application/json")
    )
    manifest = await out.commit(
        out.save_bytes(contract.canonical(evidence), media_type="application/json")
    )
    files = {"evidence.json": manifest, "conditions.json": frozen}
    files.update({f"artifact-{index:03}{file_suffix(file.media_type)}": file
                  for index, file in enumerate(outputs)})
    return report_bundle(files, out)
