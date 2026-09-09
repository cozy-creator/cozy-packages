"""Private client composition over public Runtime/Eval APIs; no App or deployment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
from cozy_eval import contract
from cozy_eval.activations import ActivationMeasurements, read_activation_pair
from cozy_eval.assessment import Environment, Rendered, assess_lane, conditions_digest, seal
from cozy_eval.conditions import Conditions, Workload
from cozy_eval.errors import ConfigError, DataError
from cozy_eval.jobs import (
    compare_media,
    measure_activation_pair,
    measure_media,
    measure_quality,
    measure_weights,
)
from cozy_eval.jobs.measurements import ImageSource
from cozy_eval.measurement_facts import VideoMeasurements, read_media, read_pair
from cozy_eval.metrics.adherence import ChecklistSet, load_checklists
from cozy_eval.outputs import Arm, OutputScores, score_outputs
from cozy_eval.protocol import Verdict
from cozy_eval.quality_facts import read_quality
from cozy_eval.weights import read_weights
from cozy_runtime.author import ActivationCapture, FileAsset, ModelArtifact, Outputs, Tree
from cozy_runtime.author.publication import attach_assessment, publish_release, upload_checkpoint

REQUIRED = ("weights", "activations", "video", "quality")


@dataclass(frozen=True)
class Policy:
    conditions: Conditions
    workload: Workload
    checklists: ChecklistSet
    checklist_bytes: bytes


def validate_policy(policy: Policy) -> None:
    """Reject malformed/underspecified experiments before any model call is made."""
    conditions, workload = policy.conditions, policy.workload
    if conditions.family != "sdxl" or set(conditions.required) != set(REQUIRED):
        raise ConfigError("SDXL assessment requires weights, activations, video and quality")
    if not conditions.same_pod:
        raise ConfigError("the SDXL reference, repeat and candidate must share one worker boot")
    if len(workload.prompts) != 8 or len(set(workload.prompts)) != 8:
        raise ConfigError("SDXL assessment requires eight distinct authored prompts")
    if (workload.width, workload.height, workload.frames, workload.fps) != (1024, 1024, 1, 0):
        raise ConfigError("SDXL proof geometry is 1024x1024, one still frame")
    if not 1 <= workload.steps <= 50:
        raise ConfigError("SDXL steps must be in its published 1..50 request range")
    if conditions.inputs.prompts != workload.prompts or conditions.inputs.seeds != workload.seeds:
        raise ConfigError("workload prompts/seeds differ from the exact policy inputs")
    if "unet" not in workload.capture or 0 not in workload.capture_steps:
        raise ConfigError("SDXL assessment captures UNet at teacher-forced step zero")
    if len(workload.checklist_ids) != 8:
        raise ConfigError("every prompt needs its authored quality checklist")
    for index, payload in enumerate(workload.payloads):
        if payload.get("prompt") != workload.prompts[index] or payload.get("seed") != workload.seed(
            index
        ):
            raise ConfigError("SDXL payload prompt/seed differs from the workload identity")
        if payload.get("aspect_ratio") != "1:1" or payload.get("steps") != workload.steps:
            raise ConfigError("SDXL payload geometry/steps differ from the workload identity")
        if payload.get("hidiffusion") is not False:
            raise ConfigError("this weight-quantization experiment explicitly disables HiDiffusion")
        if set(payload) & {"model", "capture"}:
            raise ConfigError("models and capture are managed call options, not payload overrides")
        checklist = policy.checklists.t2i.get(workload.checklist_ids[index])
        if checklist is None or not checklist.items:
            raise ConfigError("a workload checklist is absent or empty")
    if not conditions.quality:
        raise ConfigError("quality must have explicit proposed or ratified budgets")


def load_policy(
    directory: Path, out: Outputs, conditions_file: str = "conditions.proposed.json"
) -> Policy:
    """Read versioned policy data; its digest is approved separately from measurement."""
    conditions = contract.load((directory / conditions_file).read_bytes(), expect=Conditions).body
    workloads = msgspec.json.decode(
        (directory / "workloads.json").read_bytes(), type=tuple[Workload, ...]
    )
    if len(workloads) != 1:
        raise ConfigError("the SDXL proof uses one eight-prompt workload")
    raw = (directory / "checklists.json").read_bytes()
    path = out.temporary_file(".json")
    path.write_bytes(raw)
    policy = Policy(conditions, workloads[0], load_checklists(path), raw)
    validate_policy(policy)
    return policy


def require_approved_policy(policy: Policy, approved_conditions: str) -> str:
    validate_policy(policy)
    expected = conditions_digest(policy.conditions)
    if not approved_conditions or approved_conditions != expected:
        raise ConfigError("publication requires the explicitly ratified conditions digest")
    return expected


def require_publishable(report: Any, policy: Policy, approved_conditions: str) -> None:
    """A numerical result under a proposal is not authorization to publish a lane."""
    expected = require_approved_policy(policy, approved_conditions)
    report.contract_check()
    if report.subject.conditions != expected:
        raise ConfigError("assessment used different conditions than the approved policy")
    if report.verdict is not Verdict.PASS:
        raise ConfigError(f"SDXL publication refused: assessment verdict is {report.verdict}")
    if tuple(report.required) != tuple(policy.conditions.required):
        raise ConfigError("assessment omitted a required policy category")


class Assessment:
    """Adapt real managed SDXL calls and immutable facts to Eval's existing fold."""

    def __init__(
        self,
        *,
        generate: Callable[..., Any],
        out: Outputs,
        policy: Policy,
        reference: ModelArtifact,
        candidate: ModelArtifact,
        judge: str,
    ):
        validate_policy(policy)
        if not judge:
            raise ConfigError("quality needs an explicit owner-selected judge checkpoint")
        self.generate, self.out, self.policy = generate, out, policy
        self.reference, self.candidate, self.judge = reference, candidate, judge
        self.images: dict[str, Any] = {}
        self.captures: dict[str, Tree] = {}
        self.requests: set[str] = set()
        self.checklists: FileAsset | None = None

    async def render(
        self, workload: Workload, index: int, *, arm: str, model: str, capture: tuple[str, ...]
    ) -> Rendered:
        artifact = self.candidate if arm == "candidate" else self.reference
        if model != artifact.manifest.digest:
            raise DataError("assessment arm selected a different native model artifact")
        # Every arm is a new serving call. There is no memoized render wrapper.
        pending = self.generate(
            **workload.payloads[index],
            model=artifact,
            capture=ActivationCapture(capture, workload.capture_steps),
        )
        result = await pending
        observation = pending.observation
        if not pending.request_id or pending.request_id in self.requests:
            raise DataError("a render reused another arm's request identity")
        if observation is None or observation.capture is None or observation.capture.tree is None:
            raise DataError("the managed render did not return its actual activation capture")
        if (result.width, result.height, result.steps) != (
            workload.width,
            workload.height,
            workload.steps,
        ):
            raise DataError("the actual render geometry differs from the declared workload")
        self.requests.add(pending.request_id)
        media = self.out.temporary_file(".png")
        media.write_bytes(result.image.read_bytes())
        self.images[str(media)] = result.image
        tree = observation.capture.tree
        self.captures[str(tree.path)] = tree
        env = observation.environment
        return Rendered(
            request_id=pending.request_id,
            checkpoint=artifact.manifest.digest,
            media=str(media),
            capture=str(tree.path),
            environment=Environment(
                runtime_version=env.runtime_version,
                worker_image=env.worker_image_digest,
                accelerator=env.accelerator,
                driver=env.driver,
                cuda=env.cuda,
                worker_boot_id=env.worker_boot_id,
                execution_lane=env.execution_lane,
                execution_contract=env.execution_contract_digest,
                kernel_symbol=env.kernel_symbol,
            ),
        )

    async def activations(
        self, reference: Sequence[str], repeat: Sequence[str], candidate: Sequence[str]
    ) -> ActivationMeasurements:
        async def pairs(paths: Sequence[str]):
            rows = []
            for left, right in zip(reference, paths, strict=True):
                measured = await measure_activation_pair(
                    reference=self.captures[left], candidate=self.captures[right]
                )
                rows.append(read_activation_pair(measured.facts.read_bytes()))
            return tuple(rows)

        return ActivationMeasurements(await pairs(candidate), await pairs(repeat))

    async def outputs(
        self, reference: Arm, repeat: Arm, candidate: Arm, **options: Any
    ) -> OutputScores:
        async def singles(arm: Arm):
            rows = []
            for path in arm.media:
                measured = await measure_media(media=ImageSource(self.images[path]))
                rows.append(read_media(measured.facts.read_bytes()))
            return tuple(rows)

        async def pairs(arm: Arm):
            rows = []
            for left, right in zip(reference.media, arm.media, strict=True):
                measured = await compare_media(
                    reference=ImageSource(self.images[left]),
                    candidate=ImageSource(self.images[right]),
                )
                rows.append(read_pair(measured.facts.read_bytes()))
            return tuple(rows)

        video = VideoMeasurements(
            await singles(reference),
            await singles(repeat),
            await singles(candidate),
            await pairs(candidate),
            await pairs(repeat),
        )
        quality = []
        assert self.checklists is not None
        for index, path in enumerate(candidate.media):
            workload = self.policy.workload
            measured = await measure_quality(
                media=ImageSource(self.images[path]),
                prompt=workload.prompts[index],
                seed=workload.seed(index),
                checklist_id=workload.checklist_ids[index],
                checklists=self.checklists,
                metrics=tuple(b.name for b in self.policy.conditions.quality),
                judge=self.judge,
            )
            quality.append(read_quality(measured.facts.read_bytes()))
        return score_outputs(
            reference,
            repeat,
            candidate,
            **options,
            video_measurements=video,
            quality_measurements=tuple(quality),
        )

    async def run(self):
        # The read-only measurement is a normal memoized Eval leaf. ModelArtifact
        # inputs become granted manifest-only Models; no writer or dummy output opens.
        measured = await measure_weights(reference=self.reference, candidate=self.candidate)
        weights = read_weights(measured.facts.read_bytes())
        self.checklists = await self.out.commit(
            self.out.save_bytes(self.policy.checklist_bytes, media_type="application/json")
        )
        return await assess_lane(
            self.reference.manifest.digest,
            self.candidate.manifest.digest,
            (self.policy.workload,),
            self.policy.conditions,
            weight_measurements=weights,
            render=self.render,
            measure_outputs=self.outputs,
            measure_activations=self.activations,
            checklists=self.policy.checklists,
        )


async def retain_report(report: Any, policy: Policy, out: Outputs) -> tuple[FileAsset, FileAsset]:
    report_file = await out.commit(out.save_bytes(seal(report), media_type="application/json"))
    workloads = contract.canonical(msgspec.to_builtins((policy.workload,)))
    workload_file = await out.commit(out.save_bytes(workloads, media_type="application/json"))
    return report_file, workload_file


async def publish(
    report: Any,
    policy: Policy,
    candidate: ModelArtifact,
    *,
    report_file: FileAsset,
    workload_file: FileAsset,
    approved_conditions: str,
    destination: str,
    release: str,
    lane: str,
    expected_revision: int | None = None,
):
    require_publishable(report, policy, approved_conditions)
    if not destination or not release or not lane:
        raise ConfigError("publication requires explicit destination, release and lane")
    checkpoint = await upload_checkpoint(candidate, destination=destination)
    attached = await attach_assessment(checkpoint, report=report_file, workloads=workload_file)
    if attached.verdict != Verdict.PASS.value:
        raise ConfigError(f"Hub inspection did not admit the report: {attached.verdict}")
    return await publish_release(
        destination=destination,
        release=release,
        lanes={lane: checkpoint},
        expected_revision=expected_revision,
    )
