"""Private client composition over public Runtime/Eval APIs; no App or deployment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import msgspec
from cozy_eval import contract
from cozy_eval.activations import ActivationMeasurements, ActivationPairFacts, read_activation_pair
from cozy_eval.assessment import AssessmentReport, Environment, Rendered, assess_lane, seal
from cozy_eval.conditions import Workload
from cozy_eval.errors import ConfigError, DataError
from cozy_eval.jobs import (
    compare_media,
    measure_activation_pair,
    measure_media,
    measure_quality,
    measure_weights,
)
from cozy_eval.jobs.measurements import ImageSource
from cozy_eval.measurement_facts import (
    MediaFacts,
    PairFacts,
    VideoMeasurements,
    read_media,
    read_pair,
)
from cozy_eval.outputs import Arm, OutputScores, score_outputs
from cozy_eval.protocol import Verdict
from cozy_eval.quality_facts import read_quality
from cozy_eval.weights import read_weights
from cozy_runtime.author import ActivationCapture, FileAsset, ModelArtifact, Outputs, Tree
from cozy_runtime.author.publication import (
    ReleaseReceipt,
    attach_assessment,
    publish_release,
    upload_checkpoint,
)

from .configuration import Policy, require_publishable, validate_policy


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
        judge: ModelArtifact,
    ):
        validate_policy(policy)
        if not isinstance(judge, ModelArtifact):
            raise ConfigError(
                "quality needs a retained ModelArtifact from actual judge preparation"
            )
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
        if result.guidance != workload.payloads[index]["guidance"] or result.hidiffusion_applied:
            raise DataError("the actual render guidance or HiDiffusion differs from the workload")
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
        async def pairs(paths: Sequence[str]) -> tuple[ActivationPairFacts, ...]:
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
        async def singles(arm: Arm) -> tuple[MediaFacts, ...]:
            rows = []
            for path in arm.media:
                measured = await measure_media(media=ImageSource(self.images[path]))
                rows.append(read_media(measured.facts.read_bytes()))
            return tuple(rows)

        async def pairs(arm: Arm) -> tuple[PairFacts, ...]:
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

    async def run(self) -> AssessmentReport:
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
) -> ReleaseReceipt:
    require_publishable(report, policy, approved_conditions)
    if report.subject.candidate_checkpoint != candidate.manifest.digest:
        raise ConfigError("publication candidate differs from the assessed checkpoint")
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
