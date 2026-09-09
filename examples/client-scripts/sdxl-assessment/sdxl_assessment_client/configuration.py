"""Explicit experiment inputs and publication eligibility; no model execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
from cozy_eval import contract
from cozy_eval.assessment import conditions_digest
from cozy_eval.conditions import Conditions, Workload
from cozy_eval.errors import ConfigError
from cozy_eval.metrics.adherence import ChecklistSet, load_checklists
from cozy_eval.protocol import Verdict

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
    directory: Path, conditions_file: str = "conditions.absolute.proposed.v2.json"
) -> Policy:
    """Read versioned policy data; its digest is approved separately from measurement."""
    conditions = contract.load((directory / conditions_file).read_bytes(), expect=Conditions).body
    workloads = msgspec.json.decode(
        (directory / "workloads.json").read_bytes(), type=tuple[Workload, ...]
    )
    if len(workloads) != 1:
        raise ConfigError("the SDXL proof uses one eight-prompt workload")
    raw = (directory / "checklists.json").read_bytes()
    policy = Policy(conditions, workloads[0], load_checklists(directory / "checklists.json"), raw)
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
    workload = policy.workload
    if [(w.entrypoint, w.digest) for w in report.subject.workloads] != [
        (workload.entrypoint, workload.digest())
    ]:
        raise ConfigError("assessment used different workloads than the approved policy")
    if report.protocol.family != policy.conditions.family:
        raise ConfigError("assessment belongs to a different model family")
    requests = [request for arm in report.subject.arms.values() for request in arm.requests]
    if (
        len(requests) != 24
        or len(set(requests)) != 24
        or any(len(arm.requests) != 8 for arm in report.subject.arms.values())
    ):
        raise ConfigError("assessment must contain 24 independent render requests")
    if report.verdict is not Verdict.PASS:
        raise ConfigError(f"SDXL publication refused: assessment verdict is {report.verdict}")
    if tuple(report.required) != tuple(policy.conditions.required):
        raise ConfigError("assessment omitted a required policy category")
