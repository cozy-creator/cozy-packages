"""Authored control inputs and partition checks; these are not measured labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
from cozy_eval import contract
from cozy_eval.assessment import CONDITIONS_ABSENT, conditions_digest
from cozy_eval.conditions import Conditions, Workload
from cozy_eval.errors import ConfigError
from cozy_eval.ladder import PinnedInputs
from cozy_eval.metrics.adherence import load_checklists

from .configuration import Policy, load_policy, validate_policy

CASES = ("null", "quantized", "scale_x2", "wrong_object", "wrong_color", "blur", "flat")
SPLITS = ("calibration", "held_out")


@dataclass(frozen=True)
class ControlInputs:
    split: str
    policy: Policy
    expectations: dict[str, Any]
    base_conditions: str


def load_controls(
    directory: Path, split: str, conditions_file: str = "conditions.absolute.proposed.v2.json"
) -> ControlInputs:
    if split not in SPLITS:
        raise ConfigError("select calibration or held_out, never final admission inputs")
    proposed = load_policy(directory / "policy", conditions_file)
    bank = directory / "controls"
    workload = msgspec.json.decode((bank / f"{split}.workload.json").read_bytes(), type=Workload)
    other_split = "held_out" if split == "calibration" else "calibration"
    other = msgspec.json.decode(
        (bank / f"{other_split}.workload.json").read_bytes(), type=Workload
    )
    partitions = ((workload, other), (workload, proposed.workload), (other, proposed.workload))
    for left, right in partitions:
        if set(left.prompts) & set(right.prompts) or set(left.seeds) & set(right.seeds):
            raise ConfigError("calibration, held-out and final admission inputs must be disjoint")
    conditions = msgspec.structs.replace(
        proposed.conditions, inputs=PinnedInputs(prompts=workload.prompts, seeds=workload.seeds)
    )
    path = bank / f"{split}.checklists.json"
    policy = Policy(conditions, workload, load_checklists(path), path.read_bytes())
    validate_policy(policy)
    expectations = msgspec.json.decode((bank / f"{split}.expectations.json").read_bytes())
    if expectations.get("split") != split or len(expectations.get("labels", ())) != 8:
        raise ConfigError("control expectations must identify their exact eight-prompt split")
    for name in ("wrong_object_prompts", "wrong_color_prompts"):
        changed = expectations.get(name, ())
        if len(changed) != 8 or any(
            not isinstance(value, str) or not value or value == workload.prompts[index]
            for index, value in enumerate(changed)
        ):
            raise ConfigError("counterfactual controls require eight distinct authored prompts")
    for index, row in enumerate(expectations["labels"]):
        if (
            row.get("prompt_id") != workload.checklist_ids[index]
            or row.get("status") != "authored_expectation_unreviewed"
            or row.get("independent_human_judgment") is not False
            or any(len(row.get(case, ())) != 3 for case in CASES)
            or any(value is not None and type(value) is not bool
                   for case in CASES for value in row.get(case, ()))
        ):
            raise ConfigError("control labels must stay authored expectations until reviewed")
    return ControlInputs(split, policy, expectations, conditions_digest(proposed.conditions))


def validate_selection(inputs: ControlInputs, case: str, frozen_policy: str) -> str:
    if case not in CASES:
        raise ConfigError(f"control case must be one of {CASES}")
    digest: str = conditions_digest(inputs.policy.conditions)
    if inputs.split == "held_out" and frozen_policy != digest:
        raise ConfigError(
            "held-out execution requires its exact policy digest frozen before measurement"
        )
    return digest


def policy_bytes(conditions: Conditions) -> bytes:
    value: bytes = contract.dump(
        conditions,
        protocol_absent=CONDITIONS_ABSENT,
    )
    return value
