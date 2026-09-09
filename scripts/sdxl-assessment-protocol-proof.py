"""Validate authored SDXL protocol/gate inputs without model or Runtime invocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import msgspec
from cozy_eval.assessment import conditions_digest
from cozy_eval.errors import ConfigError
from sdxl_assessment_client import (
    load_policy,
    require_approved_policy,
    require_publishable,
    validate_policy,
)


def refuses(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ConfigError:
        return
    raise AssertionError("invalid/unapproved protocol was accepted")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "examples/client-scripts/sdxl-assessment/sdxl_assessment_client/policy"
    policy = load_policy(directory)
    assert len(policy.workload.prompts) == 8 and len(policy.workload.checklist_ids) == 8
    assert all(
        "PROPOSED" in budget.provenance
        for budget in (
            *policy.conditions.activation_absolute,
            *policy.conditions.weights,
            *policy.conditions.video,
            *policy.conditions.quality,
        )
    )
    assert policy.conditions.activation_absolute and not policy.conditions.caps
    assert not policy.conditions.video_paired
    refuses(lambda: require_approved_policy(policy, ""))
    refuses(lambda: require_approved_policy(policy, "sha256:" + "0" * 64))
    # No report or publication primitive is reached for an unapproved policy.
    refuses(lambda: require_publishable(None, policy, ""))
    original = policy.workload
    changed = msgspec.structs.replace(original, width=512)
    refuses(lambda: validate_policy(replace(policy, workload=changed)))
    changed = msgspec.structs.replace(original, capture=("vae",))
    refuses(lambda: validate_policy(replace(policy, workload=changed)))
    changed = msgspec.structs.replace(original, seeds=tuple(seed + 1 for seed in original.seeds))
    refuses(lambda: validate_policy(replace(policy, workload=changed)))
    payloads = tuple(dict(payload) for payload in original.payloads)
    payloads[0]["seed"] += 1
    changed = msgspec.structs.replace(original, payloads=payloads)
    refuses(lambda: validate_policy(replace(policy, workload=changed)))
    payloads = tuple(dict(payload) for payload in original.payloads)
    payloads[0]["hidiffusion"] = True
    changed = msgspec.structs.replace(original, payloads=payloads)
    refuses(lambda: validate_policy(replace(policy, workload=changed)))
    changed = msgspec.structs.replace(
        original, checklist_ids=("absent", *original.checklist_ids[1:])
    )
    refuses(lambda: validate_policy(replace(policy, workload=changed)))
    assert require_approved_policy(
        policy, conditions_digest(policy.conditions)
    ) == conditions_digest(policy.conditions)
    print(
        "protocol controls pass: eight proposed prompts, exact payload identity, "
        "capture/checklist/geometry "
        "refusals and unapproved publication refusal; "
        "no model, numeric SDXL result or ratification claimed"
    )


if __name__ == "__main__":
    main()
