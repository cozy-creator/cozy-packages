"""Validate authored control partition/selection; never fetch, render or run a job."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cozy_eval import contract
from cozy_eval.assessment import conditions_digest
from cozy_eval.errors import ConfigError
from sdxl_assessment_client.control_inputs import (
    CASES,
    load_controls,
    policy_bytes,
    validate_selection,
)


def refuses(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ConfigError:
        return
    raise AssertionError("invalid control selection was accepted")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "examples/client-scripts/sdxl-assessment/sdxl_assessment_client"
    calibration = load_controls(directory, "calibration")
    held_out = load_controls(directory, "held_out")
    assert calibration.policy.conditions.inputs != held_out.policy.conditions.inputs
    for case in CASES:
        assert validate_selection(calibration, case, "")
        assert validate_selection(held_out, case, conditions_digest(held_out.policy.conditions))
    refuses(lambda: load_controls(directory, "final"))
    refuses(lambda: validate_selection(calibration, "unknown", ""))
    refuses(lambda: validate_selection(held_out, "null", ""))
    refuses(lambda: validate_selection(held_out, "null", calibration.base_conditions))
    for inputs in (calibration, held_out):
        frozen = contract.document_digest(policy_bytes(inputs.policy.conditions))
        assert frozen == conditions_digest(inputs.policy.conditions)
        assert len(inputs.policy.workload.prompts) == 8
        assert all(
            row["status"] == "authored_expectation_unreviewed"
            and row["independent_human_judgment"] is False
            for row in inputs.expectations["labels"]
        )
    print("control selection passes: three disjoint input partitions, seven named cases, "
          "unfrozen held-out refusal and authored-only labels; no image or weight execution")
    print("calibration conditions:", conditions_digest(calibration.policy.conditions))
    print("held-out conditions:", conditions_digest(held_out.policy.conditions))


if __name__ == "__main__":
    main()
