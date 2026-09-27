"""Explicit metadata migration for the original 345-frame H3 table checkpoints.

The serving loader never interprets legacy plan hashes. This producer-only upgrade proves
which historical row order the old stamp names before preserving those tensor bytes.
"""

from __future__ import annotations

from collections.abc import Mapping

from cozy_runtime.author import UnsupportedInput, canonical_json
from cozy_runtime.models.minimax_h3.table_layout import TableLayout

from .plans import TASKS, Task, TimestepPlan

_LEGACY_345_PLANS: dict[Task, str] = {
    "fl2va": "sha256:8cd647f223acb56f864773e1a86bd8bcc0bb8d7a1c83ce2de33b7209844dd049",
    "ref2va": "sha256:3ec1b8e59c8b5dc74a4656d299d25ae206249cbd2b3f90b3c2981f8123b1b4ac",
}


def upgrade_legacy_table_config(raw: bytes, plans: Mapping[Task, TimestepPlan]) -> bytes:
    """Replace only proved legacy stamps; preserve already explicit, valid table layouts.

    Reconstructing the historical document from the current validated plan and checking its
    original digest proves the rows are unchanged. A later scheduler change fails this
    proof, requiring recomputation instead of relabeling the old table bytes.
    """
    value = canonical_json.decode(raw)
    if not isinstance(value, dict):
        raise UnsupportedInput("H3 source has no model config", code="h3_restamp_config_shape")
    for task in TASKS:
        component = value.get(f"{task}_dit")
        stamp = component.get("cozy_h3") if isinstance(component, dict) else None
        if not isinstance(stamp, dict) or stamp.get("task") != task:
            raise UnsupportedInput("H3 source has no task metadata", code="h3_restamp_config_shape")
        if stamp.get("modulation") == "full":
            continue
        if stamp.get("modulation") != "adaln-pruned":
            raise UnsupportedInput(
                "H3 source has unknown modulation metadata", code="h3_restamp_config_shape"
            )
        if "table_keys" in stamp:
            TableLayout.parse(stamp["table_keys"])
            continue
        plan = plans[task]
        legacy = canonical_json.decode(plan.canonical_bytes)
        legacy["frames"] = 345
        old_digest = stamp.get("timestep_plan_digest")
        if old_digest != plan.digest and not (
            old_digest == _LEGACY_345_PLANS[task]
            and canonical_json.digest(legacy) == old_digest
        ):
            raise UnsupportedInput(
                f"{task} legacy timestep plan is not a proven unchanged table layout; "
                "regenerate its AdaLN tables from the full generating weights",
                code="h3_restamp_table_layout",
            )
        del stamp["timestep_plan_digest"]
        stamp["table_keys"] = plan.table_keys
    return canonical_json.encode(value)
