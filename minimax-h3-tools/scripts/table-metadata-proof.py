#!/usr/bin/env python3
"""Prove table metadata upgrades and no-cast restamp behavior without GPU/weight I/O."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

from cozy_runtime.author import (
    ConformanceError,
    UnsupportedInput,
    canonical_json,
)
from h3_tables import job
from h3_tables.adaln_operations import _bank_config
from h3_tables.legacy_config import upgrade_legacy_table_config
from h3_tables.model_config import dual_adaln_pruned_config, parse_production_config
from h3_tables.plans import TASKS


def refuses(action: Callable[[], Any]) -> None:
    try:
        action()
    except (ConformanceError, UnsupportedInput, ValueError):
        return
    raise AssertionError("invalid metadata was accepted")


def main() -> None:
    plans = {task: job._production_plan(task) for task in TASKS}
    current = job._asset("model-config.json")
    sections = parse_production_config(current)
    assert dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"]) == current
    legacy = canonical_json.decode(current)
    for task, plan in plans.items():
        document = canonical_json.decode(plan.canonical_bytes)
        document["frames"] = 345
        stamp = legacy[f"{task}_dit"]["cozy_h3"]
        del stamp["table_keys"]
        stamp["timestep_plan_digest"] = canonical_json.digest(document)
    legacy_raw = canonical_json.encode(legacy)
    assert upgrade_legacy_table_config(legacy_raw, plans) == current
    assert upgrade_legacy_table_config(current, plans) == current

    # The historical digest proof fails if the producer's row meanings or schedule changed.
    for change in ("keys", "schedule"):
        plan = plans["fl2va"]
        changed = canonical_json.decode(plan.canonical_bytes)
        if change == "keys":
            changed["table_keys"]["final_normalization"][0]["timestep"] = (0.5).hex()
        else:
            changed["schedules"][0]["evaluations"][1]["video_sigma"] = (0.5).hex()
        altered = replace(plan, canonical_bytes=canonical_json.encode(changed))
        refuses(partial(upgrade_legacy_table_config, legacy_raw, {**plans, "fl2va": altered}))

    # Different valid row order is preserved; the restamp must not relabel existing bytes.
    custom = canonical_json.decode(current)
    for task in TASKS:
        stamp = custom[f"{task}_dit"]["cozy_h3"]
        stamp["generating_projection_digest"] = "sha256:" + "ab" * 32
        for rows in stamp["table_keys"].values():
            rows.reverse()
            for index, row in enumerate(rows):
                row["index"] = index
    custom_raw = canonical_json.encode(custom)
    assert upgrade_legacy_table_config(custom_raw, plans) == custom_raw
    unknown = copy.deepcopy(legacy)
    unknown["fl2va_dit"]["cozy_h3"]["timestep_plan_digest"] = "sha256:" + "ff" * 32
    refuses(partial(upgrade_legacy_table_config, canonical_json.encode(unknown), plans))
    duplicate = canonical_json.decode(current)
    rows = duplicate["fl2va_dit"]["cozy_h3"]["table_keys"]["final_normalization"]
    rows[1]["timestep"] = rows[0]["timestep"]
    refuses(partial(upgrade_legacy_table_config, canonical_json.encode(duplicate), plans))

    # Body and bank selection use row layouts, not formatting or full plan identities.
    bank = canonical_json.decode(_bank_config("fl2va", "sha256:" + "11" * 32))
    assert "plan" not in bank and bank["table_keys"] == plans["fl2va"].table_keys
    print(
        "PASS legacy provenance + layout proof; arbitrary row order preserved; "
        "unknown origins and duplicate labels refused"
    )


if __name__ == "__main__":
    main()
