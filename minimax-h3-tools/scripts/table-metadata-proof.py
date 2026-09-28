#!/usr/bin/env python3
"""Prove producer table metadata and bank row layouts without GPU/weight I/O."""

from __future__ import annotations

from cozy_runtime.author import canonical_json
from h3_tables import job
from h3_tables.adaln_operations import _bank_config
from h3_tables.model_config import dual_adaln_pruned_config, parse_production_config
from h3_tables.plans import TASKS


def main() -> None:
    plans = {task: job._production_plan(task) for task in TASKS}
    current = job._asset("model-config.json")
    sections = parse_production_config(current)
    assert dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"]) == current

    # Body and bank selection use row layouts, not formatting or full plan identities.
    bank = canonical_json.decode(_bank_config("fl2va", "sha256:" + "11" * 32))
    assert "plan" not in bank and bank["table_keys"] == plans["fl2va"].table_keys
    print("PASS producer config reproduces the committed asset; bank rows carry table_keys")


if __name__ == "__main__":
    main()
