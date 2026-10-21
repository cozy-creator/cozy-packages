#!/usr/bin/env python3
"""Prove producer table metadata, bank row layouts, and that real lane headers' component configs
are the documents the producer writes."""

from __future__ import annotations

from pathlib import Path

import tensorfs
from cozy_runtime.author import canonical_json
from h3_tables import job
from h3_tables.adaln_operations import _bank_config
from h3_tables.model_config import (
    COMPONENTS,
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
)
from h3_tables.plans import TASKS
from h3_tables.source import config

VECTORS = Path(__file__).resolve().parents[2] / "scripts" / "lane-headers"


def main() -> None:
    plans = {task: job._production_plan(task) for task in TASKS}
    current = job._asset("model-config.json")
    sections = parse_production_config(current)
    assert dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"]) == current
    assert sorted(sections) == sorted(COMPONENTS)
    for name, want in (("minimax-h3", current), ("minimax-h3-full", dual_full_config(sections))):
        header = tensorfs.open(VECTORS / f"{name}.cozytensors", local_files_only=True)
        assert config(header) == want, f"{name}'s component configs are not the producer's"

    # Body and bank selection use row layouts, not formatting or full plan identities.
    bank = canonical_json.decode(_bank_config("fl2va", "sha256:" + "11" * 32))
    assert "plan" not in bank and bank["table_keys"] == plans["fl2va"].table_keys
    print("PASS the lane headers' component configs are the producer's; bank rows carry table_keys")


if __name__ == "__main__":
    main()
