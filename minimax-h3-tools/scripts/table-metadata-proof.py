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
    DIT_COMPONENT,
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
)
from h3_tables.plans import TimestepPlan
from h3_tables.table_layout import TableLayout
from h3_tables.source import config

VECTORS = Path(__file__).resolve().parents[2] / "scripts" / "lane-headers"

def main() -> None:
    # This proof checks geometry from captured model headers, not scheduler numerics.
    source = tensorfs.open(VECTORS / "minimax-h3.cozytensors", local_files_only=True)
    plans = {}
    for task, component in DIT_COMPONENT.items():
        table_keys = source.components[component].config["cozy_h3"]["table_keys"]
        layout = TableLayout.parse(table_keys)
        plans[task] = TimestepPlan(
            task, "", canonical_json.encode({"table_keys": table_keys}),
            layout.timesteps, layout.block_keys, (),
        )
    current = job._asset("model-config.json")
    sections = parse_production_config(current)
    assert dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"]) == current
    assert sorted(sections) == sorted(COMPONENTS)
    for name, want in (("minimax-h3", current), ("minimax-h3-full", dual_full_config(sections))):
        header = tensorfs.open(VECTORS / f"{name}.cozytensors", local_files_only=True)
        assert config(header) == want, f"{name}'s component configs are not the producer's"

    # Body and bank selection use row layouts, not formatting or full plan identities.
    bank = canonical_json.decode(_bank_config("fl2va", "sha256:" + "11" * 32, plans["fl2va"]))
    assert "plan" not in bank and bank["table_keys"] == plans["fl2va"].table_keys
    print("PASS the lane headers' component configs are the producer's; bank rows carry table_keys")


if __name__ == "__main__":
    main()
