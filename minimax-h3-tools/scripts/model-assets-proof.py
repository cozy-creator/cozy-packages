#!/usr/bin/env python3
"""CPU proof of real checkpoint assets, source identity and derived asset retention.

Tiny native tensors exercise the actual TensorFS/Runtime writer APIs. Plans are regenerated
from the independent scheduler oracle. This is not a full-model or GPU numerical proof.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import tensorfs
from cozy_runtime.author import UnsupportedInput
from cozy_runtime.weights_store import Derivation, Output, replay_key
from h3_tables.adaln_operations import METADATA, _bank_config
from h3_tables.model_config import DIT_COMPONENT
from h3_tables.plans import TASKS, parse_plan
from h3_tables.source import TIMESTEP_PLAN, copy_assets, timestep_plan
from h3_tables.turbo import turbo_plan

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from h3_plans import oracle_plan


def refuse(call: Callable[[], object]) -> None:
    try:
        call()
    except UnsupportedInput as error:
        assert error.code == "adaln_plan", error
    else:
        raise AssertionError("an invalid or missing model plan was accepted")


def checkpoint(root: Path, name: str, assets: dict[str, bytes], *, config: dict | None = None,
               component: str = "transformer") -> tensorfs.Checkpoint:
    with tensorfs.Writer(root / f"{name}.cozytensors", cache_dir=root / "cache") as writer:
        part = writer.component(component, cls=("", "", ""), config=config or {})
        part.put("tiny", b"\x00\x00", dtype="BF16", shape=(1,))
        for key, value in assets.items():
            part.asset(key, value)
        return writer.commit()


def run(root: Path) -> None:
    raw = oracle_plan("fl2va").canonical_bytes()
    raw_ref = oracle_plan("ref2va").canonical_bytes()
    plan = parse_plan(raw, task="fl2va")
    source = checkpoint(root, "source", {TIMESTEP_PLAN: raw, "extra.json": b"{}"})
    assert timestep_plan(source, "fl2va") == plan
    refuse(lambda: timestep_plan(source, "ref2va"))
    missing = checkpoint(root, "missing", {})
    refuse(lambda: timestep_plan(missing, "fl2va"))
    wrong = checkpoint(root, "wrong-task", {TIMESTEP_PLAN: raw_ref})
    refuse(lambda: timestep_plan(wrong, "fl2va"))
    invalid = checkpoint(root, "invalid", {TIMESTEP_PLAN: b"not json"})
    refuse(lambda: timestep_plan(invalid, "fl2va"))
    overflowing_plan = json.loads(raw)
    overflowing_plan["video_shift"] = "0x1p+99999"
    overflowing = checkpoint(root, "overflowing", {
        TIMESTEP_PLAN: json.dumps(overflowing_plan).encode(),
    })
    refuse(lambda: timestep_plan(overflowing, "fl2va"))
    mismatched = checkpoint(root, "mismatch", {TIMESTEP_PLAN: raw}, config={
        "cozy_h3": {"modulation": "adaln-pruned", "table_keys": {}},
    })
    refuse(lambda: timestep_plan(mismatched, "fl2va"))

    # Header row positions remain authoritative when quantization carries tables through.
    # All approved rows plus an extra row are valid even after reordering both axes.
    table_keys = copy.deepcopy(plan.table_keys)
    for entries in table_keys.values():
        entries.reverse()
        for index, entry in enumerate(entries):
            entry["index"] = index
    table_keys["final_normalization"].append({
        "index": len(table_keys["final_normalization"]), "timestep": float(2).hex(),
    })
    table_keys["block_modulation"].append({
        "index": len(table_keys["block_modulation"]), "timestep": float(2).hex(),
        "modality": "video", "modality_tag": 0,
    })
    compatible_config = {"cozy_h3": {
        "modulation": "adaln-pruned", "table_keys": table_keys,
    }}
    compatible = checkpoint(root, "reordered-extra", {TIMESTEP_PLAN: raw},
                            config=compatible_config)
    assert timestep_plan(compatible, "fl2va") == plan
    carried_derivation = Derivation(base=compatible, order=(("transformer", "tiny"),))
    transaction = replay_key("owner", "code", "carried", carried_derivation.work(1 << 20))
    with Output(root / "carried", "carried", transaction, carried_derivation, 1 << 20) as out:
        carried = out.commit()
    assert carried.components["transformer"].config == compatible_config
    assert list(carried.chunks) == list(compatible.chunks)
    contents = bytearray(2)
    carried.components["transformer"].tensors.read_into("tiny", contents)
    assert contents == b"\x00\x00"
    assert timestep_plan(carried, "fl2va") == plan
    missing_keys = copy.deepcopy(table_keys)
    missing_keys["block_modulation"].pop(0)
    for index, entry in enumerate(missing_keys["block_modulation"]):
        entry["index"] = index
    missing_coverage = checkpoint(root, "missing-coverage", {TIMESTEP_PLAN: raw}, config={
        "cozy_h3": {"modulation": "adaln-pruned", "table_keys": missing_keys},
    })
    refuse(lambda: timestep_plan(missing_coverage, "fl2va"))

    # Checkpoint identity, rather than packaged MemoResource bytes, binds every plan byte.
    changed = checkpoint(root, "changed", {TIMESTEP_PLAN: raw + b"\n", "extra.json": b"{}"})
    assert source.id != changed.id
    assert timestep_plan(changed, "fl2va") == plan
    before = Derivation(base=None, sources=(source,), order=(("transformer", "tiny"),))
    after = Derivation(base=None, sources=(changed,), order=before.order)
    assert replay_key("owner", "code", "model", before.work(1 << 20)) != replay_key(
        "owner", "code", "model", after.work(1 << 20)
    )

    # Projection/bank retention preserves task plan and unrelated component assets.
    transaction = replay_key("owner", "code", "model", before.work(1 << 20))
    with Output(root / "output", "model", transaction, before, 1 << 20) as out:
        copy_assets(out, source, "transformer")
        out.component("transformer").asset(METADATA, _bank_config("fl2va", source.id, plan))
        out.copy("transformer", "tiny", source)
        bank = out.commit()
    assert bank.components["transformer"].asset(TIMESTEP_PLAN) == raw
    assert bank.components["transformer"].asset("extra.json") == b"{}"
    assert timestep_plan(bank, "fl2va") == plan

    # Native base-derived output keeps existing non-plan assets and may replace plan bytes.
    derivation = Derivation(base=source, sources=(changed,), order=before.order)
    transaction = replay_key("owner", "code", "retable", derivation.work(1 << 20))
    with Output(root / "retable", "retable", transaction, derivation, 1 << 20) as out:
        out.component("transformer").asset(TIMESTEP_PLAN, raw + b"\n")
        retabled = out.commit()
    assert retabled.components["transformer"].asset(TIMESTEP_PLAN) == raw + b"\n"
    assert retabled.components["transformer"].asset("extra.json") == b"{}"

    for task in TASKS:
        name = f"{task}_turbo"
        turbo_raw = oracle_plan(name).canonical_bytes()
        adapter = checkpoint(root, name, {TIMESTEP_PLAN: turbo_raw}, component=name)
        assert turbo_plan(adapter, task, name).steps == (8,)
        refuse(lambda adapter=adapter, task=task, name=name: timestep_plan(adapter, task, component=name))
        full = checkpoint(root, f"{task}-standard", {
            TIMESTEP_PLAN: oracle_plan(task).canonical_bytes(),
        }, component=name)
        try:
            turbo_plan(full, task, name)
        except UnsupportedInput:
            pass
        else:
            raise AssertionError("PDD accepted a standard schedule")
        assert DIT_COMPONENT[task] in ("transformer", "transformer_ref")
    print(json.dumps({"source_assets_required": True, "task_and_rows_checked": True,
        "plan_asset_changes_checkpoint_and_derivation": True, "projection_bank_assets_retained": True,
        "retable_assets_preserved": True, "pdd_exact_plan_required": True,
        "reordered_superset_header_preserved": True, "overflow_refused": True,
        "gpu_or_full_model_qualified": False}, sort_keys=True))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="h3-model-assets-") as directory:
        run(Path(directory))
