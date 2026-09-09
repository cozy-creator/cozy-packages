#!/usr/bin/env python3
"""Exact four-output H3 callable and table-contract red arms."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from cozy_runtime.author import (
    WeightsPart,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
    canonical_json,
)
from cozy_runtime.derive.quantization import (
    h3_quantization_plan,
    prepare_quantization,
    quantization_additions,
)
from h3_tables import lanes as lane_recipes
from h3_tables.job import (
    FP8_SPEC,
    MAX_TABLE_BYTES,
    MXFP8_SPEC,
    PLAIN_SPEC,
    _full_order,
    _full_targets,
    _lane_targets,
    _retable_targets,
    _table_additions,
)
from h3_tables.kernel import H3Topology, removed_keys, source_shapes, table_shapes
from h3_tables.model_config import parse_production_config
from h3_tables.order import current_order
from h3_tables.plans import parse_declared_plan, parse_plan
from h3_tables.source import official_full_specs, source_only_keys, text_source_only_keys

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "src/h3_tables/assets"


def refuse(raw: bytes, task: str) -> None:
    try:
        parse_plan(raw, task=task)
    except ValueError:
        return
    raise RuntimeError(f"changed {task} TimestepPlan did not refuse")


def structure(rows: list[tuple[str, str, str, tuple[int, ...]]]) -> WeightsSource:
    return WeightsSource(
        ("model",),
        tuple(
            WeightsSourceTensor(
                component, key, dtype, shape, (WeightsSourcePart("value", dtype, shape),)
            )
            for component, key, dtype, shape in rows
        ),
    )


def prove_retable(sections: dict[str, dict[str, object]]) -> None:
    """The retable declarations edit only table keys and refuse the wrong sources."""
    tables = _table_additions(sections)
    shared = [(c, f"{c}.w", "f32", (1,)) for c in ("text_encoder", "video_vae", "audio_vae")]
    pruned_rows, full_rows = list(shared), list(shared)
    for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
        component = f"{task}_dit"
        topology = H3Topology.from_config(sections[section])
        pruned_rows += [(component, "proj_in.weight", "bf16", (2, 2))]
        pruned_rows += [(component, key, "bf16", (1, 6, 8)) for key in tables[task]]
        full_rows += [(component, "proj_in.weight", "bf16", (2, 2))]
        full_rows += [
            (component, key, {"torch.float32": "f32", "torch.bfloat16": "bf16"}[str(dtype)], shape)
            for key, (dtype, shape) in source_shapes(topology).items()
        ]
    bank, retabled = _retable_targets(
        structure(pruned_rows), structure(full_rows), sections, tables
    )
    if set(bank) != {"fl2va_dit", "ref2va_dit"}:
        raise RuntimeError("the table bank must hold exactly both DiTs")
    for task in ("fl2va", "ref2va"):
        component = f"{task}_dit"
        target = retabled[component]
        if (
            target.source != "pruned"
            or set(target.drop) != set(tables[task])
            or target.add != tables[task]
        ):
            raise RuntimeError(f"retable {task} target is not the exact table replacement")
        present = {key for owner, key, *_ in full_rows if owner == component}
        if (
            bank[component].source != "full"
            or set(bank[component].drop) != present
            or bank[component].add != tables[task]
        ):
            raise RuntimeError(f"table bank {task} target must drop every full row")
    if any(retabled[c].source != "pruned" or retabled[c].drop for c, *_ in shared):
        raise RuntimeError("retable shared components must derive unchanged from pruned")
    for name, bad_pruned, bad_full in (
        ("missing table", pruned_rows[:-1], full_rows),
        ("dynamic weights kept", [*pruned_rows, full_rows[-1]], full_rows),
        ("full lacks modulation", pruned_rows, full_rows[:-1]),
    ):
        try:
            _retable_targets(structure(bad_pruned), structure(bad_full), sections, tables)
        except ValueError:
            continue
        raise RuntimeError(f"retable accepted a source with {name}")


def _dit_structure(sections: dict[str, Any]) -> tuple[WeightsSourceTensor, ...]:
    """The two FULL DiT components as granted structure, from the package's own contract."""
    rows: list[WeightsSourceTensor] = []
    for component, section in (("fl2va_dit", "transformer"), ("ref2va_dit", "transformer_ref")):
        specs = dict(official_full_specs(sections[section]))
        specs["rope.inv_freq"] = ("f32", (16,))
        for key, (dtype, shape) in specs.items():
            rows.append(
                WeightsSourceTensor(
                    component=component,
                    key=key,
                    logical_dtype=dtype,
                    shape=tuple(shape),
                    parts=(WeightsSourcePart("value", dtype, tuple(shape)),),
                    encoding=PLAIN_SPEC,
                )
            )
    return tuple(rows)


def _targets_of(
    name: str,
    sections: dict[str, Any],
    tables: Any,
    full_targets: Any,
    dits: tuple[WeightsSourceTensor, ...],
    quantization: Any,
) -> dict[str, Any]:
    lane = lane_recipes.LANES[name]
    selections = {
        component: lane_recipes.select(
            component,
            treatment,
            lane_recipes.carried(full_targets[component], dits),
            dit_plan=quantization,
        )
        for component, treatment in lane.components.items()
    }
    return _lane_targets(lane, sections, tables, full_targets, selections)


def main() -> None:
    sections = parse_production_config((ASSETS / "model-config.json").read_bytes())
    whole = current_order((ASSETS / "whole-order.json").read_bytes())
    full = _full_order(sections, whole.rows)
    if len(full) != 3968:
        raise RuntimeError(f"full dual order has {len(full)} rows, expected 3968")
    if len(text_source_only_keys()) != 156:
        raise RuntimeError("official 1,058-row text source does not trim exactly to 902 rows")
    full_targets = _full_targets()
    for component in ("fl2va_dit", "ref2va_dit"):
        target = full_targets[component]
        if (
            target.source != "dits"
            or target.source_component != component
            or target.drop != source_only_keys()
        ):
            raise RuntimeError(f"{component} lost its direct full source mapping")

    tables = _table_additions(sections)
    prove_retable(sections)
    quantization = prepare_quantization(h3_quantization_plan())
    if (
        quantization.components != ["dit"]
        or len(quantization.order) != 584
        or len(quantization.tensors) != 313
        or any(tensor.component != "dit" for tensor in quantization.tensors)
    ):
        raise RuntimeError("quantization changed its closed 584/313 component plan")
    # Every lane's targets now come from its catalogue row, so this proof drives the exact
    # declaration path the job does rather than a second spelling of it.
    dits = _dit_structure(sections)
    pruned = _targets_of("bf16-adaln-pruned", sections, tables, full_targets, dits, quantization)
    fp8 = _targets_of("fp8-adaln-pruned", sections, tables, full_targets, dits, quantization)
    mxfp8 = _targets_of("mxfp8-adaln-pruned", sections, tables, full_targets, dits, quantization)

    measured_bytes: dict[str, int] = {}
    for task, source, component in (
        ("fl2va", "transformer", "fl2va_dit"),
        ("ref2va", "transformer_ref", "ref2va_dit"),
    ):
        raw = (ASSETS / f"timestep-plan.{task}.json").read_bytes()
        plan = parse_declared_plan(raw)
        topology = H3Topology.from_config(sections[source])
        shapes = table_shapes(topology, plan)
        emitted = sum(2 * shape[0] * shape[1] * shape[2] for shape in shapes.values())
        if len(shapes) != 51 or emitted > MAX_TABLE_BYTES:
            raise RuntimeError(
                f"{task} table contract has {len(shapes)} tensors and {emitted} bytes; "
                f"expected 51 tensors within {MAX_TABLE_BYTES} bytes"
            )
        measured_bytes[task] = emitted
        dynamic = set(removed_keys(topology))
        if len(dynamic) != 106 or source_only_keys() != ("rope.inv_freq",):
            raise RuntimeError(f"{task} does not drop 106 dynamic rows plus the native rope buffer")
        target = pruned[component]
        if (
            target.source != "dits"
            or target.source_component != component
            or set(target.drop) != dynamic | set(source_only_keys())
            or set(target.add) != set(shapes)
        ):
            raise RuntimeError(f"{component} lost its exact direct pruned edit")
        for key, shape in shapes.items():
            declared = target.add[key]
            if (
                declared.logical_dtype != "bf16"
                or declared.shape != shape
                or declared.encoding != PLAIN_SPEC
                or dict(declared.parts) != {"value": WeightsPart("bf16", shape)}
            ):
                raise RuntimeError(f"{component}/{key} changed its exact BF16 table bytes")
        selected = {tensor.key for tensor in quantization.tensors}
        for name, target, spec in (
            ("fp8", fp8[component], FP8_SPEC),
            ("mxfp8", mxfp8[component], MXFP8_SPEC),
        ):
            expected = quantization_additions(
                "fp8-rowwise/1" if name == "fp8" else "mxfp8/1",
                quantization,
                "dit",
            )
            if (
                set(target.drop) != dynamic | set(source_only_keys()) | selected
                or set(target.add) != set(shapes) | selected
                or set(expected) != selected
                or any(tensor.encoding != spec for tensor in expected.values())
            ):
                raise RuntimeError(f"{component} changed its exact {name} edit")
        refuse(raw, "ref2va" if task == "fl2va" else "fl2va")
        changed = copy.deepcopy(json.loads(raw))
        changed["schedules"][0]["evaluations"][0]["video_sigma"] = (0.5).hex()
        refuse(canonical_json.encode(changed), task)
        if plan.steps != (30, 40, 50):
            raise RuntimeError(f"{task} plan serves {plan.steps}, expected 30/40/50 steps")

    package_interface = json.loads((PROJECT / "metadata" / "package-interface.json").read_bytes())
    if "model_productions" in package_interface:
        raise RuntimeError("package retained the retired model-production graph")
    declared = package_interface.get("jobs")
    if not isinstance(declared, list):
        raise TypeError("package jobs are not one interface list")
    jobs = {str(row["name"]): row for row in declared}
    if set(jobs) != {
        "assemble_full",
        "lanes",
        "retable",
        "quantize-artifact",
        "apply-adaln",
        "compute-adaln-tables",
        "select-adaln-weights",
        "assemble-full-artifact",
        "retable-adaln",
        "restamp",
    }:
        raise RuntimeError(f"package callable compatibility changed: {sorted(jobs)}")
    job = jobs["lanes"]
    models = {row["path"]: row for row in job["models"]}
    if set(models) != {"lanes.models.dits", "lanes.models.shared"}:
        raise RuntimeError("lanes changed its two typed source slots")
    outputs = {output["output_id"]: output for output in job["weights_outputs"]}
    if set(outputs) != set(lane_recipes.LANES):
        raise RuntimeError(
            f"declared outputs {sorted(outputs)} are not the catalogue {sorted(lane_recipes.LANES)}"
        )
    if any("required_contract" in output for output in outputs.values()):
        raise RuntimeError("lanes retained a publish-time tensor requirements contract")
    if "resources" in job:
        raise RuntimeError("lanes should derive and measure resources instead of authoring them")
    retable = jobs["retable"]
    if {row["path"] for row in retable["models"]} != {
        "retable.models.full",
        "retable.models.pruned",
    } or {output["output_id"] for output in retable["weights_outputs"]} != {
        "adaln-pruned",
        "tables",
    }:
        raise RuntimeError("retable changed its two typed sources or two outputs")
    print(
        f"H3 LANE CONTRACT PASS jobs={len(jobs)} graphs=0 outputs={len(outputs)} full_rows=3968 "
        "task_rows=583 shared_text_drop=156 quantized_per_task=313 tables_per_task=51 "
        f"table_bytes_per_task={measured_bytes} table_budget_per_task={MAX_TABLE_BYTES} "
        "source_drop=rope dynamic_drops_per_task=106 "
        "direct_siblings=1 changed_plan=refused steps=30/40/50"
    )


if __name__ == "__main__":
    main()
