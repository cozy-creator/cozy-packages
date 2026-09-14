#!/usr/bin/env python3
"""Prove optional source drops and declaration reuse through the author geometry view.

Uses the existing full construction order and optional source-only lists; no second
order file, tensor storage, GPU computation, or claimed numerical evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from cozy_runtime.author import canonical_json
from cozy_runtime.derive.quantization import (
    prepare_quantization,
)
from h3_tables import job, lanes, operations
from h3_tables.kernel import H3Topology, removed_keys
from h3_tables.model_config import parse_production_config
from h3_tables.order import current_order
from h3_tables.order import full_order as compute_order
from h3_tables.quantization import h3_quantization_plan
from h3_tables.source import TARGET_COMPONENT, H3FullTransformer, full_targets
from tensorfs.derived import Part, Source, SourceCapability, SourceInspection, Tensor


class Structures:
    def __init__(self, rows: dict[str, tuple[tuple[str, str], ...]]) -> None:
        self.rows = rows
        self.calls: Counter[str] = Counter()

    def tensorfs_source(self, source: Any) -> SourceCapability:
        def inspect(_components: Any, _configs: Any) -> SourceInspection:
            self.calls[source.checkpoint_ref] += 1
            components: dict[str, dict[str, Tensor]] = {}
            for component, key in self.rows[source.checkpoint_ref]:
                components.setdefault(component, {})[key] = Tensor(
                    "f32", (1,), job.PLAIN_SPEC, {"value": Part("f32", (1,))}
                )
            return SourceInspection(Source(source.checkpoint_ref, 1), components, {})

        return SourceCapability(source.checkpoint_ref, 1, inspect)


def main() -> None:
    module: Any = job
    sections = parse_production_config(job._asset("model-config.json"))
    whole = current_order(job._asset("whole-order.json"))
    full_order = compute_order(sections, whole.rows)
    original = full_targets()
    optional = tuple(
        (component, key) for component, target in original.items() for key in target.drop
    )
    assert len(optional) == 158 and len(full_order) == 3968
    source = H3FullTransformer.for_test(checkpoint_ref="test://complete-full")
    clean = Structures({source.checkpoint_ref: full_order})
    targets = module._select_full_targets(clean, {"dits": source, "shared": source})
    assert clean.calls == {source.checkpoint_ref: 1}
    assert all(not target.drop for target in targets.values())
    assert {component: target.source for component, target in targets.items()} == {
        component: target.source for component, target in original.items()
    }

    dits = H3FullTransformer.for_test(checkpoint_ref="test://native-dits")
    shared = H3FullTransformer.for_test(checkpoint_ref="test://native-shared")
    native = Structures(
        {
            dits.checkpoint_ref: tuple(
                row for row in (*full_order, *optional) if row[0] in TARGET_COMPONENT.values()
            ),
            shared.checkpoint_ref: tuple(
                row for row in (*full_order, *optional) if row[0] not in TARGET_COMPONENT.values()
            ),
        }
    )
    selected = module._select_full_targets(native, {"dits": dits, "shared": shared})
    assert selected == original
    assert native.calls == {dits.checkpoint_ref: 1, shared.checkpoint_ref: 1}

    tables = job._table_additions(sections)
    quantization = prepare_quantization(h3_quantization_plan())

    def pruned(lane_name: str, full_targets: dict[str, Any]) -> dict[str, Any]:
        lane = lanes.LANES[lane_name]
        selections = {
            component: lanes.select(component, treatment, dit_rows, dit_plan=quantization)
            for component, treatment in lane.components.items()
        }
        return job._lane_targets(lane, sections, tables, full_targets, selections)

    dit_rows = SourceInspection(
        Source("sha256:" + "01" * 32, 1),
        {
            component: {
                tensor.key: Tensor(
                    tensor.logical_dtype,
                    tensor.shape,
                    job.PLAIN_SPEC,
                    {"value": Part(tensor.logical_dtype, tensor.shape)},
                )
                for tensor in quantization.tensors
            }
            for component in TARGET_COMPONENT.values()
        },
        {},
    )
    for lane_name in ("bf16-pruned", "fp8-pruned", "mxfp8-pruned"):
        encoding = lanes.LANES[lane_name].components.get("fl2va_dit")
        old = pruned(lane_name, original)
        raw = pruned(lane_name, selected)
        cleaned = pruned(lane_name, targets)
        assert raw == old
        for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
            component = TARGET_COMPONENT[task]
            dynamic = set(removed_keys(H3Topology.from_config(sections[section])))
            assert dynamic <= set(cleaned[component].drop)
            assert set(cleaned[component].drop) == set(old[component].drop) - set(
                original[component].drop
            )
            assert cleaned[component].add == old[component].add
            if encoding is not None:
                assert {row.key for row in quantization.tensors} <= set(cleaned[component].drop)
        assert cleaned["text_encoder"].drop == ()

    # Unknown extras and missing required destinations are not converted to drops.
    changed = Structures({source.checkpoint_ref: (*full_order[1:], ("fl2va_dit", "unexpected"))})
    changed_targets = module._select_full_targets(changed, {"dits": source, "shared": source})
    assert changed_targets == targets
    assert all("unexpected" not in target.drop for target in changed_targets.values())
    # Family policy is ordinary data. The Runtime shared operation enforces these
    # fingerprints against native granted metadata before reading/writing values.
    policy = operations.quantization_plan()
    assert policy.components == tuple(TARGET_COMPONENT.values())
    assert policy.output_precision == "preserve"
    for expected_order in (whole.rows, full_order):
        digest = canonical_json.digest([[component, key] for component, key in expected_order])
        assert digest in policy.source_order_digests
    for invalid_order in (
        tuple(reversed(full_order)),
        full_order[1:],
        (*whole.rows, ("text_encoder", "extra.weight")),
    ):
        digest = canonical_json.digest([[component, key] for component, key in invalid_order])
        assert digest not in policy.source_order_digests

    assert policy.keys == tuple(tensor.key for tensor in quantization.tensors)
    assert len(policy.keys) == 313
    assert all(
        "adaln_proj" not in key and "time_embedder" not in key and not key.startswith("norm_out.")
        for key in policy.keys
    )
    selected = tuple(
        (component, key, tensor)
        for component, rows in dit_rows.components.items()
        for key, tensor in rows.items()
    )

    def identity(tensors: tuple[tuple[str, str, Tensor], ...]) -> str:
        return canonical_json.digest(
            [
                [
                    component,
                    key,
                    tensor.logical_dtype,
                    list(tensor.shape),
                    [[role, part.dtype, list(part.shape)] for role, part in tensor.parts.items()],
                ]
                for component, key, tensor in tensors
            ]
        )

    assert identity(selected) == policy.selected_schema_digest
    component, key, first = selected[0]
    for bad in (
        selected[1:],
        (*selected, (component, "unexpected.weight", first)),
        ((component, key, replace(first, shape=(1, 32))), *selected[1:]),
        ((component, key, replace(first, logical_dtype="f32")), *selected[1:]),
        (
            (component, key, replace(first, parts={"data": Part("f8_e4m3fn", first.shape)})),
            *selected[1:],
        ),
    ):
        assert identity(bad) != policy.selected_schema_digest
    print(
        "H3 full-source targets PASS native declarations unchanged; one read per exact source; "
        "158 already-removed optional rows omitted; required pruning/replacement drops unchanged"
    )


if __name__ == "__main__":
    main()
