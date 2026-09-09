#!/usr/bin/env python3
"""Prove optional source drops and declaration reuse through the author geometry view.

Uses the existing full construction order and optional source-only lists; no second
order file, tensor storage, GPU computation, or claimed numerical evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from cozy_runtime.author import (
    UnsupportedInput,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
)
from cozy_runtime.derive.quantization import h3_quantization_plan, prepare_quantization
from h3_tables import job, lanes, operations
from h3_tables.kernel import H3Topology, removed_keys


class Structures:
    def __init__(self, rows: dict[str, tuple[tuple[str, str], ...]]) -> None:
        self.rows = rows
        self.calls: Counter[str] = Counter()

    def structure(self, source: Any) -> WeightsSource:
        self.calls[source.checkpoint_ref] += 1
        return WeightsSource(
            (),
            tuple(
                WeightsSourceTensor(
                    component, key, "f32", (1,), (WeightsSourcePart("value", "f32", (1,)),)
                )
                for component, key in self.rows[source.checkpoint_ref]
            ),
        )


def main() -> None:
    module: Any = job
    sections = job.parse_production_config(job._asset("model-config.json"))
    whole = job.current_order(job._asset("whole-order.json"))
    full_order = job._full_order(sections, whole.rows)
    original = job._full_targets()
    optional = tuple(
        (component, key) for component, target in original.items() for key in target.drop
    )
    assert len(optional) == 158 and len(full_order) == 3968
    source = job.H3FullTransformer.for_test(checkpoint_ref="test://complete-full")
    clean = Structures({source.checkpoint_ref: full_order})
    targets = module._select_full_targets(clean, {"dits": source, "shared": source})
    assert clean.calls == {source.checkpoint_ref: 1}
    assert all(not target.drop for target in targets.values())
    assert {component: target.source for component, target in targets.items()} == {
        component: target.source for component, target in original.items()
    }

    dits = job.H3FullTransformer.for_test(checkpoint_ref="test://native-dits")
    shared = job.H3FullTransformer.for_test(checkpoint_ref="test://native-shared")
    native = Structures(
        {
            dits.checkpoint_ref: tuple(
                row for row in (*full_order, *optional) if row[0] in job.TARGET_COMPONENT.values()
            ),
            shared.checkpoint_ref: tuple(
                row
                for row in (*full_order, *optional)
                if row[0] not in job.TARGET_COMPONENT.values()
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

    dit_rows = tuple(
        WeightsSourceTensor(
            component,
            tensor.key,
            tensor.logical_dtype,
            tensor.shape,
            (WeightsSourcePart("value", tensor.logical_dtype, tensor.shape),),
        )
        for component in job.TARGET_COMPONENT.values()
        for tensor in quantization.tensors
    )
    for lane_name in ("bf16-adaln-pruned", "fp8-adaln-pruned", "mxfp8-adaln-pruned"):
        encoding = lanes.LANES[lane_name].components.get("fl2va_dit")
        old = pruned(lane_name, original)
        raw = pruned(lane_name, selected)
        cleaned = pruned(lane_name, targets)
        assert raw == old
        for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
            component = job.TARGET_COMPONENT[task]
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
    for expected_order in (whole.rows, full_order):
        rows = Structures({source.checkpoint_ref: tuple(reversed(expected_order))})
        assert operations._quantization_order(rows.structure(source)) == expected_order
    for invalid_order in (full_order[1:], (*whole.rows, ("text_encoder", "extra.weight"))):
        rows = Structures({source.checkpoint_ref: invalid_order})
        try:
            operations._quantization_order(rows.structure(source))
        except UnsupportedInput as error:
            assert error.code == "quantization_source"
        else:
            raise AssertionError("incomplete or foreign H3 construction roster accepted")

    # The memoized export must select the exact current 313 weights in each task.
    # These are real geometry values; this arm makes no byte custody/GPU claim.
    selected = tuple(
        WeightsSourceTensor(component, tensor.key, tensor.logical_dtype, tensor.shape,
                            (WeightsSourcePart("value", tensor.logical_dtype, tensor.shape),))
        for component in job.TARGET_COMPONENT.values()
        for tensor in quantization.tensors
    )
    selection = operations._quantization_plan(WeightsSource((), selected))
    assert len(selection.tensors) == 313
    assert all("adaln_proj" not in tensor.key and "time_embedder" not in tensor.key
               and not tensor.key.startswith("norm_out.") for tensor in selection.tensors)
    for bad in (
        selected[1:],
        (replace(selected[0], shape=(1, 32)), *selected[1:]),
        (replace(selected[0], logical_dtype="f32"), *selected[1:]),
        (replace(selected[0], parts=(WeightsSourcePart("data", "f8_e4m3fn", selected[0].shape),)),
         *selected[1:]),
    ):
        try:
            operations._quantization_plan(WeightsSource((), bad))
        except UnsupportedInput as error:
            assert error.code == "quantization_source"
        else:
            raise AssertionError("an incompatible quantization source was accepted")
    print(
        "H3 full-source targets PASS native declarations unchanged; one read per exact source; "
        "158 already-removed optional rows omitted; required pruning/replacement drops unchanged"
    )


if __name__ == "__main__":
    main()
