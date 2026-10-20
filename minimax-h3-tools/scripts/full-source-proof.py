#!/usr/bin/env python3
"""Prove optional source drops and declaration reuse through the author geometry view.

Uses the existing full construction order and optional source-only lists; no second
order file, tensor storage, GPU computation, or claimed numerical evidence. Rows a
treatment selects by shape carry their real geometry (`source_geometry.py`), so every lane
resolves its treatments over each granted source exactly as `_produce_lanes` does.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from typing import Any

from cozy_runtime.author import Weights, canonical_json
from cozy_runtime.derive.quantization import (
    prepare_quantization,
)
from h3_tables import job, lanes, operations
from h3_tables.kernel import H3Topology, removed_keys
from h3_tables.model_config import parse_production_config
from h3_tables.order import current_order
from h3_tables.order import full_order as compute_order
from h3_tables.quantization import h3_quantization_plan
from h3_tables.source import TARGET_COMPONENT, full_targets
from source_geometry import VIDEO_VAE_ROWS, conditioner_rows, plain
from tensorfs.derived import Part, Source, SourceCapability, SourceInspection, Tensor

GEOMETRY = {("text_encoder", key): spec for key, spec in conditioner_rows().items()} | {
    ("vae", key): spec for key, spec in VIDEO_VAE_ROWS.items()
}


def weights(name: str) -> Weights[Any]:
    return Weights("sha256:" + hashlib.sha256(name.encode()).hexdigest())


class Structures:
    def __init__(self, rows: dict[str, tuple[tuple[str, str], ...]]) -> None:
        self.rows = rows
        self.calls: Counter[str] = Counter()

    def tensorfs_source(self, source: Any) -> SourceCapability:
        def inspect(_components: Any, _configs: Any) -> SourceInspection:
            self.calls[source.checkpoint_ref] += 1
            components: dict[str, dict[str, Tensor]] = {}
            for component, key in self.rows[source.checkpoint_ref]:
                spec = GEOMETRY.get((component, key), ("f32", (1,)))
                components.setdefault(component, {})[key] = plain(*spec)
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
    source = weights("complete-full")
    sources = {"dits": source, "shared": source}
    clean = Structures({source.checkpoint_ref: full_order})
    clean_granted = module._structures(clean, sources)
    targets = module._select_full_targets(clean, sources, clean_granted)
    assert clean.calls == {source.checkpoint_ref: 1}
    assert all(not target.drop for target in targets.values())
    assert {component: target.source for component, target in targets.items()} == {
        component: target.source for component, target in original.items()
    }

    dits = weights("native-dits")
    shared = weights("native-shared")
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
    native_sources = {"dits": dits, "shared": shared}
    native_granted = module._structures(native, native_sources)
    selected = module._select_full_targets(native, native_sources, native_granted)
    assert selected == original
    assert native.calls == {dits.checkpoint_ref: 1, shared.checkpoint_ref: 1}

    tables = job._table_additions(sections)
    quantization = prepare_quantization(h3_quantization_plan())

    def declared(
        lane_name: str, full_targets: dict[str, Any], granted: dict[str, SourceInspection]
    ) -> dict[str, Any]:
        """The lane's targets as `_produce_lanes` declares them over this granted source."""
        lane = lanes.LANES[lane_name]
        selections = {
            component: lanes.select(
                component,
                treatment,
                lanes.carried(full_targets[component], granted[full_targets[component].source]),
                dit_plan=quantization,
                allow_inert=component in lanes.NORMALISED_COMPONENTS,
            )
            for component, treatment in lanes.lane_treatments(lane).items()
        }
        return job._lane_targets(lane, sections, tables, full_targets, selections)

    for lane_name, lane in lanes.LANES.items():
        old = declared(lane_name, original, native_granted)
        raw = declared(lane_name, selected, native_granted)
        cleaned = declared(lane_name, targets, clean_granted)
        assert raw == old
        for component in lanes.COMPONENTS:
            assert set(cleaned[component].drop) == set(old[component].drop) - set(
                original[component].drop
            ), (lane_name, component)
            assert cleaned[component].add == old[component].add, (lane_name, component)
        for task, section in (("fl2va", "transformer"), ("ref2va", "transformer_ref")):
            component = TARGET_COMPONENT[task]
            if lane.modulation == "adaln-pruned":
                dynamic = set(removed_keys(H3Topology.from_config(sections[section])))
                assert dynamic <= set(cleaned[component].drop)
            if component in lane.components:
                assert {row.key for row in quantization.tensors} <= set(cleaned[component].drop)
        # A clean conditioner drops exactly what the lane encodes: h3a-028's 350 decoder
        # linears, never a source-only row (lm_head among them) it no longer carries.
        conditioner = cleaned["text_encoder"]
        assert set(conditioner.drop) == set(conditioner.add)
        assert len(conditioner.add) == (350 if "text_encoder" in lane.components else 0)
        assert set(cleaned["vae"].drop) == {
            "decoder.proj_in.weight",
            "decoder.transformer_blocks.0.attn.to_q.weight",
            "post_quant_conv.weight",
        }

    # Unknown extras and missing required destinations are not converted to drops.
    changed = Structures({source.checkpoint_ref: (*full_order[1:], ("transformer", "unexpected"))})
    changed_targets = module._select_full_targets(changed, sources)
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
        (component, tensor.key, plain(tensor.logical_dtype, tensor.shape))
        for component in TARGET_COMPONENT.values()
        for tensor in quantization.tensors
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
