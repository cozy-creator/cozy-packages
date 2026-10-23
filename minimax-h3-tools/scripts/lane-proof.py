#!/usr/bin/env python3
"""The H3 lanes and quantize recipes resolved against real cozytensors headers, on the CPU.

`--pruned` is a released AdaLN-pruned bf16 lane's header and `--full` a bf16-full lane's, both
with D4's component names. Everything that decides an output's tensors runs on them: the rows a
lane copies and computes, every treatment's selection, the recipes' per-component counts (the
production lanes': 313/313/350, int8 also the video VAE decoder's 216, nvfp4 300/300 plus the
conditioner at int8.convrot) and the
refusals. No weight byte is read.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import tensorfs
from cozy_runtime.author import UnsupportedInput, canonical_json
from h3_tables import adaln_operations, job, lanes
from h3_tables.model_config import COMPONENTS, DIT_COMPONENT, parse_full_config
from h3_tables.order import current_order, full_order
from h3_tables.plans import Task, TimestepPlan
from h3_tables.table_layout import TableLayout
from h3_tables.source import rows

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "scripts" / "lane-headers"
def plans() -> dict[Task, TimestepPlan]:
    """Geometry from captured pruned headers, without requiring a serving environment.

    This fixture deliberately carries no schedule; model-assets-proof.py separately
    checks complete source plans against the independent serving scheduler oracle.
    """
    source = tensorfs.open(VECTORS / "minimax-h3.cozytensors", local_files_only=True)
    result = {}
    for task, component in DIT_COMPONENT.items():
        table_keys = source.components[component].config["cozy_h3"]["table_keys"]
        layout = TableLayout.parse(table_keys)
        result[task] = TimestepPlan(
            task, "", canonical_json.encode({"table_keys": table_keys}),
            layout.timesteps, layout.block_keys, (),
        )
    return result


def refuses(code: str, call: Callable[[], object]) -> None:
    try:
        call()
    except UnsupportedInput as exc:
        assert exc.code == code, f"refused {exc.code}, expected {code}: {exc}"
        return
    raise AssertionError(f"expected the refusal {code}")


def counts(selections: list[lanes.Selection]) -> dict[str, tuple[str | None, int]]:
    return {s.component: (s.treatment.encode, len(s.encoded)) for s in selections}


def recipes(pruned: tensorfs.Checkpoint) -> None:
    dits = {"transformer", "transformer_ref"}
    for encoding, dit_scheme, dit_rows, te_scheme in (
        ("fp8.rowwise", "fp8.rowwise", 313, "fp8.rowwise"),
        ("int8.convrot", "int8.convrot", 313, "int8.convrot"),
        ("nvfp4", "nvfp4", 300, "int8.convrot"),
    ):
        got = counts(job.quantizer_selections(pruned, encoding))
        want = {dit: (dit_scheme, dit_rows) for dit in dits} | {"text_encoder": (te_scheme, 350)}
        if encoding == "int8.convrot":
            want["vae"] = ("int8.convrot", 216)
        assert got == want, f"{encoding}: {got}"
        if encoding == "nvfp4":
            for selection in job.quantizer_selections(pruned, encoding):
                if selection.component in dits:
                    own = {t.key for t in selection.plan.tensors if t.encoding == "int8.convrot"}
                    assert own == set(lanes.NVFP4_DIT_INT8), own
        print(f"PASS --{encoding}: {got}")
    present = rows(pruned)
    refuses("h3_component_refused", lambda: lanes.select(
        "audio_vae", lanes.Treatment(encode="fp8.rowwise"), pruned, present["audio_vae"]))
    refuses("h3_keep_unmatched", lambda: lanes.select(
        "text_encoder", lanes.Treatment(encode="fp8.rowwise", keep=("model.no_such.weight",)),
        pruned, present["text_encoder"]))
    refuses("h3_lane_source", lambda: job.lane_rows("bf16-full", present, plans()))
    print("PASS refusals: audio_vae, an unmatched keep, a pruned source for a lane")


def lanes_from_full(full: tensorfs.Checkpoint) -> None:
    present = rows(full)
    sections = parse_full_config(job._asset("model-config.json"))
    current = current_order(job._asset("whole-order.json")).rows
    for name, order_want, computed_want in (
        ("bf16-full", full_order(sections, current), 0),
        ("bf16", current, 102),
        ("fp8", current, 102),
    ):
        order, carried, computed = job.lane_rows(name, present, plans())
        assert order == tuple(order_want) and len(computed) == computed_want, name
        assert set(carried) == set(COMPONENTS), name
        assert all(row in computed or row[1] in carried[row[0]] for row in order), name
        selections = [
            lanes.select(c, t, full, carried[c], dit_plan=job.dit_plan(),
                         allow_inert=c in lanes.NORMALISED_COMPONENTS)
            for c, t in lanes.lane_treatments(job.LANES[name]).items()
        ]  # fmt: skip
        encoded = {s.component: len(s.encoded) for s in selections if s.encoded}
        cast = {s.component: len(s.cast) for s in selections}
        sizes = {c: len(rows) for c, rows in carried.items()}
        print(f"PASS {name}: {len(order)} rows ({len(computed)} computed), carried {sizes}, "
              f"encoded {encoded}, cast {cast}")  # fmt: skip
        if name == "fp8":
            assert encoded == {"transformer": 313, "transformer_ref": 313, "text_encoder": 350}
    refuses("h3_quantize_unpruned", lambda: job.quantizer_selections(full, "fp8.rowwise"))
    print("PASS a full source is refused by the quantize recipes")


def adaln(pruned: tensorfs.Checkpoint, full: tensorfs.Checkpoint) -> None:
    tables = job.retable_rows(rows(pruned), rows(full), plans())
    assert {c: len(keys) for c, keys in tables.items()} == {c: 51 for c in DIT_COMPONENT.values()}
    assert adaln_operations._body_kind(rows(pruned)) is True
    assert adaln_operations._body_kind(rows(full)) is False
    print("PASS retable reads 51 tables per DiT; the AdaLN operations tell full from pruned")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruned", type=Path, default=VECTORS / "minimax-h3.cozytensors")
    parser.add_argument("--full", type=Path, default=VECTORS / "minimax-h3-full.cozytensors")
    args = parser.parse_args()
    pruned, full = (tensorfs.open(path, local_files_only=True) for path in (args.pruned, args.full))
    recipes(pruned)
    lanes_from_full(full)
    adaln(pruned, full)
    print("lane-proof green")


if __name__ == "__main__":
    main()
