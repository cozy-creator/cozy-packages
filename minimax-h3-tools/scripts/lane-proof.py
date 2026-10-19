#!/usr/bin/env python3
"""The H3 lanes and quantize recipes resolved against real cozytensors/1 headers, on the CPU.

`--pruned` is a released AdaLN-pruned bf16 lane's header and `--full` a bf16-full lane's, both
with D4's component names. Everything that decides an output's tensors runs on them: the rows a
lane copies and computes, every treatment's selection, the recipes' per-component counts (the
production lanes': 313/313/350, nvfp4 276/276 plus the conditioner at int8.convrot) and the
refusals. No weight byte is read.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tensorfs
from cozy_runtime.author import UnsupportedInput
from h3_tables import adaln_operations, job, lanes
from h3_tables.model_config import COMPONENTS, DIT_COMPONENT, parse_production_config
from h3_tables.order import current_order, full_order

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "scripts" / "lane-headers"


def source(path: Path) -> Any:
    """A granted checkpoint as the producer reads it: its parsed header."""
    return SimpleNamespace(checkpoint=tensorfs.Checkpoint.parse(path.read_bytes()), manifest="")


def refuses(code: str, call: Callable[[], object]) -> None:
    try:
        call()
    except UnsupportedInput as exc:
        assert exc.code == code, f"refused {exc.code}, expected {code}: {exc}"
        return
    raise AssertionError(f"expected the refusal {code}")


def counts(selections: list[lanes.Selection]) -> dict[str, tuple[str | None, int]]:
    return {s.component: (s.treatment.encode, len(s.encoded)) for s in selections}


def recipes(pruned: Any) -> None:
    dits = {"transformer", "transformer_ref"}
    for encoding, dit_scheme, dit_rows, te_scheme in (
        ("fp8.rowwise", "fp8.rowwise", 313, "fp8.rowwise"),
        ("int8.convrot", "int8.convrot", 313, "int8.convrot"),
        ("nvfp4", "nvfp4", 276, "int8.convrot"),
    ):
        got = counts(job.quantizer_selections(pruned, encoding))
        want = {dit: (dit_scheme, dit_rows) for dit in dits} | {"text_encoder": (te_scheme, 350)}
        assert got == want, f"{encoding}: {got}"
        print(f"PASS --{encoding}: {got}")
    present = pruned.checkpoint.tensors
    refuses("h3_component_refused", lambda: lanes.select(
        "audio_vae", lanes.Treatment(encode="fp8.rowwise"), pruned, present["audio_vae"]))
    refuses("h3_keep_unmatched", lambda: lanes.select(
        "text_encoder", lanes.Treatment(encode="fp8.rowwise", keep=("model.no_such.weight",)),
        pruned, present["text_encoder"]))
    refuses("h3_lane_source", lambda: job.lane_rows("bf16-full", present))
    print("PASS refusals: audio_vae, an unmatched keep, a pruned source for a lane")


def lanes_from_full(full: Any) -> None:
    present = full.checkpoint.tensors
    sections = parse_production_config(job._asset("model-config.json"))
    current = current_order(job._asset("whole-order.json")).rows
    for name, order_want, computed_want in (
        ("bf16-full", full_order(sections, current), 0),
        ("bf16", current, 102),
        ("fp8", current, 102),
    ):
        order, carried, computed = job.lane_rows(name, present)
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


def adaln(pruned: Any, full: Any) -> None:
    tables = job.retable_rows(pruned.checkpoint.tensors, full.checkpoint.tensors)
    assert {c: len(keys) for c, keys in tables.items()} == {c: 51 for c in DIT_COMPONENT.values()}
    assert adaln_operations._body_kind(pruned.checkpoint.tensors) is True
    assert adaln_operations._body_kind(full.checkpoint.tensors) is False
    print("PASS retable reads 51 tables per DiT; the AdaLN operations tell full from pruned")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruned", type=Path, default=VECTORS / "minimax-h3.cozytensors")
    parser.add_argument("--full", type=Path, default=VECTORS / "minimax-h3-full.cozytensors")
    args = parser.parse_args()
    pruned, full = source(args.pruned), source(args.full)
    recipes(pruned)
    lanes_from_full(full)
    adaln(pruned, full)
    print("lane-proof green")


if __name__ == "__main__":
    main()
