#!/usr/bin/env python3
"""A package's declared recipes resolve against real cozytensors/1 headers, in its own environment:

    sdxl/.venv/bin/python scripts/lane-plans-proof.py sdxl

Each header in `lane-headers/` is a family's source lane as its Hub holds it, header only
(`lane-headers/fetch.py`), so its names and geometry are the converted ones (D4: SDXL's
`text_model.`, H3's Diffusers components). The recipes are read as the static describe records
them; Runtime's own choice (`derive.recipes.choose`) must pick each for its lane from that header,
and Runtime's own selection must take exactly the tensors the published lane encodes.
"""

from __future__ import annotations

import fnmatch
import importlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import msgspec
import tensorfs
from cozy_runtime.author import QuantizationPlan, SelectionPlan
from cozy_runtime.derive.operations import _selection
from cozy_runtime.derive.recipes import choose
from cozy_runtime.internal import package_interface, static_interface

ROOT = Path(__file__).resolve().parent.parent

#: package -> (header, recipe, tensors per scheme): the recipe Runtime picks for its lane from that
#: header. A set of components is a module constant naming the components a header must hold.
LANES: dict[str, list[tuple[str, str, dict[str, int]]]] = {
    "sdxl": [
        ("sdxl", "fp8", {"fp8.rowwise": 743}),
        ("sdxl", "int8", {"int8.convrot": 906}),
    ],
    "anima": [
        ("anima", "fp8", {"fp8.rowwise": 453}),
        ("anima", "int8", {"int8.convrot": 476}),
    ],
    "seedvr2": [("seedvr2", "int8", {"int8.convrot": 288})],
    "qwen-image-2": [("qwen-image-2", "int8", {"int8.convrot": 476})],
    "character-describer": [("qwen3.5-9b", "int8", {"int8.convrot": 200})],
    "flux-3-klein": [
        ("flux-3-klein", "int8", {"int8.convrot": 490}),
        ("flux-3-klein", "nvfp4", {"int8.convrot": 232, "nvfp4": 258}),
        ("flux-3-klein-upsampler", "upsampler_int8", {"int8.convrot": 201}),
        ("flux-3-klein-upsampler", "upsampler_nvfp4", {"nvfp4": 284}),
        ("qwen3.5-9b", "drafter_head", {"": 15}),
    ],
}
#: package -> (module, header, constant): component sets a header must hold.
COMPONENTS = {
    "minimax-h3": [
        ("minimax_h3.official", "minimax-h3", "_WEIGHTED_CONFIG_SECTIONS"),
        ("minimax_h3.official", "minimax-h3-full", "_WEIGHTED_CONFIG_SECTIONS"),
    ],
}


def header(name: str) -> tensorfs.Checkpoint:
    path = ROOT / "scripts" / "lane-headers" / f"{name}.cozytensors"
    return tensorfs.open(path, local_files_only=True)


def plan(row: Any) -> QuantizationPlan | SelectionPlan:
    """A row's plan as the interface records it: nothing of the package is imported."""
    return msgspec.convert(row.plan, QuantizationPlan | SelectionPlan)


def count(ck: tensorfs.Checkpoint, chosen: QuantizationPlan | SelectionPlan) -> Counter[str]:
    if isinstance(chosen, QuantizationPlan):
        return Counter(t.encoding or chosen.encoding or "" for t in _selection(ck, chosen).tensors)
    names = [f"{c}.{k}" for c, component in ck.components.items() for k in component.tensors]
    return Counter({"": sum(any(fnmatch.fnmatchcase(n, r) for r in chosen.include) for n in names)})


def main(package: str) -> None:
    if package not in LANES and package not in COMPONENTS:
        print(f"{package}: no lane plans")
        return
    described = static_interface.build(ROOT / package)
    recipes = package_interface.parse(package_interface.canonical_bytes(described)).recipes
    rows = {row.name: row for row in recipes}
    for name, recipe, want in LANES.get(package, []):
        ck = header(name)
        row, chosen = choose(recipes, ck, lane=rows[recipe].lane, recipe="", package=package, plan=plan)
        assert row.name == recipe, f"{package} {row.lane} on {name} chose {row.name}, not {recipe}"
        got = count(ck, chosen)
        assert got == Counter(want), f"{package} {recipe} on {name}: {dict(got)} != {want}"
        print(f"{package} {recipe} on {name}: {dict(got)}")
    for module, name, constant in COMPONENTS.get(package, []):
        sys.path.insert(0, str(ROOT / package))
        held = getattr(importlib.import_module(module), constant)
        missing = held - set(header(name).components)
        assert not missing, f"{name} lacks components {sorted(missing)}"
        print(f"{package} {constant} on {name}: {len(held)} components")
    print(f"{package} lane plans PASS")


if __name__ == "__main__":
    main(sys.argv[1])
