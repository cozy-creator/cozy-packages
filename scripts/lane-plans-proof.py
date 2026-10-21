#!/usr/bin/env python3
"""A package's lane plans resolve against real cozytensors/1 headers, in its own environment:

    sdxl/.venv/bin/python scripts/lane-plans-proof.py sdxl

Each header in `lane-headers/` is a family's source lane as its Hub holds it, header only
(`lane-headers/fetch.py`), so its names and geometry are the converted ones (D4: SDXL's
`text_model.`, H3's Diffusers components). Every plan goes through Runtime's own selection and
must take exactly the tensors the published lane encodes.
"""

from __future__ import annotations

import fnmatch
import importlib
import sys
from collections import Counter
from pathlib import Path

import tensorfs
from cozy_runtime.derive.operations import QuantizationPlan, SelectionPlan, _selection
from cozy_runtime.derive.quantization import column_block, prepare_source_quantization

ROOT = Path(__file__).resolve().parent.parent

#: package -> module, then (header, plan attribute, lane encoding, tensors per scheme). A tuple
#: plan is a whole-component lane, a SelectionPlan keeps tensors, a set names components.
LANES = {
    "sdxl": ("sdxl", [
        ("sdxl", "FP8", "fp8.rowwise", {"fp8.rowwise": 743}),
        ("sdxl", "INT8", "int8.convrot", {"int8.convrot": 906}),
    ]),
    "anima": ("anima", [
        ("anima", "FP8", "fp8.rowwise", {"fp8.rowwise": 453}),
        ("anima", "INT8", "int8.convrot", {"int8.convrot": 476}),
    ]),
    "seedvr2": ("seedvr2.quantize", [("seedvr2", "INT8", "int8.convrot", {"int8.convrot": 288})]),
    "qwen-image-2": ("qwen_image_2.quantize", [
        ("qwen-image-2", "INT8", "int8.convrot", {"int8.convrot": 476}),
    ]),
    "character-describer": ("character_describer", [
        ("qwen3.5-9b", "INT8", "int8.convrot", {"int8.convrot": 200}),
    ]),
    "flux-3-klein": ("flux_3_klein.quantize", [
        ("flux-3-klein", "INT8", "int8.convrot", {"int8.convrot": 490}),
        ("flux-3-klein", "NVFP4", "int8.convrot", {"int8.convrot": 232, "nvfp4": 258}),
        ("flux-3-klein-upsampler", "UPSAMPLER_INT8", "int8.convrot", {"int8.convrot": 201}),
        ("flux-3-klein-upsampler", "UPSAMPLER_NVFP4", "nvfp4", {"nvfp4": 284}),
        ("qwen3.5-9b", "DRAFTER_HEAD", "", {"": 15}),
    ]),
    "minimax-h3": ("minimax_h3.official", [
        ("minimax-h3", "_WEIGHTED_CONFIG_SECTIONS", "", {"": 5}),
        ("minimax-h3-full", "_WEIGHTED_CONFIG_SECTIONS", "", {"": 5}),
    ]),
}


def resolve(header: str, plan: object, encoding: str) -> Counter[str]:
    ck = tensorfs.open(ROOT / "scripts" / "lane-headers" / f"{header}.cozytensors", local_files_only=True)
    if isinstance(plan, QuantizationPlan):
        return Counter(t.encoding or encoding for t in _selection(ck, plan, encoding).tensors)
    if isinstance(plan, tuple):
        chosen = prepare_source_quantization(ck, components=plan, block=column_block(encoding))
        return Counter({encoding: len(chosen.tensors)})
    if isinstance(plan, SelectionPlan):
        names = [f"{c}.{k}" for c, component in ck.components.items() for k in component.tensors]
        return Counter({"": sum(any(fnmatch.fnmatchcase(n, r) for r in plan.include) for n in names)})
    assert isinstance(plan, set), plan
    missing = plan - set(ck.components)
    assert not missing, f"{header} lacks components {sorted(missing)} (has {sorted(ck.components)})"
    return Counter({"": len(plan)})


def main(package: str) -> None:
    if package not in LANES:
        print(f"{package}: no lane plans")
        return
    sys.path.insert(0, str(ROOT / package))
    module, lanes = LANES[package]
    plans = importlib.import_module(module)
    for header, name, encoding, want in lanes:
        got = resolve(header, getattr(plans, name), encoding)
        assert got == Counter(want), f"{package} {name} on {header}: {dict(got)} != {want}"
        print(f"{package} {name} on {header}: {dict(got)}")
    print(f"{package} lane plans PASS")


if __name__ == "__main__":
    main(sys.argv[1])
