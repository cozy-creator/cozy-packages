#!/usr/bin/env python3
"""Measure BF16 LoRA row-partition parity on CUDA, including the former tail path."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3"))

from turbo import LoRAFactors  # noqa: E402


def apply(
    lora: LoRAFactors, x: torch.Tensor, base: torch.Tensor, degree: int, *, legacy: bool
) -> torch.Tensor:
    out = base.clone()
    for source, target in zip(
        torch.tensor_split(x, degree, dim=0), torch.tensor_split(out, degree, dim=0), strict=True
    ):
        if not legacy:
            lora.accumulate(source, target)
            continue
        # Negative control: the previous production path varied both GEMMs' tail M.
        for start in range(0, source.shape[0], 256):
            rows = source[start : start + 256]
            partial = F.linear(rows, lora.lora_down.to(rows.dtype))
            update = F.linear(partial, lora.lora_up.to(rows.dtype)).to(target.dtype)
            target[start : start + rows.shape[0]].add_(lora.scale * update)
    return out


def difference(one: torch.Tensor, two: torch.Tensor) -> dict[str, Any]:
    count, maximum, affected = 0, 0.0, 0
    # Bound comparison temporaries too; the real H3 geometry has 270M outputs.
    for start in range(0, one.shape[0], 1024):
        a, b = one[start : start + 1024], two[start : start + 1024]
        unequal = a != b
        count += int(unequal.sum())
        affected += int(unequal.any(dim=1).sum())
        maximum = max(maximum, float((a.float() - b.float()).abs().max()))
    return {"unequal_elements": count, "affected_rows": affected, "max_abs": maximum}


def case(rows: int, in_features: int, out_features: int) -> dict[str, Any]:
    lora = LoRAFactors(in_features, out_features, 64, 1.0).cuda()
    lora.lora_down.normal_(0, 0.02)
    lora.lora_up.normal_(0, 0.02)
    inputs = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(rows, out_features, device="cuda", dtype=torch.bfloat16) * 0.02
    result: dict[str, Any] = {
        "rows": rows, "in_features": in_features, "out_features": out_features, "rank": 64
    }
    for legacy in (True, False):
        one = apply(lora, inputs, base, 2, legacy=legacy)
        two = apply(lora, inputs, base, 4, legacy=legacy)
        result["legacy" if legacy else "current"] = difference(one, two)
        del one, two
    if result["current"]["unequal_elements"]:
        raise AssertionError(f"current LoRA differs across row partitions: {result}")
    timings: dict[bool, list[float]] = {True: [], False: []}
    for _ in range(3):
        for legacy in (True, False):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = apply(lora, inputs, base, 2, legacy=legacy)
            torch.cuda.synchronize()
            timings[legacy].append((time.perf_counter() - start) * 1000)
            del out
    result["legacy_median_ms"] = statistics.median(timings[True])
    result["current_median_ms"] = statistics.median(timings[False])
    result["relative_time"] = result["current_median_ms"] / result["legacy_median_ms"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-red", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This numerical proof requires a CUDA GPU.")
    torch.manual_seed(7101)
    with torch.no_grad():
        cases = [case(1031, 5376, 5376), case(37712, 5376, 7168)]
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "cases": cases,
        "limits": (
            "Same GPU and tensors, row partition only; not a collective or full H3 clip proof."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_red and not any(row["legacy"]["unequal_elements"] for row in cases):
        raise AssertionError("This device did not reproduce the old tail divergence.")


if __name__ == "__main__":
    main()
