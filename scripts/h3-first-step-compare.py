"""Compare retained first-step traces without treating sampled equality as complete proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    for key in ("format", "prompt", "seed", "duration_s", "base_checkpoint", "adapter_checkpoint"):
        if left[key] != right[key]:
            raise ValueError(f"incomparable traces: {key} differs")
    if left["format"] != "h3.first-step-boundaries/1":
        raise ValueError("unsupported trace format")
    if left["completed_steps"] != 1 or right["completed_steps"] != 1:
        raise ValueError("both traces must have completed exactly one scheduler step")
    left_records, right_records = left["records"], right["records"]

    def coordinates(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
        return [(row["forward"], row["name"]) for row in rows]

    if coordinates(left_records) != coordinates(right_records):
        raise ValueError("trace boundaries or their order differ")
    boundaries = []
    for a, b in zip(left_records, right_records, strict=True):
        for key in ("dtype", "coverage", "observed_shape", "sample_flat_indices"):
            if a[key] != b[key]:
                raise ValueError(f"incomparable boundary {a['name']}: {key} differs")
        boundaries.append(
            {
                "forward": a["forward"],
                "name": a["name"],
                "coverage": a["coverage"],
                "equal": a["sha256"] == b["sha256"],
                "sample_max_abs_error": max(
                    (
                        abs(x - y)
                        for x, y in zip(a["sample_values"], b["sample_values"], strict=True)
                    ),
                    default=0.0,
                ),
            }
        )
    return {
        "left_request": left["request_id"],
        "right_request": right["request_id"],
        "all_observed_boundaries_equal": all(row["equal"] for row in boundaries),
        "first_observed_difference": next((row for row in boundaries if not row["equal"]), None),
        "coverage": "Observed boundaries only; the first differing operation is not proven.",
        "boundaries": boundaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    print(
        json.dumps(
            [
                {"candidate": str(path), **compare(reference, json.loads(path.read_text()))}
                for path in args.candidates
            ],
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
