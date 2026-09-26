#!/usr/bin/env python3
"""Exercise fixed references and private AV context through the real Runtime broker.

Synthetic rendering qualifies custody, clocks and errors, not continuity quality.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("h3-cuts-proof.py")
    spec = importlib.util.spec_from_file_location("h3_cuts_proof", path)
    assert spec is not None and spec.loader is not None
    proof = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = proof
    spec.loader.exec_module(proof)
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for frames in (22, 39, 56):
        results[str(frames)] = proof.drive(
            root / str(frames),
            continuous=True,
            context_frames=frames,
            duration_s=10,
        )
        assert results[str(frames)]["frames"] == 720
    proof.drive(root / "standard", continuous=True, turbo=False)
    proof.drive(root / "partial", continuous=True, fail=2)
    proof.drive(root / "cancel", continuous=True, cancel=1)
    proof.drive(root / "first-fails", continuous=True, fail=0)
    proof.drive(root / "reference-fails", continuous=True, reference_failure=True)
    proof.drive(root / "reference-cancels", continuous=True, reference_cancel=True)
    proof.drive(root / "prompt-preflight", continuous=True, prompt="x" * 3500, refuse=True)
    shortened = proof.drive(
        root / "frame-shortening", continuous=True, context_frames=56, duration_s=15
    )
    assert shortened["frames"] == (15 + 12 + 12) * 24
    proof.drive(root / "unknown-ref", continuous=True, bad="unknown", refuse=True)
    (root / "evidence.json").write_text(
        json.dumps(
            {
                "cases": results,
                "actual_h3_inference": False,
                "private_completed_av_context": True,
                "two_public_assets": True,
                "exact_delivered_frames": True,
                "prompt_and_frame_preflight_before_qwen": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("motion context custody, exact clocks, references and interruptions: PASS")


if __name__ == "__main__":
    main()
