#!/usr/bin/env python3
"""Prove each committed execution contract asset against cozy-runtime's own reader.

The lane contracts of attention-quantization.md §2 (decision #707) and h3a-012 are typed out
here and the package assets must be their canonical bytes exactly — the bytes `attention-lane`
writes into the produced header. cozy-runtime's `execution_contract` reader, the one that
applies a contract or refuses typed (cr-109/cr-114), must then admit each asset at the same
document and digest as the design read independently.

This runs in the repo's light check environment, at the CURRENT Runtime, because the contract
schema is Runtime's and a package that writes it must be read by the release that reads it.
`h3-attention-lane-proof.py` proves what the job does with the same asset bytes, in the
package's own locked environment; the two meet on the file this one validates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cozy_runtime.author import canonical_json
from cozy_runtime.internal.execution_contract import read

ASSETS = Path(__file__).resolve().parent.parent / "minimax-h3-tools/src/h3_tables/assets"

#: attention-quantization.md §2 (decision #707) and h3a-012's same-class kernel, verbatim.
DESIGN: dict[str, dict[str, Any]] = {
    "sm90-attn8": {
        "device": "sm90",
        "activations": "bf16",
        "class": "quantized",
        "weights": {"route": "encoded_gemm"},
        "attention": {
            "distribution": "sageattention",
            "version": "2.2.0",
            "entry": "sageattn_qk_int8_pv_fp8_cuda_sm90",
            "kwargs": {
                "tensor_layout": "NHD",
                "qk_quant_gran": "per_thread",
                "pv_accum_dtype": "fp32+fp32",
                "smooth_k": True,
                "is_causal": False,
            },
            "scheme": (
                "qk int8 per-thread (Q64/16, K128/128) K-mean-smoothed; v fp8-e4m3 "
                "per-channel; pv fp32+fp32"
            ),
        },
    },
    "sm90-fa3": {
        "device": "sm90",
        "activations": "bf16",
        "class": "same",
        "weights": {"route": "encoded_gemm"},
        "attention": {
            "distribution": "flash-attn3",
            "version": "1",
            "revision": "7cb368cf8278b583132eb72cbf312d54586df2e2",
            "variant": "torch-stable-abi29-cu130-x86_64-linux",
            "entry": "flash_attn_func",
            "kwargs": {
                "causal": False,
                "q_descale": None,
                "k_descale": None,
                "v_descale": None,
                "num_splits": 1,
                "deterministic": False,
            },
            "scheme": (
                "bf16 q/k/v with no descale (the bf16 path, not fp8); fp32 softmax and PV "
                "accumulation; full non-causal attention, no window, no softcap"
            ),
        },
    },
}


def main() -> None:
    proven: dict[str, dict[str, str]] = {}
    for name, document in DESIGN.items():
        asset = (ASSETS / f"execution.{name}.json").read_bytes()
        if asset != canonical_json.encode(document):
            raise AssertionError(f"execution.{name}.json is not the design document's bytes")
        canonical = canonical_json.encode(document)
        contract, expected = read({"execution": asset}), read({"execution": canonical})
        assert contract is not None and expected is not None
        assert contract.document() == expected.document() == document
        assert contract.digest() == expected.digest()
        proven[name] = {
            "config_digest": canonical_json.digest_bytes(asset),
            "contract_digest": contract.digest(),
            "class": contract.class_,
            "route": contract.weights.route,
            "entry": f"{contract.attention.distribution}.{contract.attention.entry}",
        }
    print("H3 execution contracts PASS " + json.dumps(proven, sort_keys=True))


if __name__ == "__main__":
    main()
