#!/usr/bin/env python
"""THE KEY-EXACTNESS PROOF — the H3 graph this endpoint constructs against the artifact's
own header, decided on the control plane with ZERO weight bytes.

An endpoint's construction names destinations; the fill plane matches them against the
artifact's topology by exact key. So the question "will this endpoint serve the job-001
dual snapshot" has a component-by-component answer that costs nothing to ask: build every
component on `meta`, walk it the way `cozy_runtime.author.census` walks it, and diff the
(key, shape, dtype) table against the pinned safetensors header of the carrier the recipe
declares. A missing key, an extra key, a transposed shape or a wrong dtype is a failed
serve on a rented card; here it is a diff on this box.

    nice -n 19 .venv/bin/python scripts/h3-keys.py [component ...]

THE SOURCE IS DECLARED, and unreadable is a REFUSAL, never a skip: the pinned evidence bank
`~/cozy_v2/h3-evidence` (tfs-010) carries the exact header bytes of
`Comfy-Org/MiniMax-H3@4cc1d817…`, which is proto-001's selection `D-comfy-curve-fp8` and
job-001's recipe. If the bank is not there this script must not print a verdict.

Two carriers per transformer, on purpose. The recipe's SERVED carrier is the fp8 one, whose
header additionally carries 550 encoding-ROLE siblings (`.weight_scale`, `.input_scale`,
`.comfy_quant`) that are parts of a destination and not destinations. The LOGICAL table —
532 keys — is the fp8 header with those roles folded out, and its dtypes are the bf16
sibling's, because a weight-only fp8 encoding decodes to the destination's compute dtype at
fill. Both facts are asserted here rather than assumed.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys
from typing import Any, NoReturn

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
BANK = pathlib.Path.home() / "cozy_v2" / "h3-evidence"
ROW = "comfy-org-h3.json"

#: The recipe's five components (job-001), each as (served carrier, dtype carrier). The
#: two differ only for an ENCODED component: fp8 stores fp8 and fills bf16.
CARRIERS: dict[str, tuple[str, str]] = {
    "transformer": (
        "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
    ),
    "transformer_ref": (
        "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
    ),
    "text_encoder": (
        "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
    ),
    "video_vae": (
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    "audio_vae": (
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
}

#: Sibling suffixes that are ENCODING ROLES of a destination, not destinations. The name
#: set is the border's (cozytensors' registry); it is spelled here because this script
#: reads a raw upstream header, which has no border in it.
ENCODING_ROLES = (".weight_scale", ".input_scale", ".comfy_quant")

DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

Table = dict[str, tuple[tuple[int, ...], torch.dtype]]


def refuse(message: str) -> NoReturn:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)


def header(path: str) -> dict[str, Any]:
    """The pinned header bytes for one banked file. Bank absent or row absent REFUSES."""
    row_file = BANK / "rows" / ROW
    if not row_file.is_file():
        refuse(
            f"the declared evidence source {row_file} is not readable — this proof cites "
            "pinned upstream headers and cannot be run without them"
        )
    row: dict[str, Any] = json.loads(row_file.read_bytes())
    banked = [f for f in row["banked_files"] if f["path"] == path]
    if not banked:
        refuse(f"{path} is not banked in {ROW}")
    digest = banked[0]["fetch"]["bytes_sha256"]
    cached = BANK / "cache" / digest
    if not cached.is_file():
        refuse(f"the cached header bytes {digest[:16]}… for {path} are missing from the bank")
    raw = cached.read_bytes()
    import hashlib

    if hashlib.sha256(raw).hexdigest() != digest:
        refuse(f"the cached header bytes for {path} do not match their banked digest")
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


def pinned(component: str) -> Table:
    """The artifact's own destination table: keys and shapes from the SERVED carrier,
    dtypes from the carrier whose stored dtype is the fill dtype."""
    served, dtyped = CARRIERS[component]
    head = header(served)
    dtype_head = head if served == dtyped else header(dtyped)
    logical = {
        k: v
        for k, v in head.items()
        if k != "__metadata__" and not k.endswith(ENCODING_ROLES)
    }
    table: Table = {}
    for key, spec in logical.items():
        ref = dtype_head.get(key)
        if ref is None:
            refuse(f"{component}: {key} is in the served carrier and not in the dtype carrier")
        table[key] = (tuple(spec["shape"]), DTYPES[ref["dtype"]])
    return table


def constructed(component: str) -> Table:
    """The graph the endpoint builds, censused the way the runtime censuses it: one
    `state_dict()` walk of the component root, on `meta`, so no byte is allocated."""
    sys.path.insert(0, str(ROOT / "h3"))
    from h3_arch import build_component

    with torch.device("meta"):
        module = build_component(component)
    return {
        key: (tuple(tensor.shape), tensor.dtype)
        for key, tensor in module.state_dict().items()
    }


def summarize(keys: list[str], limit: int = 8) -> list[str]:
    """Templates, not a wall of block indices — 50 keys differing the same way is ONE fact."""
    families = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in keys)
    return [f"{count}x {name}" for name, count in sorted(families.items())][:limit]


def check(component: str) -> bool:
    want = pinned(component)
    got = constructed(component)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    shape = sorted(k for k in set(want) & set(got) if want[k][0] != got[k][0])
    dtype = sorted(k for k in set(want) & set(got) if want[k][1] != got[k][1])
    ok = not (missing or extra or shape or dtype)
    mark = "OK  " if ok else "FAIL"
    print(f"{mark} {component:16s} pinned {len(want):5d}  constructed {len(got):5d}")
    for label, keys in (
        ("missing (the artifact has it, the graph does not)", missing),
        ("extra (the graph demands it, the artifact has none)", extra),
        ("shape", shape),
        ("dtype", dtype),
    ):
        if not keys:
            continue
        print(f"       {len(keys)} {label}")
        for line in summarize(keys):
            print(f"         {line}")
        for key in keys[:3]:
            if key in want and key in got:
                print(f"         {key}: pinned {want[key]} vs constructed {got[key]}")
    return ok


def main() -> int:
    names = sys.argv[1:] or list(CARRIERS)
    unknown = [n for n in names if n not in CARRIERS]
    if unknown:
        refuse(f"unknown component(s): {', '.join(unknown)}")
    results = [check(name) for name in names]
    total = sum(len(pinned(n)) for n in names)
    print(f"\n{sum(results)}/{len(results)} components key-exact over {total} destinations")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
