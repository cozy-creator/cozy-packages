#!/usr/bin/env python
"""Every endpoint lock that carries torch must carry the SAME torch.

Torch, torchvision, torchaudio, triton and the `nvidia-*-cu12` wheels are one ABI-coupled
unit: torchvision's compiled ops link against torch's C++ ABI, so an endpoint built against
one torch raises undefined-symbol errors under another. Decision #633 gives one
PlatformTarget exactly one torch family and makes a second family a second substrate
variant, never a per-endpoint choice — so the drift that matters is BETWEEN this repo's
locks, and it is cheap to observe here instead of expensively on a card.

    python scripts/torch_family.py
"""

from __future__ import annotations

import pathlib
import sys

import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The coupled unit. `nvidia-*-cu12` are torch's own pinned CUDA runtime wheels and move
#: with it, so a divergence there is the same defect wearing a different name.
PREFIXES = ("torch", "triton", "nvidia-")


def family(lock: pathlib.Path) -> dict[str, str]:
    with open(lock, "rb") as handle:
        document = tomllib.load(handle)
    found: dict[str, str] = {}
    for package in document.get("package", []):
        name = str(package.get("name", ""))
        if name.startswith(PREFIXES):
            found[name] = str(package.get("version", ""))
    return found


def main() -> int:
    locks = sorted(ROOT.glob("*/uv.lock"))
    if not locks:
        print("no endpoint lock found — this check would pass vacuously", file=sys.stderr)
        return 1
    seen: dict[str, dict[str, str]] = {}
    drift: list[str] = []
    for lock in locks:
        endpoint = lock.parent.name
        found = family(lock)
        if not found:
            print(f"{endpoint}: no torch family (nothing to agree with)")
            continue
        for name, version in sorted(found.items()):
            first = seen.setdefault(name, {})
            for other, other_version in first.items():
                if other_version != version:
                    drift.append(f"{name}: {other} pins {other_version}, {endpoint} pins {version}")
            first[endpoint] = version
        torch = found.get("torch", "(absent)")
        print(f"{endpoint}: torch {torch}, {len(found)} coupled wheels")
    if drift:
        print("\nREFUSED — one PlatformTarget, one torch family (#633):", file=sys.stderr)
        for line in sorted(set(drift)):
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"\none torch family across {len(locks)} lock(s): {len(seen)} coupled wheels agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
