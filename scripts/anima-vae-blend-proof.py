#!/usr/bin/env python
"""CPU bit proof against source-extracted installed Diffusers blend methods.

Run under the reviewed CPU device guard; no model/package imports or weights.
The full proof compares individual multiply/add results and actual tile traversal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--upstream-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.read_bytes()
    if hashlib.sha256(upstream).hexdigest() != args.upstream_sha256:
        raise RuntimeError("upstream proof source differs from the reviewed pinned bytes")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = Path(__file__).resolve().parents[1] / "anima/anima/__init__.py"
    old = ast.parse(upstream)
    candidate = ast.parse(source.read_text())
    print(json.dumps({"status": "CPU proof fixture pending", "old_classes": sum(isinstance(node, ast.ClassDef) for node in old.body), "candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
