#!/usr/bin/env python
"""Import one endpoint's real stack inside its own locked environment.

The compiled extensions are the ABI proof. `torchvision.ops` and `torch` resolve C++
symbols at import, so a lock that pins the wrong torch fails HERE with an undefined-symbol
error — a fact no static check and no lock comparison can reach. Run it with the endpoint's
own interpreter, never the checking venv:

    sdxl/.venv/bin/python scripts/endpoint_import.py sdxl
"""

from __future__ import annotations

import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: endpoint directory -> (module to import, compiled peers that must load beside it)
ENDPOINTS = {
    "sdxl": ("sdxl", ("torch", "torchvision", "torchvision.ops", "triton", "transformers",
                      "diffusers", "tensorfs", "av")),
    "quality-judge": ("quality_judge", ("torch", "triton", "transformers", "tokenizers",
                                        "numpy", "PIL.Image", "tensorfs", "av")),
    "h3": ("h3", ("torch", "torchvision", "torchvision.ops", "triton", "transformers",
                  "diffusers", "tensorfs", "av")),
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ENDPOINTS:
        print(f"usage: {argv[0]} <{'|'.join(ENDPOINTS)}>", file=sys.stderr)
        return 2
    endpoint = argv[1]
    module_name, peers = ENDPOINTS[endpoint]
    sys.path.insert(0, str(ROOT / endpoint))
    for peer in peers:
        importlib.import_module(peer)
    torch = importlib.import_module("torch")
    print(f"{endpoint}: torch {torch.__version__}, {len(peers)} peers imported")
    module = importlib.import_module(module_name)
    app = module.app
    print(f"{endpoint}: {module_name}:app registers {sorted(app._registry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
