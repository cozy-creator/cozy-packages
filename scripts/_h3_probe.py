"""Select the H3 probe subject without letting installed-wheel checks see repo source."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import NoReturn

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_TREE = (ROOT / "h3").resolve()


def select(argv: list[str], *modules: str) -> tuple[bool, dict[str, pathlib.Path]]:
    """Pop ``--installed-wheel`` and return exact origins for its subject modules."""
    installed = "--installed-wheel" in argv[1:]
    argv[1:] = [arg for arg in argv[1:] if arg != "--installed-wheel"]
    if not installed:
        sys.path.insert(0, str(SOURCE_TREE))
        return False, {}

    origins: dict[str, pathlib.Path] = {}
    for name in modules:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            _refuse(f"installed-wheel subject {name!r} is not importable")
        origin = pathlib.Path(spec.origin).resolve()
        if origin == SOURCE_TREE or SOURCE_TREE in origin.parents:
            _refuse(f"--installed-wheel resolved {name!r} from source tree {origin}")
        origins[name] = origin
    return True, origins


def _refuse(message: str) -> NoReturn:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)
