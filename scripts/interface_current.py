#!/usr/bin/env python3
"""Check a committed PackageInterface against this tree's `cozy-runtime --json describe` (stdin).

Fails only when the committed document omits or contradicts what the tree describes. Ordering,
formatting, and keys that one Runtime version emits and the other never does are ignored, so a
Runtime release that adds interface metadata does not invalidate an unchanged package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def keys(value: Any, found: set[str]) -> set[str]:
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            keys(item, found)
    elif isinstance(value, list):
        for item in value:
            keys(item, found)
    return found


def project(value: Any, shared: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: project(v, shared) for k, v in value.items() if k in shared}
    if isinstance(value, list):
        items = [project(v, shared) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def difference(committed: Any, described: Any, path: str) -> str:
    if isinstance(committed, dict) and isinstance(described, dict):
        for key in sorted(committed.keys() | described.keys()):
            if key not in committed or key not in described:
                return f"{path}.{key}"
            found = difference(committed[key], described[key], f"{path}.{key}")
            if found:
                return found
        return ""
    if isinstance(committed, list) and isinstance(described, list):
        named = {str(item.get("name")) for item in committed + described if isinstance(item, dict)}
        for name in sorted(named):
            left = [item for item in committed if isinstance(item, dict) and item.get("name") == name]
            right = [item for item in described if isinstance(item, dict) and item.get("name") == name]
            if left != right:
                if len(left) != 1 or len(right) != 1:
                    return f"{path}[{name}]"
                return difference(left[0], right[0], f"{path}[{name}]") or f"{path}[{name}]"
        return "" if committed == described else f"{path}[]"
    return "" if committed == described else path


def main() -> None:
    path = Path(sys.argv[1])
    committed = json.loads(path.read_bytes())
    described = json.loads(sys.stdin.buffer.read())
    shared = keys(committed, set()) & keys(described, set())
    found = difference(project(committed, shared), project(described, shared), "$")
    if found:
        raise SystemExit(
            f"{path} omits or contradicts this tree at {found}; "
            "regenerate it with `cozy-runtime --json describe`"
        )
    print(f"{path} agrees with this tree's describe output")


if __name__ == "__main__":
    main()
