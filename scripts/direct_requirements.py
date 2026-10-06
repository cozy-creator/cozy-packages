#!/usr/bin/env python
"""Every direct-reference requirement in this environment is the one installed.

`uv pip check` compares versions only, so a lock that overrides a dependency's
`name @ git+...@<commit>` with a registry release passes it, and the dependency fails
only when it runs (minimax-h3 1.26.0 overrode qwen-image-2's Diffusers commit; a
callee runs in its caller's environment). Run it with the environment's interpreter:

    minimax-h3/.venv/bin/python scripts/direct_requirements.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import sys

from packaging.requirements import Requirement


def installed_reference(name: str) -> str | None:
    try:
        raw = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return ""
    direct = json.loads(raw)
    vcs = direct.get("vcs_info")
    url = direct["url"]
    return f"{vcs['vcs']}+{url}@{vcs['commit_id']}" if vcs else url


def main() -> int:
    failures = []
    for dist in metadata.distributions():
        for line in dist.requires or []:
            requirement = Requirement(line)
            if not requirement.url or (
                requirement.marker and not requirement.marker.evaluate({"extra": ""})
            ):
                continue
            have = installed_reference(requirement.name)
            if have != requirement.url:
                failures.append(
                    f"{dist.metadata['Name']} requires {requirement.name} @ {requirement.url}; "
                    f"this environment has {have or 'a registry release'}"
                )
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"{len(failures)} direct-reference requirement(s) unmet")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
