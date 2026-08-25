#!/usr/bin/env python
"""Structural fences for the endpoint packages. Static analysis, never a test suite.

Four properties CI must not let drift, each checked as a fact about the source rather than
as a convention someone remembers:

  1. author-surface-only  an endpoint imports `cozy_runtime.author` and nothing else from
                          the runtime (boundaries.md). `cozy_runtime.internal`, the worker
                          protocol, TensorFS or a hub client inside endpoint code is the
                          boundary violation the whole author surface exists to prevent.
  2. no-identifiers       code states CAPABILITY, bindings state SELECTION (§1.0/§1.1).
                          A repo, release, checkpoint digest or model revision spelled in
                          endpoint code is a binding hard-coded into a build.
  3. torch-free-import    endpoint module scope may IMPORT nothing heavy: `describe` runs
                          in a disposable container with no GPU and no weights, and a
                          module-scope `import torch` makes the surface contract
                          unreadable without a CUDA image.
  4. no-test-suite        tracker README #160.

    nice -n 19 .venv/bin/python scripts/fence.py
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from collections.abc import Iterator

ROOT = pathlib.Path(__file__).resolve().parent.parent
Fence = tuple[list[str], str]

#: Every endpoint project: a directory with an endpoint.toml.
def projects() -> list[pathlib.Path]:
    return sorted(p.parent for p in ROOT.glob("*/endpoint.toml"))


def endpoint_modules() -> list[pathlib.Path]:
    return sorted(f for project in projects() for f in project.rglob("*.py"))


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


def ours(path: pathlib.Path) -> bool:
    """Source WE wrote. A venv under the repo is not: it holds other people's test suites,
    and a fence that reports mypy's own `test_emit.py` is a fence nobody reads."""
    return not any(
        part.startswith(".venv") or part in ("site-packages", "__pycache__", ".git")
        for part in path.parts
    )


def imports(tree: ast.AST) -> Iterator[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno


def module_scope_imports(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Imports that run at IMPORT time — module-level `if`/`try`/`with` included, function
    and class bodies excluded, which is exactly where a heavy import belongs."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno
        elif isinstance(node, ast.If | ast.Try | ast.With):
            for attr in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, attr, []) or []:
                    if isinstance(child, ast.ExceptHandler):
                        stack.extend(child.body)
                    else:
                        stack.append(child)


def fence_author_surface() -> Fence:
    bad: list[str] = []
    for path in endpoint_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module, line in imports(tree):
            top = module.split(".")[0]
            if top == "cozy_runtime" and not module.startswith("cozy_runtime.author"):
                bad.append(
                    f"{rel(path)}:{line}: {module!r} — an endpoint imports "
                    "`cozy_runtime.author` and nothing else from the runtime"
                )
            if top in ("tensorfs", "tensorhub", "cozy_creator", "grpc", "requests", "httpx"):
                bad.append(
                    f"{rel(path)}:{line}: {module!r} — endpoint code speaks to no store, "
                    "no hub and no network; every byte it sees arrives as a typed input"
                )
    return bad, f"{len(endpoint_modules())} endpoint modules import author only"


#: Things that identify an ARTIFACT rather than a capability. Deliberately literal: this
#: catches the spelling a hurried edit actually uses.
IDENTIFIERS = (
    (re.compile(r"\bsha256:[0-9a-f]{16}"), "a checkpoint digest"),
    (re.compile(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+"), "a pinned release ref"),
    (re.compile(r"\bfrom_pretrained\b"), "a from_pretrained call (the loader constructs)"),
    (re.compile(r"\bhf_hub_download\b|\bsnapshot_download\b"), "a weight fetch"),
)


def fence_identifiers() -> Fence:
    bad: list[str] = []
    for path in endpoint_modules():
        source = path.read_text()
        # Docstrings and comments name the runtime's own docs (`jobs.md §7`); the rule is
        # about CODE, so the check runs over the source with strings and comments removed.
        code = _strip_literals(source)
        for line_no, line in enumerate(code.splitlines(), 1):
            for pattern, what in IDENTIFIERS:
                if pattern.search(line):
                    bad.append(
                        f"{rel(path)}:{line_no}: {what} in endpoint code — code states "
                        f"capability, bindings state selection: {line.strip()[:80]}"
                    )
    return bad, f"{len(IDENTIFIERS)} identifier shapes over {len(endpoint_modules())} modules"


def _strip_literals(source: str) -> str:
    """The source with every string literal and comment blanked, line numbers preserved."""
    import io
    import tokenize

    lines = source.splitlines()
    out = [list(line) for line in lines]
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        (row0, col0), (row1, col1) = token.start, token.end
        for row in range(row0 - 1, row1):
            start = col0 if row == row0 - 1 else 0
            end = col1 if row == row1 - 1 else len(out[row])
            for col in range(start, min(end, len(out[row]))):
                out[row][col] = " "
    return "\n".join("".join(row) for row in out)


HEAVY = {"torch", "transformers", "diffusers", "PIL", "numpy", "cv2", "safetensors",
         "tokenizers", "accelerate", "scipy"}


def fence_light_import() -> Fence:
    bad: list[str] = []
    for path in endpoint_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module, line in module_scope_imports(tree):
            if module.split(".")[0] in HEAVY:
                bad.append(
                    f"{rel(path)}:{line}: module-scope import of {module!r} — `describe` "
                    "runs with no GPU, no weights and no CUDA image, so a heavy import "
                    "belongs inside the function that needs it"
                )
    return bad, f"{len(HEAVY)} heavy packages must stay lazily imported"


def fence_no_tests() -> Fence:
    bad = [rel(path) for path in ROOT.rglob("test_*.py") if ours(path)]
    bad += [rel(path) for path in ROOT.rglob("tests") if path.is_dir() and ours(path)]
    for path in ROOT.rglob("*.py"):
        if not ours(path):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for module, line in imports(tree):
            if module.split(".")[0] in ("pytest", "unittest", "hypothesis", "nose"):
                bad.append(f"{rel(path)}:{line}: imports {module!r}")
    return bad, "no tests/ tree, no test_*.py, no test framework imported"


FENCES = (
    ("author-surface-only", fence_author_surface),
    ("no-identifiers-in-code", fence_identifiers),
    ("light-module-scope", fence_light_import),
    ("no-test-suite", fence_no_tests),
)


def main() -> int:
    failed = 0
    for name, fence in FENCES:
        bad, what = fence()
        if bad:
            failed += 1
            print(f"FENCE RED — {name}:")
            for line in bad:
                print(f"  {line}")
        else:
            print(f"fence green — {name} ({what})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
