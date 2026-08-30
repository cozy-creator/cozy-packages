#!/usr/bin/env python
"""Structural fences for the package sources. Static analysis, never a test suite.

Fourteen properties CI must not let drift, each checked as a fact about the source rather than
as a convention someone remembers:

  1. author-surface-only  a package imports `cozy_runtime.author` and nothing else from
                          the runtime (boundaries.md). `cozy_runtime.internal`, the worker
                          protocol, TensorFS or a hub client inside package code is the
                          boundary violation the whole author surface exists to prevent.
  2. no-identifiers       code states CAPABILITY, bindings state SELECTION (§1.0/§1.1).
                          A model, release, checkpoint digest or model revision spelled in
                          package code is a binding hard-coded into a build.
  3. torch-free-import    package module scope may IMPORT nothing heavy: `describe` runs
                          in a disposable container with no GPU and no weights, and a
                          module-scope `import torch` makes the surface contract
                          unreadable without a CUDA image. Checked over the IMPORT CLOSURE
                          of `[application] object`, so a package may bring its own model
                          library (H3 brings the whole MiniMax architecture) as long as
                          nothing reaches it at import time.
  4. no-memory-choreography
                          the mechanisms se-001 DELETED are gone from the source: runtime
                          quantization, source-format parsing, offload/pinning, allocator
                          and stream commands, compile markers, checkpoint reads.
  5. no-test-suite        tracker README #160.
  6. h3-media-boundary    H3 consumes Runtime decoded values; it does not open media itself.
  7. h3-official-hardcut  the official dual-task implementation has no legacy/community route.
  8. env-free-packages   package code reads no environment (se-016). Configuration is
                          typed bindings and settings; the executor ERASES `COZY_*`/token
                          env anyway, so an env read is a channel that never works in
                          production.
  9. h3-binding-identity  H3 releases describe content, never tracker issue numbers; every
                          default model binding selects the same immutable release.
10. descriptor-minimality
                          committed descriptor/1 files carry no retired unused facts.
 11. h3-adaln-pruned-vocabulary
                          H3 source and contracts carry no retired modulation spelling.
12. typed-model-bindings package.toml names selected model resources with `model` only.
13. package-manifest-hardcut
                          package.toml and PackageDescriptor/1 are the only source metadata;
                          the retired source filenames and canonical namespace are absent.
14. private-h3-shapes    H3 config, plan, and probe files are identified by their package
                          member and strict shape, not another globally versioned schema tag.

    nice -n 19 .venv/bin/python scripts/fence.py
"""

from __future__ import annotations

import ast
import io
import json
import pathlib
import re
import sys
import tokenize
from collections.abc import Callable, Iterator

import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
Fence = tuple[list[str], str]


#: Every package project: a directory with a package.toml.
def projects() -> list[pathlib.Path]:
    return sorted(p.parent for p in ROOT.glob("*/package.toml"))


def package_modules() -> list[pathlib.Path]:
    return sorted(f for project in projects() for f in project.rglob("*.py") if ours(f))


def h3_modules() -> list[pathlib.Path]:
    return sorted(f for f in (ROOT / "h3").rglob("*.py") if ours(f))


def h3_owned_modules() -> list[pathlib.Path]:
    return sorted([*h3_modules(), *ROOT.glob("scripts/h3-*.py")])


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


def ours(path: pathlib.Path) -> bool:
    """Source WE wrote. A venv under the repo is not: it holds other people's test suites,
    and a fence that reports mypy's own `test_emit.py` is a fence nobody reads."""
    return not any(
        part.startswith(".venv") or part in ("site-packages", "__pycache__", ".git")
        for part in path.parts
    )


def _from_name(node: ast.ImportFrom, package: str) -> str | None:
    """`from .dit import x` inside `h3_arch` is `h3_arch.dit`.

    RELATIVE IMPORTS USED TO BE INVISIBLE HERE — `node.level == 0` skipped every one of
    them — and a fence that cannot see half the import graph is the failure the closure
    rule exists to prevent. Found by firing the arm, not by reading the code.
    """
    if node.level == 0:
        return node.module
    parts = package.split(".") if package else []
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, node.module] if node.module else base) or None


def imports(tree: ast.AST, package: str = "") -> Iterator[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            name = _from_name(node, package)
            if name:
                yield name, node.lineno


def _type_checking_guard(node: ast.If) -> bool:
    """`if TYPE_CHECKING:` — a block that NEVER runs, so nothing in it is an import."""
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def module_scope_imports(tree: ast.Module, package: str = "") -> Iterator[tuple[str, int]]:
    """Imports that run at IMPORT time — module-level `if`/`try`/`with` included, function
    and class bodies excluded, which is exactly where a heavy import belongs."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            name = _from_name(node, package)
            if name:
                yield name, node.lineno
        elif isinstance(node, ast.If) and _type_checking_guard(node):
            continue
        elif isinstance(node, ast.If | ast.Try | ast.With):
            for attr in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, attr, []) or []:
                    if isinstance(child, ast.ExceptHandler):
                        stack.extend(child.body)
                    else:
                        stack.append(child)


def fence_author_surface() -> Fence:
    bad: list[str] = []
    for path in package_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module, line in imports(tree):
            top = module.split(".")[0]
            if top == "cozy_runtime" and not module.startswith("cozy_runtime.author"):
                bad.append(
                    f"{rel(path)}:{line}: {module!r} — a package imports "
                    "`cozy_runtime.author` and nothing else from the runtime"
                )
            if top in ("tensorfs", "tensorhub", "cozy_creator", "grpc", "requests", "httpx"):
                bad.append(
                    f"{rel(path)}:{line}: {module!r} — package code speaks to no store, "
                    "no hub and no network; every byte it sees arrives as a typed input"
                )
    return bad, f"{len(package_modules())} package modules import author only"


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
    for path in package_modules():
        source = path.read_text()
        # DOCSTRINGS AND COMMENTS ONLY are blanked, and the distinction is load-bearing:
        # prose names the runtime's own docs and the pinned upstream revision this port
        # was read from, while an ORDINARY string literal is the only way anyone would
        # actually hard-code a ref. Blanking every string (the first cut) made this fence
        # blind to its own subject — a planted `_pin = "cozy/minimax-h3@se-001"` passed.
        code = _strip_docs(source)
        for line_no, line in enumerate(code.splitlines(), 1):
            for pattern, what in IDENTIFIERS:
                if pattern.search(line):
                    bad.append(
                        f"{rel(path)}:{line_no}: {what} in package code — code states "
                        f"capability, bindings state selection: {line.strip()[:80]}"
                    )
    return bad, f"{len(IDENTIFIERS)} identifier shapes over {len(package_modules())} modules"


def _doc_spans(source: str) -> set[tuple[int, int]]:
    """(lineno, col_offset) of every DOCSTRING — module, class, function."""
    spans: set[tuple[int, int]] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            spans.add((body[0].value.lineno, body[0].value.col_offset))
    return spans


def _strip_docs(source: str) -> str:
    """The source with comments and DOCSTRINGS blanked, ordinary string literals kept."""
    docs = _doc_spans(source)
    return _blank(
        source,
        lambda token: (
            token.type == tokenize.COMMENT
            or (token.type == tokenize.STRING and token.start in docs)
        ),
    )


def _strip_literals(source: str) -> str:
    """The source with every string literal and comment blanked, line numbers preserved."""
    return _blank(source, lambda token: token.type in (tokenize.STRING, tokenize.COMMENT))


def _blank(source: str, select: Callable[[tokenize.TokenInfo], bool]) -> str:
    out = [list(line) for line in source.splitlines()]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if not select(token):
            continue
        (row0, col0), (row1, col1) = token.start, token.end
        for row in range(row0 - 1, row1):
            start = col0 if row == row0 - 1 else 0
            end = col1 if row == row1 - 1 else len(out[row])
            for col in range(start, min(end, len(out[row]))):
                out[row][col] = " "
    return "\n".join("".join(row) for row in out)


HEAVY = {
    "torch",
    "transformers",
    "diffusers",
    "PIL",
    "numpy",
    "cv2",
    "safetensors",
    "tokenizers",
    "accelerate",
    "scipy",
}


def entry_module(project: pathlib.Path) -> str:
    """The module `describe` imports, from the project's own `[application] object`."""
    for line in (project / "package.toml").read_text().splitlines():
        match = re.match(r"""\s*object\s*=\s*["']([^"':]+):""", line)
        if match:
            return match.group(1)
    raise SystemExit(f"{rel(project)}/package.toml declares no [application] object")


def resolve(project: pathlib.Path, module: str) -> pathlib.Path | None:
    """An in-project module name to its file. Out-of-project names resolve to None — a
    third-party package's own module scope is its business, not this fence's."""
    parts = module.split(".")
    for candidate in (
        project.joinpath(*parts).with_suffix(".py"),
        project.joinpath(*parts) / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def fence_light_import() -> Fence:
    """THE IMPORT CLOSURE, not the file.

    A package that brings its own model library (H3 brings the whole MiniMax
    architecture) has files that import torch at module scope and are imported only inside
    `load`. The per-file version of this rule refuses those and admits the failure it
    exists to prevent — a light-looking package module importing a heavy one indirectly.
    So the rule is: nothing reachable from `[application] object` BY MODULE-SCOPE IMPORTS
    may pull a heavy package. `describe` runs in a container with no GPU, no weights and
    no CUDA image, and this is the property that keeps it running there.
    """
    bad: list[str] = []
    checked = 0
    for project in projects():
        entry = entry_module(project)
        seen: set[str] = set()
        stack = [(entry, entry)]
        while stack:
            module, via = stack.pop()
            if module in seen:
                continue
            seen.add(module)
            path = resolve(project, module)
            if path is None:
                continue
            checked += 1
            tree = ast.parse(path.read_text(), filename=str(path))
            package = module if path.name == "__init__.py" else module.rpartition(".")[0]
            for imported, line in module_scope_imports(tree, package):
                if imported.split(".")[0] in HEAVY:
                    chain = f" (reached from {entry} via {via})" if via != module else ""
                    bad.append(
                        f"{rel(path)}:{line}: module-scope import of {imported!r}"
                        f"{chain} — `describe` runs with no GPU, no weights and no CUDA "
                        "image, so a heavy import belongs inside the function that needs it"
                    )
                stack.append((imported, module))
    return bad, f"{len(HEAVY)} heavy packages absent from {checked} import-closure modules"


#: THE DELETED MECHANISMS (se-001), as spellings rather than as a review convention:
#: runtime quantization, source-format parsing, sequencer memory choreography, compile
#: markers, offload/park and checkpoint reads.
#:
#: WHAT THIS RULE DELIBERATELY DOES NOT CLAIM. It does not refuse `.cpu()` or
#: `.to(x.device)`. Copying a decoded result to the host for the output tail, and following
#: a weight to wherever the runtime already put it, are DATA FLOW across the one boundary
#: an author still owns — not memory management. A blunter pattern reads as a stronger
#: proof and is a worse one: it turns red on every shipped package's output tail, and a
#: fence that must be suppressed is a fence nobody reads. What is NOT covered here is
#: therefore reviewed rather than proven. se-001's record named ONE live instance —
#: `sdxl.py` constructing `torch.device("cuda", 0)`, an author naming a device — and #534c
#: closed it: the handler reads the envelope off the conditioning the placed encoders
#: returned. No package here names a device now, and this fence still does not prove that.
CHOREOGRAPHY = (
    (re.compile(r"\.pin_memory\(|\.share_memory\(|\bpinned_memory\b"), "host pinning"),
    (
        re.compile(r"\btorch\.cuda\.(empty_cache|synchronize|set_device|memory_|Stream|Event)"),
        "an allocator or stream command",
    ),
    (
        re.compile(
            r"\bdevice_map\b|\benable_model_cpu_offload\b|\benable_sequential_cpu_offload\b"
            r"|\baccelerate\.dispatch_model\b|\boffload_state_dict\b"
        ),
        "an offload directive",
    ),
    (
        re.compile(r"\btorch\.compile\b|\bmark_dynamic\b|\btorch\.export\b|\baot_compile\b"),
        "a compile marker",
    ),
    (
        re.compile(
            r"\bquantize_\b|\bbitsandbytes\b|\btorchao\b|\bGPTQ\b|\bAwqConfig\b"
            r"|\bquantization_config\b"
        ),
        "runtime quantization",
    ),
    (
        re.compile(r"\bsafe_open\b|\bsafetensors\.torch\b|\btorch\.load\b|\bload_state_dict\b"),
        "a checkpoint read or source-format parse",
    ),
)


def fence_no_choreography() -> Fence:
    """se-001's structural deletion proof: the replaced mechanisms are gone from the
    SOURCE, not from a reviewer's memory. Applies to every file in a package project,
    model library included — a vendored architecture that stages or re-quantizes its own
    weights is exactly the thing the port was supposed to remove."""
    bad: list[str] = []
    for path in package_modules():
        code = _strip_literals(path.read_text())
        for line_no, line in enumerate(code.splitlines(), 1):
            for pattern, what in CHOREOGRAPHY:
                if pattern.search(line):
                    bad.append(
                        f"{rel(path)}:{line_no}: {what} in package code — device "
                        f"placement, offload and encoding are the runtime's, not the "
                        f"author's: {line.strip()[:80]}"
                    )
    modules = len(package_modules())
    return bad, f"{len(CHOREOGRAPHY)} deleted-mechanism shapes over {modules} modules"


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


def fence_h3_media_boundary() -> Fence:
    """Ref2VA receives public Runtime values; a second media plane is a boundary defect."""
    bad: list[str] = []
    forbidden_imports = {"PIL", "av"}
    forbidden_attrs = {"read_bytes", "_local", "from_file"}
    for path in h3_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module, line in imports(tree):
            if module.split(".")[0] in forbidden_imports:
                bad.append(f"{rel(path)}:{line}: imports {module!r} — Runtime owns media decode")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "read_bytes"
                and ast.unparse(node.value).startswith("_ASSETS /")
            ):
                continue  # Exact package assets are model code, never request media.
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                bad.append(
                    f"{rel(path)}:{node.lineno}: uses .{node.attr} — H3 receives only public "
                    "decoded media values"
                )
    return bad, "H3 has no Pillow, PyAV, private asset, or from-file decoder"


def fence_h3_official_hardcut() -> Fence:
    """The official Diffusers path replaces the interim port; it does not sit beside it."""
    bad: list[str] = []
    legacy_dir = ROOT / "h3" / "h3_arch"
    if legacy_dir.exists():
        bad.append("h3/h3_arch: legacy community architecture still exists")

    forbidden_defs = {"generate", "reference_to_video", "generate_long"}
    forbidden_import_roots = {"comfy", "comfy_kitchen", "diffsynth", "h3_arch"}
    for path in h3_owned_modules():
        if "trust_remote_code" in _strip_docs(path.read_text()):
            bad.append(
                f"{rel(path)}: trust_remote_code — H3 executes only the pinned installed libraries"
            )
        tree = ast.parse(path.read_text(), filename=str(path))
        for module, line in imports(tree):
            if module.split(".")[0] in forbidden_import_roots:
                bad.append(f"{rel(path)}:{line}: imports community implementation {module!r}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name in forbidden_defs
            ):
                bad.append(
                    f"{rel(path)}:{node.lineno}: legacy or composite action {node.name!r} exists"
                )
    return bad, "legacy graph/actions and community imports are absent from H3"


_ENV_ATTRS = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}


def fence_no_env() -> Fence:
    """se-016: any `os.environ`/`os.getenv` spelling in package code is red — whether
    dotted, imported by name, or aliased at import."""
    bad: list[str] = []
    for path in package_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr in _ENV_ATTRS
            ):
                bad.append(
                    f"{rel(path)}:{node.lineno}: os.{node.attr} — package code reads no "
                    "environment; configuration arrives as typed bindings and settings"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                for alias in node.names:
                    if alias.name in _ENV_ATTRS:
                        bad.append(
                            f"{rel(path)}:{node.lineno}: from os import {alias.name} — "
                            "package code reads no environment"
                        )
    modules = len(package_modules())
    return bad, f"{len(_ENV_ATTRS)} env-read spellings absent from {modules} modules"


def fence_h3_binding_identity() -> Fence:
    """A release name is product identity, not the issue that happened to cut it."""
    binding = (ROOT / "h3" / "package.toml").read_text()
    releases = re.findall(r'^release\s*=\s*"([^"]+)"\s*$', binding, flags=re.MULTILINE)
    lanes = re.findall(r'^lane\s*=\s*"([^"]+)"\s*$', binding, flags=re.MULTILINE)
    bad: list[str] = []
    if not releases:
        bad.append("h3/package.toml: no default model release is bound")
    if len(set(releases)) > 1:
        bad.append(f"h3/package.toml: default model bindings disagree: {sorted(set(releases))}")
    if releases != ["1.0.0"]:
        bad.append(f"h3/package.toml: default release is {releases!r}, expected ['1.0.0']")
    for release in releases:
        if re.search(r"(?:^|[-_.])se-\d+(?:$|[-_.])", release):
            bad.append(
                f"h3/package.toml: release {release!r} contains a tracker issue, "
                "not only content identity"
            )
    if lanes != ["profile=fp8-adaln-pruned"]:
        bad.append(
            "h3/package.toml: bare local binding must use the exact "
            f"profile selector, got {lanes!r}"
        )
    return bad, "H3 binds release 1.0.0 with the Hopper/local profile=fp8-adaln-pruned selector"


def fence_typed_model_bindings() -> Fence:
    """Every package default uses the typed model noun; the generic key is retired."""

    bad: list[str] = []
    count = 0
    for project in projects():
        path = project / "package.toml"
        document = tomllib.loads(path.read_text())
        bindings = document.get("bindings", {})
        if not isinstance(bindings, dict):
            bad.append(f"{rel(path)}: [bindings] is not a table")
            continue
        for name, value in bindings.items():
            count += 1
            if not isinstance(value, dict) or not isinstance(value.get("model"), str):
                bad.append(f"{rel(path)}: binding {name!r} does not name its model")
            if isinstance(value, dict) and "repo" in value:
                bad.append(f"{rel(path)}: binding {name!r} uses the retired generic key")
    return bad, f"{count} default bindings use the typed model key"


def fence_sdxl_defaults() -> Fence:
    """The release selection and quality defaults move together as one package release."""

    manifest = tomllib.loads((ROOT / "sdxl" / "package.toml").read_text())
    binding = manifest.get("bindings", {}).get("generate.models.model", {})
    expected = {
        "model": "paul/wai-illustrious",
        "release": "17.0.0",
        "lane": "bf16",
    }
    bad = (
        []
        if binding == expected
        else [f"sdxl/package.toml: got {binding!r}, expected {expected!r}"]
    )

    tree = ast.parse((ROOT / "sdxl" / "sdxl" / "__init__.py").read_text())
    guidance = None
    square_only = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Txt2ImgInput":
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == "guidance"
                    and isinstance(item.value, ast.Constant)
                ):
                    guidance = item.value.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "denoise"
        ):
            square_only = any(ast.unparse(arg) == "width == height" for arg in node.args)
    if guidance != 7.0:
        bad.append(f"sdxl: default guidance is {guidance!r}, expected 7.0")
    if not square_only:
        bad.append("sdxl: HiDiffusion selection is not derived from square output geometry")
    return bad, "WAI Illustrious 17.0.0 bf16, CFG 7, square-only HiDiffusion"


def fence_h3_adaln_pruned_vocabulary() -> Fence:
    """The pre-launch hardcut has one name; the retired modulation name is refused."""
    retired = "baked"
    paths = {
        ROOT / "README.md",
        ROOT / "h3" / "package.descriptor.json",
        ROOT / "h3" / "package.toml",
        *h3_owned_modules(),
        *(ROOT / "h3" / "timestep-plans").glob("*.json"),
    }
    bad: list[str] = []
    for path in sorted(paths):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if retired in line.lower():
                bad.append(
                    f"{rel(path)}:{line_no}: retired H3 modulation spelling: {line.strip()[:80]}"
                )
    return bad, f"retired H3 modulation spelling absent from {len(paths)} contract files"


def fence_descriptor_minimality() -> Fence:
    forbidden = {
        "attribute",
        "capabilities",
        "config_schema",
        "context_facts",
        "default",
        "default_sources",
        "demand",
        "discriminator",
        "emits_media",
        "error_model",
        "frozen",
        "gpu",
        "kind",
        "max_audio_channels",
        "max_audio_samples",
        "max_decoded_bytes",
        "max_pixels_per_frame",
        "max_video_frames",
        "placement",
        "preflight",
        "protocol",
        "request_features",
        "requires",
        "schema",
        "secret_schema",
        "secrets",
        "services",
        "settings",
        "shape_axes",
        "struct",
        "surface_digest",
        "values",
    }
    bad: list[str] = []

    def visit(value: object, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    bad.append(f"{where}.{key}: retired descriptor fact")
                visit(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{where}[{index}]")

    expected = {"application", "entrypoints", "format", "jobs"}
    for project in projects():
        path = project / "package.descriptor.json"
        try:
            document = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            bad.append(f"{rel(path)}: unreadable: {exc}")
            continue
        if set(document) != expected or document.get("format") != "cozy.package.descriptor/1":
            bad.append(f"{rel(path)}: root fields/format are not exact descriptor/1")
        visit(document, rel(path))
    return bad, f"{len(projects())} descriptor/1 files carry only consumed facts"


def fence_package_manifest_hardcut() -> Fence:
    bad: list[str] = []
    retired_noun = "end" + "point"
    retired_namespace = f"cozy.{retired_noun}."
    for retired in (f"{retired_noun}.toml", f"{retired_noun}.descriptor.json"):
        bad.extend(rel(path) for path in ROOT.glob(f"*/{retired}"))
    checked = [
        ROOT / "README.md",
        *ROOT.glob("scripts/*.py"),
        *(project / "package.toml" for project in projects()),
        *(project / "package.descriptor.json" for project in projects()),
    ]
    for path in checked:
        if retired_namespace in path.read_text():
            bad.append(f"{rel(path)}: retired Cozy package canonical namespace")
    return bad, f"{len(projects())} package manifests and descriptors use one package namespace"


def fence_private_h3_shapes() -> Fence:
    retired = {
        "cozy.minimax_h3.dit/1",
        "cozy.minimax_h3.production_probe/3",
        "cozy.minimax_h3.text_conditioner/1",
        "cozy.minimax_h3.timestep_plan/1",
    }
    bad: list[str] = []
    for path in [*h3_owned_modules(), *(ROOT / "h3" / "timestep-plans").glob("*.json")]:
        source = path.read_text()
        for value in retired:
            if value in source:
                bad.append(f"{rel(path)}: private H3 member carries retired schema {value!r}")
    return bad, "four private H3 schemas replaced by member-specific closed shapes"


FENCES = (
    ("author-surface-only", fence_author_surface),
    ("no-identifiers-in-code", fence_identifiers),
    ("light-module-scope", fence_light_import),
    ("no-memory-choreography", fence_no_choreography),
    ("no-test-suite", fence_no_tests),
    ("h3-media-boundary", fence_h3_media_boundary),
    ("h3-official-hardcut", fence_h3_official_hardcut),
    ("env-free-packages", fence_no_env),
    ("h3-binding-identity", fence_h3_binding_identity),
    ("typed-model-bindings", fence_typed_model_bindings),
    ("sdxl-defaults", fence_sdxl_defaults),
    ("h3-adaln-pruned-vocabulary", fence_h3_adaln_pruned_vocabulary),
    ("descriptor-minimality", fence_descriptor_minimality),
    ("package-manifest-hardcut", fence_package_manifest_hardcut),
    ("private-h3-shapes", fence_private_h3_shapes),
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
