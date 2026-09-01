#!/usr/bin/env python
"""Structural fences for the package sources. Static analysis, never a test suite.

Sixteen properties CI must not let drift, each checked as a fact about the source rather than
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
15. publication-metadata every publishable package declares its catalog organization and its
                          distribution name carries no redundant `-package` suffix; its wheel
                          exposes exactly one `cozy.application` entry matching package.toml.
16. native-publication-wheels
                          every reachable non-base dependency that Creator cannot mirror as an
                          exact `py3-none-any` registry wheel is one explicit local wheel whose
                          stored bytes match package-local provenance and the current lock.
17. model-execution-ownership
                          modeled packages select Runtime's complete execution capability; package
                          metadata never falsely claims TensorFS as a direct dependency.

    nice -n 19 .venv/bin/python scripts/fence.py
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import pathlib
import re
import sys
import tokenize
import urllib.parse
from collections.abc import Callable, Iterator

import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
Fence = tuple[list[str], str]


# Creator cc2593b8's complete current CPU base roots; CUDA is a superset. This consumer-side
# closure fence must move with that explicit publication contract, never infer platform ownership
# from whichever packages happen to be installed on a developer machine.
PLATFORM_OWNED_ROOTS = frozenset(
    {
        "av",
        "cffi",
        "click",
        "cozy-runtime",
        "cozy-runtime-cuda-kernels",
        "cryptography",
        "filelock",
        "fsspec",
        "grpcio",
        "jinja2",
        "markupsafe",
        "mpmath",
        "msgspec",
        "networkx",
        "numpy",
        "packaging",
        "pillow",
        "pip",
        "protobuf",
        "pycparser",
        "setuptools",
        "sympy",
        "tensorfs",
        "torch",
        "torchaudio",
        "torchvision",
        "typing-extensions",
        "uv",
    }
)
FIRST_PARTY_LOCAL_WHEELS = frozenset({"cozy-eval", "cozy-runtime", "tensorfs"})
REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


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
    """The bare local install selects one exact immutable model lane."""

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
    return bad, "SDXL binds the exact paul/wai-illustrious@17.0.0/bf16 lane"


def fence_anima_defaults() -> Fence:
    """The bare local install selects one exact immutable model lane."""

    manifest = tomllib.loads((ROOT / "anima" / "package.toml").read_text())
    binding = manifest.get("bindings", {}).get("generate.models.model", {})
    expected = {
        "model": "paul/anima",
        "release": "1.0.0",
        "lane": "bf16",
    }
    bad = (
        []
        if binding == expected
        else [f"anima/package.toml: got {binding!r}, expected {expected!r}"]
    )
    return bad, "Anima binds the exact paul/anima@1.0.0/bf16 lane"


def fence_anima_progress() -> Fence:
    """Anima projects the maintained Diffusers loop onto Runtime's measured lane."""

    path = ROOT / "anima" / "anima" / "__init__.py"
    tree = ast.parse(path.read_text())
    stages: set[str] = set()
    measured_steps = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        stage = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "stage"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            "",
        )
        if node.func.attr == "progress" and stage:
            stages.add(stage)
        if node.func.attr == "step_callback" and stage == "denoise":
            measured_steps = True
    bad: list[str] = []
    missing = {"conditioning", "decoding"} - stages
    if missing:
        bad.append(f"anima: missing progress stages {sorted(missing)}")
    if not measured_steps:
        bad.append("anima: denoise loop does not use Runtime's measured step_callback")
    return bad, "Anima reports conditioning, measured denoising steps, and decoding"


def fence_descriptor_format() -> Fence:
    """Every descriptor is exactly a descriptor/1 document with the four root fields."""
    expected = {"application", "entrypoints", "format", "jobs"}
    bad: list[str] = []
    for project in projects():
        path = project / "package.descriptor.json"
        try:
            document = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            bad.append(f"{rel(path)}: unreadable: {exc}")
            continue
        if set(document) != expected or document.get("format") != "cozy.package.descriptor/1":
            bad.append(f"{rel(path)}: root fields/format are not exact descriptor/1")
    return bad, f"{len(projects())} descriptor/1 files carry the four root fields"


def fence_publication_metadata() -> Fence:
    bad: list[str] = []
    for project in projects():
        path = project / "pyproject.toml"
        try:
            document = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            bad.append(f"{rel(path)}: unreadable: {exc}")
            continue
        tool = document.get("tool")
        cozy = tool.get("cozy") if isinstance(tool, dict) else None
        organization = cozy.get("organization") if isinstance(cozy, dict) else None
        if (
            not isinstance(organization, str)
            or not organization
            or organization != organization.strip()
        ):
            bad.append(f"{rel(path)}: [tool.cozy].organization is not one non-empty string")
        project_table = document.get("project")
        name = project_table.get("name") if isinstance(project_table, dict) else None
        if not isinstance(name, str) or not name or name.endswith("-package"):
            bad.append(f"{rel(path)}: [project].name has an absent or redundant package name")
        entry_points = (
            project_table.get("entry-points") if isinstance(project_table, dict) else None
        )
        applications = (
            entry_points.get("cozy.application") if isinstance(entry_points, dict) else None
        )
        package_manifest = tomllib.loads((project / "package.toml").read_text())
        application = package_manifest.get("application")
        expected = application.get("object") if isinstance(application, dict) else None
        if not isinstance(applications, dict) or list(applications.values()) != [expected]:
            bad.append(
                f"{rel(path)}: [project.entry-points.\"cozy.application\"] must expose "
                f"exactly the package.toml application {expected!r}"
            )
    return bad, f"{len(projects())} packages declare one catalog and installed identity"


def _distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_distribution(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = REQUIREMENT_NAME.match(value.strip())
    return _distribution_name(match.group()) if match else None


def _dependency_distribution(value: object) -> str | None:
    if isinstance(value, str):
        return _distribution_name(value)
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return _distribution_name(value["name"])
    return None


def _locked_wheel_filename(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    filename = value.get("filename")
    if isinstance(filename, str):
        return filename
    url = value.get("url")
    if not isinstance(url, str):
        return None
    return urllib.parse.unquote(pathlib.PurePosixPath(urllib.parse.urlsplit(url).path).name)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def fence_native_publication_wheels() -> Fence:
    bad: list[str] = []
    verified = 0
    for package in projects():
        pyproject_path = package / "pyproject.toml"
        lock_path = package / "uv.lock"
        provenance_path = package / "vendor" / "native-provenance.json"
        try:
            document = tomllib.loads(pyproject_path.read_text())
            lock = tomllib.loads(lock_path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            bad.append(f"{rel(package)}: project or lock is unreadable: {exc}")
            continue

        project_table = document.get("project")
        dependencies = project_table.get("dependencies") if isinstance(project_table, dict) else []
        dependency_values = dependencies if isinstance(dependencies, list) else []
        direct = {
            name
            for raw in dependency_values
            if (name := _requirement_distribution(raw)) is not None
        }
        tool = document.get("tool")
        uv = tool.get("uv") if isinstance(tool, dict) else None
        sources_value = uv.get("sources") if isinstance(uv, dict) else None
        sources = sources_value if isinstance(sources_value, dict) else {}

        rows_value = lock.get("package")
        rows: dict[str, list[dict[str, object]]] = {}
        if not isinstance(rows_value, list):
            bad.append(f"{rel(lock_path)}: package rows are absent")
            continue
        for value in rows_value:
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                bad.append(f"{rel(lock_path)}: malformed package row")
                continue
            rows.setdefault(_distribution_name(value["name"]), []).append(value)

        # CUDA kernels belong to the exact Torch/CUDA base profile. A package may see the
        # optional requirement in Runtime's wheel metadata, but activating it here would resolve
        # and publish a second copy in the overlay instead of using the qualified base capability.
        if "cozy-runtime-cuda-kernels" in rows:
            bad.append(
                f"{rel(lock_path)}: base-owned cozy-runtime-cuda-kernels resolved into the "
                "package overlay"
            )

        reachable: set[str] = set()
        queue = list(direct)
        while queue:
            name = queue.pop()
            if name in reachable:
                continue
            reachable.add(name)
            if name in PLATFORM_OWNED_ROOTS:
                continue
            candidates = rows.get(name, [])
            if len(candidates) != 1:
                bad.append(f"{rel(lock_path)}: reachable {name} has {len(candidates)} lock rows")
                continue
            row = candidates[0]
            source = row.get("source")
            if not isinstance(source, dict):
                bad.append(f"{rel(lock_path)}: reachable {name} has no source")
                continue
            if source.get("registry") is not None:
                wheels = row.get("wheels")
                wheel_values = wheels if isinstance(wheels, list) else []
                if not any(
                    isinstance(filename := _locked_wheel_filename(wheel), str)
                    and filename.lower().endswith("-py3-none-any.whl")
                    for wheel in wheel_values
                ):
                    bad.append(
                        f"{rel(lock_path)}: reachable non-base {name} remains in Creator's "
                        "native-only registry lane"
                    )
            child_values = row.get("dependencies")
            if isinstance(child_values, list):
                queue.extend(
                    child
                    for item in child_values
                    if (child := _dependency_distribution(item)) is not None
                )

        first_party_files: set[str] = set()
        local_native: dict[str, str] = {}
        for raw_name, raw_source in sources.items():
            if not isinstance(raw_name, str) or not isinstance(raw_source, dict):
                continue
            path = raw_source.get("path")
            name = _distribution_name(raw_name)
            if not isinstance(path, str) or not path.lower().endswith(".whl"):
                continue
            if name in FIRST_PARTY_LOCAL_WHEELS:
                first_party_files.add(pathlib.PurePosixPath(path).name)
            else:
                local_native[name] = path

        records: list[object] = []
        if provenance_path.exists():
            try:
                provenance = json.loads(provenance_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                bad.append(f"{rel(provenance_path)}: unreadable: {exc}")
                provenance = None
            if not isinstance(provenance, dict) or set(provenance) != {"wheels"}:
                bad.append(f"{rel(provenance_path)}: root fields are not exact")
            elif isinstance(provenance.get("wheels"), list):
                records = provenance["wheels"]
            else:
                bad.append(f"{rel(provenance_path)}: wheels is not a list")
        elif local_native:
            bad.append(f"{rel(provenance_path)}: absent for local native wheels")

        provenance_names: list[str] = []
        provenance_files: set[str] = set()
        fields = {"distribution", "filename", "length", "sha256", "source_url", "version"}
        for index, record in enumerate(records):
            where = f"{rel(provenance_path)}.wheels[{index}]"
            if not isinstance(record, dict) or set(record) != fields:
                bad.append(f"{where}: fields are not exact")
                continue
            distribution = record.get("distribution")
            filename = record.get("filename")
            length = record.get("length")
            digest = record.get("sha256")
            source_url = record.get("source_url")
            version = record.get("version")
            string_values = (distribution, filename, digest, source_url, version)
            if not all(isinstance(value, str) for value in string_values):
                bad.append(f"{where}: string identity field is malformed")
                continue
            assert isinstance(distribution, str)
            assert isinstance(filename, str)
            assert isinstance(digest, str)
            assert isinstance(source_url, str)
            assert isinstance(version, str)
            name = _distribution_name(distribution)
            provenance_names.append(name)
            provenance_files.add(filename)
            parsed = urllib.parse.urlsplit(source_url)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "files.pythonhosted.org"
                or parsed.port is not None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or urllib.parse.unquote(pathlib.PurePosixPath(parsed.path).name) != filename
            ):
                bad.append(f"{where}: source URL does not name the exact PyPI wheel")
            if name not in direct:
                bad.append(f"{where}: {name} is not an explicit runtime requirement")
            expected_path = f"vendor/{filename}"
            if local_native.get(name) != expected_path:
                bad.append(f"{where}: [tool.uv.sources] does not select {expected_path}")
            candidates = rows.get(name, [])
            if len(candidates) != 1:
                bad.append(f"{where}: {name} has {len(candidates)} lock rows")
                continue
            row = candidates[0]
            source = row.get("source")
            wheels = row.get("wheels")
            wheel_values = wheels if isinstance(wheels, list) else []
            if not isinstance(source, dict) or source.get("path") != expected_path:
                bad.append(f"{where}: lock source does not select {expected_path}")
            if row.get("version") != version or not any(
                _locked_wheel_filename(wheel) == filename
                and isinstance(wheel, dict)
                and wheel.get("hash") == f"sha256:{digest}"
                for wheel in wheel_values
            ):
                bad.append(f"{where}: lock filename, version, or SHA-256 differs")
            path = package / expected_path
            if (
                not isinstance(length, int)
                or length <= 0
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != length
                or _sha256(path) != digest
            ):
                bad.append(f"{where}: stored wheel bytes differ from provenance")
            else:
                verified += 1

        if provenance_names != sorted(set(provenance_names)):
            bad.append(f"{rel(provenance_path)}: distributions are not unique and sorted")
        if set(local_native) != set(provenance_names):
            bad.append(f"{rel(package)}: local native sources and provenance differ")
        stored = {path.name for path in (package / "vendor").glob("*.whl")}
        if stored != first_party_files | provenance_files:
            bad.append(f"{rel(package / 'vendor')}: stored wheels differ from declared sources")

    return bad, f"{verified} exact local wheels close every non-base native publication edge"


def fence_model_execution_ownership() -> Fence:
    bad: list[str] = []
    tensorfs_wheel = (
        "vendor/tensorfs-0.0.8-cp312-abi3-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
    )
    for name in ("anima", "sdxl"):
        path = ROOT / name / "pyproject.toml"
        document = tomllib.loads(path.read_text())
        project = document.get("project", {})
        dependencies = project.get("dependencies", [])
        expected = "cozy-runtime[media,model-execution]>=0.0.29,<1.0.0"
        if expected not in dependencies:
            bad.append(f"{rel(path)}: modeled package does not select {expected}")
        if any(_requirement_distribution(value) == "tensorfs" for value in dependencies):
            bad.append(f"{rel(path)}: package falsely declares TensorFS directly")
        dev = document.get("dependency-groups", {}).get("dev", [])
        if "tensorfs==0.0.8" not in dev:
            bad.append(f"{rel(path)}: temporary local TensorFS source anchor is absent")
        source = document.get("tool", {}).get("uv", {}).get("sources", {}).get("tensorfs")
        if source != {"path": tensorfs_wheel}:
            bad.append(f"{rel(path)}: local source does not bind exact TensorFS 0.0.8 wheel")
    return bad, "modeled packages select Runtime-owned TensorFS with no direct package dependency"


FENCES = (
    ("author-surface-only", fence_author_surface),
    ("no-identifiers-in-code", fence_identifiers),
    ("no-memory-choreography", fence_no_choreography),
    ("no-test-suite", fence_no_tests),
    ("h3-media-boundary", fence_h3_media_boundary),
    ("h3-official-hardcut", fence_h3_official_hardcut),
    ("env-free-packages", fence_no_env),
    ("h3-binding-identity", fence_h3_binding_identity),
    ("typed-model-bindings", fence_typed_model_bindings),
    ("sdxl-defaults", fence_sdxl_defaults),
    ("anima-defaults", fence_anima_defaults),
    ("anima-progress", fence_anima_progress),
    ("descriptor-format", fence_descriptor_format),
    ("publication-metadata", fence_publication_metadata),
    ("native-publication-wheels", fence_native_publication_wheels),
    ("model-execution-ownership", fence_model_execution_ownership),
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
