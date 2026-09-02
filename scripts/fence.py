#!/usr/bin/env python
"""Structural fences for the package sources. Static analysis, never a test suite.

Eighteen properties CI must not let drift, each checked as a fact about the source rather
than as a convention someone remembers:

  1. author-surface-only  package code AND the repo's drivers import the PUBLIC
                          `cozy_runtime.author` surface and nothing else from the runtime
                          (boundaries.md). `cozy_runtime.internal`, an underscored
                          `author._*` mechanism, the worker protocol, TensorFS or a hub
                          client is the boundary violation the author surface exists to
                          prevent — and a driver reaching past it breaks exactly as
                          silently, so the scan covers drivers too. `DRIVER_INTERNALS`
                          enumerates the exceptions a driver has EARNED, with the reason.
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
18. driver-boundary-armed  the driver rule FIRES. A boundary check nobody has watched go
                          red is a boundary check nobody knows works, so the private
                          spellings are run through the same predicate on every run.
17. model-execution-ownership
                          modeled packages select Runtime's complete execution capability; package
                          metadata never falsely claims TensorFS as a direct dependency.

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


def driver_modules() -> list[pathlib.Path]:
    """Source WE wrote that no package project owns: the repo's conformance and check
    scripts. No boundaries.md rule reaches a driver, which is exactly why an import of a
    private Runtime module here used to be invisible — it holds the same Runtime the
    packages ship against, and a `_assets` or `internal` refactor breaks it in silence."""
    inside = {f for project in projects() for f in project.rglob("*.py")}
    return sorted(f for f in ROOT.rglob("*.py") if ours(f) and f not in inside)


def h3_modules() -> list[pathlib.Path]:
    return sorted(f for f in (ROOT / "minimax-h3").rglob("*.py") if ours(f))


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


#: The runtime surface anything in this repo may import: `cozy_runtime.author` and
#: `cozy_runtime.derive` (cr-071's public derivation plane for job packages) and their
#: PUBLIC submodules. An underscored submodule — `author._assets`, `author._services` — is
#: Runtime's own mechanism, and `cozy_runtime.internal` is the plane the author surface
#: exists to hide. Both break the same way: the name stops existing on a later Runtime with
#: nothing here having said it depended on one.
PUBLIC_RUNTIME_ROOTS = (["cozy_runtime", "author"], ["cozy_runtime", "derive"])


def public_runtime_surface(module: str) -> bool:
    parts = module.split(".")
    return parts[:2] in PUBLIC_RUNTIME_ROOTS and not any(
        part.startswith("_") for part in parts[2:]
    )


#: Every private Runtime module ONE named driver may import, with the reason it is not a
#: public surface and what would retire the entry. Enumerated because an accepted coupling
#: that is written down is one a rename can find; the alternative is not fewer couplings,
#: only invisible ones.
DRIVER_INTERNALS: dict[tuple[str, str], str] = {
    ("scripts/h3-conform.py", "cozy_runtime.internal.derive"): (
        "the derive harness is Runtime's CONSTRUCTION plane and cannot become author "
        "surface: `cozy_runtime.author` is torch-free by construction and Runtime's own "
        "`checks/architecture.py` forbids it importing `cozy_runtime.internal`. The "
        "supported spelling is `cozy-model-contract-proof`, which today derives only "
        "inside a receipt-selected sandboxed build seat and has no in-process entry point "
        "for one synthetic model. Retire this entry when it grows one."
    ),
}

STORE_AND_NETWORK = ("tensorfs", "tensorhub", "cozy_creator", "grpc", "requests", "httpx")


def surface_violations(label: str, source: str, *, driver: bool) -> list[str]:
    """The boundary rule for ONE file, over its real AST imports.

    A module name is a fact about an import statement here, never a word found in the
    source: a fence that reads prose goes red on a docstring, which is a recorded incident
    in this repo and not a hypothetical one.
    """
    bad: list[str] = []
    for module, line in imports(ast.parse(source, filename=label)):
        top = module.split(".")[0]
        if top == "cozy_runtime" and not public_runtime_surface(module):
            if driver and (label, module) in DRIVER_INTERNALS:
                continue
            who = "a driver" if driver else "a package"
            how = (
                " Use the public surface, or record the exception in DRIVER_INTERNALS."
                if driver
                else ""
            )
            bad.append(
                f"{label}:{line}: {module!r} — {who} imports the PUBLIC "
                f"`cozy_runtime.author` surface and nothing else from the runtime; a "
                f"private module is a coupling that breaks silently.{how}"
            )
        if not driver and top in STORE_AND_NETWORK:
            bad.append(
                f"{label}:{line}: {module!r} — package code speaks to no store, "
                "no hub and no network; every byte it sees arrives as a typed input"
            )
    return bad


def fence_author_surface() -> Fence:
    bad: list[str] = []
    for path in package_modules():
        bad += surface_violations(rel(path), path.read_text(), driver=False)
    imported: set[tuple[str, str]] = set()
    for path in driver_modules():
        source = path.read_text()
        bad += surface_violations(rel(path), source, driver=True)
        imported |= {
            (rel(path), module) for module, _ in imports(ast.parse(source, filename=rel(path)))
        }
    for label, module in DRIVER_INTERNALS:
        if (label, module) not in imported:
            bad.append(
                f"{label}: recorded internal {module!r} is not imported — a stale exception "
                "reads as a coupling that still exists and grants one that does not"
            )
    return bad, (
        f"{len(package_modules())} package modules and {len(driver_modules())} drivers "
        f"import the public author surface, past {len(DRIVER_INTERNALS)} recorded exception(s)"
    )


#: The arm. Each is the spelling the rule exists to catch, run through the same predicate.
ARM_PRIVATE = (
    "from cozy_runtime.author._assets import asset_dec_hook, bind",
    "from cozy_runtime.author._services import Attempt",
    "from cozy_runtime.internal.derive import derive",
    "import cozy_runtime.internal.executor",
    "import cozy_runtime",
)
ARM_PUBLIC = (
    "from cozy_runtime.author import Outputs, decode_request",
    "from cozy_runtime.author.fakes import fake_attempt, fake_input",
    "import cozy_runtime.author",
    "from cozy_runtime.derive.quantization import derive_fp8",
)


def fence_driver_arm() -> Fence:
    """Fire the driver rule. A boundary check nobody has watched go red is a boundary check
    nobody knows works — and this one covers files no other rule in this repo reaches."""
    bad: list[str] = []
    for source in ARM_PRIVATE:
        if not surface_violations("scripts/arm.py", source, driver=True):
            bad.append(f"a driver importing {source!r} does not fail the fence")
        if not surface_violations("h3/arm.py", source, driver=False):
            bad.append(f"package code importing {source!r} does not fail the fence")
    for source in ARM_PUBLIC:
        if surface_violations("scripts/arm.py", source, driver=True):
            bad.append(f"a driver importing the public {source!r} fails the fence")
    recorded = "from cozy_runtime.internal.derive import derive"
    if surface_violations("scripts/h3-conform.py", recorded, driver=True):
        bad.append("the recorded exception does not admit the driver it names")
    if not surface_violations("scripts/torch_family.py", recorded, driver=True):
        bad.append("a recorded exception admits a driver it does not name")
    if not surface_violations("scripts/h3-conform.py", ARM_PRIVATE[0], driver=True):
        bad.append("a recorded exception admits a private module it does not name")
    return bad, (
        f"{len(ARM_PRIVATE)} private spellings fire the boundary, {len(ARM_PUBLIC)} public "
        "ones do not, and a recorded exception admits exactly its own driver and module"
    )


#: Things that identify an ARTIFACT rather than a capability. Deliberately literal: this
#: catches the spelling a hurried edit actually uses.
IDENTIFIERS = (
    (re.compile(r"\bsha256:[0-9a-f]{16}"), "a checkpoint digest"),
    (re.compile(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+"), "a pinned release ref"),
    (re.compile(r"\bfrom_pretrained\b"), "a from_pretrained call (the loader constructs)"),
    (re.compile(r"\bhf_hub_download\b|\bsnapshot_download\b"), "a weight fetch"),
)


#: Digest spellings a package has EARNED, with the reason. minimax-h3-tools is the H3
#: structural precompute (cr-071): its whole job is attesting the exact plan/config/
#: topology bytes it derives from, so a `sha256:` there is a self-integrity pin over its
#: own committed assets and expected outputs — never a model-selection binding, which
#: still lives in package.toml. Scoped to the checkpoint-digest shape only; every other
#: identifier rule applies to these files unchanged.
DIGEST_PINNED_PREFIXES = ("minimax-h3-tools/src/h3_tables/",)


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
                    if what == "a checkpoint digest" and rel(path).startswith(
                        DIGEST_PINNED_PREFIXES
                    ):
                        continue
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
    legacy_dir = ROOT / "minimax-h3" / "h3_arch"
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
    binding = (ROOT / "minimax-h3" / "package.toml").read_text()
    releases = re.findall(r'^release\s*=\s*"([^"]+)"\s*$', binding, flags=re.MULTILINE)
    lanes = re.findall(r'^lane\s*=\s*"([^"]+)"\s*$', binding, flags=re.MULTILINE)
    bad: list[str] = []
    if not releases:
        bad.append("h3/package.toml: no default model release is bound")
    if len(set(releases)) > 1:
        bad.append(f"h3/package.toml: default model bindings disagree: {sorted(set(releases))}")
    # One grammar, the slot path (model-code-fit §1): both H3 slots bind, and both name
    # the same release and profile selector.
    if len(releases) != 2 or set(releases) != {"1.0.0"}:
        bad.append(f"h3/package.toml: default releases are {releases!r}, expected two of '1.0.0'")
    for release in releases:
        if re.search(r"(?:^|[-_.])se-\d+(?:$|[-_.])", release):
            bad.append(
                f"h3/package.toml: release {release!r} contains a tracker issue, "
                "not only content identity"
            )
    if len(lanes) != 2 or set(lanes) != {"profile=fp8-adaln-pruned"}:
        bad.append(
            "h3/package.toml: both slot bindings must use the exact "
            f"profile selector, got {lanes!r}"
        )
    return bad, "H3 binds both slots to release 1.0.0 with the profile=fp8-adaln-pruned selector"


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


def fence_step_progress() -> Fence:
    """Every generation loop reports stage steps and package-owned overall ranges.

    Paul's ruling: no denoising (or comparably long iterative) loop runs silently — each
    iteration posts an incremental update on Runtime's measured lane."""

    def calls(path: pathlib.Path) -> tuple[set[str], set[str], set[str]]:
        progress: set[str] = set()
        steps: set[str] = set()
        overall: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
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
                progress.add(stage)
                if any(keyword.arg == "overall_fraction" for keyword in node.keywords):
                    overall.add(stage)
            if node.func.attr == "step_callback" and stage:
                steps.add(stage)
                if any(keyword.arg == "overall_range" for keyword in node.keywords):
                    overall.add(stage)
        return progress, steps, overall

    bad: list[str] = []
    anima_progress, anima_steps, anima_overall = calls(
        ROOT / "anima" / "anima" / "__init__.py"
    )
    stage_gap = {"conditioning", "decoding"} - anima_progress
    if stage_gap:
        bad.append(f"anima: missing progress stages {sorted(stage_gap)}")
    overall_gap = {"conditioning", "denoise", "decoding"} - anima_overall
    if overall_gap:
        bad.append(f"anima: missing overall progress for stages {sorted(overall_gap)}")
    required: dict[str, set[str]] = {
        "anima/anima/__init__.py": {"denoise"},
        "sdxl/sdxl/__init__.py": {"denoise"},
        "minimax-h3/h3.py": {"denoise"},
        "video-assembly/video_assembly.py": {"scan", "assemble"},
    }
    for rel, stages in required.items():
        _, steps, overall = (
            (anima_progress, anima_steps, anima_overall)
            if rel.startswith("anima/")
            else calls(ROOT / rel)
        )
        missing = stages - steps
        if missing:
            bad.append(f"{rel}: no measured step_callback for stages {sorted(missing)}")
        missing_overall = stages - overall
        if missing_overall:
            bad.append(f"{rel}: no overall_range for stages {sorted(missing_overall)}")
    return bad, "generation loops report measured stage steps and overall ranges"


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


FENCES = (
    ("author-surface-only", fence_author_surface),
    ("driver-boundary-armed", fence_driver_arm),
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
    ("step-progress", fence_step_progress),
    ("descriptor-format", fence_descriptor_format),
    ("publication-metadata", fence_publication_metadata),
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
