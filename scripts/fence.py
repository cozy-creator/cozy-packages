#!/usr/bin/env python
"""Structural fences for the package sources. Static analysis, never a test suite.

Seventeen properties CI must not let drift, each checked as a fact about the source rather
than as a convention someone remembers:

  1. author-surface-only  package code AND the repo's drivers import the PUBLIC
                          `cozy_runtime.author` surface and nothing else from the runtime
                          (boundaries.md). `cozy_runtime.internal`, an underscored
                          `author._*` mechanism, the worker protocol, TensorFS or a hub
                          client is the boundary violation the author surface exists to
                          prevent — and a driver reaching past it breaks exactly as
                          silently, so the scan covers drivers too. `DRIVER_INTERNALS`
                          enumerates the exceptions a driver has EARNED, with the reason.
  2. no-direct-artifact-loading
                          Runtime validates source-authored default ladders. Package code
                          still cannot fetch weights or bypass the verified model loader.
  3. top-level-imports    every import is at module scope (Paul, 2026-09-08): an import
                          under a def or class hides a dependency from the file's head.
                          Until se-041 that was how package code kept torch off CI's
                          torch-free `describe`; production `describe` runs in the
                          package's LOCKED environment (Creator publishes and installs
                          there), CI now does the same, and a heavy module-scope import
                          is the honest spelling. Armed on every run.
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
  9. no-package-bindings  package.toml carries no [bindings]: callable decorators own
                          authored defaults; explicit owner overrides live on the hub.
10. interface-minimality
                          committed interface/1 files carry no retired unused facts.
 11. h3-adaln-pruned-vocabulary
                          H3 source and contracts carry no retired modulation spelling.
12. package-metadata-hardcut
                          package.toml and PackageInterface/1 are the only source metadata;
                          the retired source filenames and canonical namespace are absent.
13. private-h3-shapes    H3 config, plan, and probe files are identified by their package
                          member and strict shape, not another globally versioned schema tag.
14. publication-metadata every publishable package declares its catalog organization and its
                          distribution name carries no redundant `-package` suffix; its wheel
                          exposes exactly one `cozy.application` entry matching package.toml.
15. native-publication-wheels
                          every reachable non-base dependency that Creator cannot mirror as an
                          exact `py3-none-any` registry wheel is one explicit local wheel whose
                          stored bytes match package-local provenance and the current lock.
16. driver-boundary-armed  the driver rule FIRES. A boundary check nobody has watched go
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
import itertools
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
    return parts[:2] in PUBLIC_RUNTIME_ROOTS and not any(part.startswith("_") for part in parts[2:])


#: Every private Runtime module ONE named driver may import, with the reason it is not a
#: public surface and what would retire the entry. Enumerated because an accepted coupling
#: that is written down is one a rename can find; the alternative is not fewer couplings,
#: only invisible ones.
DRIVER_INTERNALS: dict[tuple[str, str], str] = {
    (
        "examples/client-scripts/h3-lora-profile/h3_lora_scratch_probe.py",
        "cozy_runtime.internal.lora",
    ): (
        "Private experiment compares exact pinned Runtime LoRA methods using the original "
        "hook classes and request context, without mutating serving weights or global methods. "
        "Retire with this one-off scratch qualification; this is not a public author API."
    ),
    (
        "examples/client-scripts/h3-attention-oracle/attention_quantized.py",
        "cozy_runtime.internal.attention_fp8",
    ): (
        "Private experiment measures the actual production FP8 quantizer, including "
        "preprocessing, on identical captured H3 tensors. Avoid copying its implementation "
        "or relying on a whole-model pin. Retire when public diagnostic backend lookup "
        "exposes that exact operation."
    ),
    ("scripts/h3-shared-vae-repair-proof.py", "cozy_runtime.author._model"): (
        "Bind exact native fixture manifests; retire when a public native source factory can "
        "bind production-shaped manifests instead of test:// identities. "
    ),
    ("scripts/h3-restamp-native-proof.py", "cozy_runtime.author._model"): (
        "Bind the native source checkpoint for the metadata-only migration proof. "
    ),
    ("scripts/h3-turbo-cp-proof.py", "cozy_runtime.internal.parallel"): (
        "Verify actual PDD hooks and component scopes under Runtime's Ulysses installation. "
    ),
    ("scripts/h3-turbo-store-proof.py", "cozy_runtime.author._model"): (
        "Bind native source manifests for the overlay integration proof. "
    ),
    ("scripts/h3-longform-proof.py", "cozy_runtime.author._calls"): (
        "the long-form driver builds the real child broker so the chain is proven through "
        "Runtime's own call exchange rather than a fake. Retire when a public child-call "
        "harness can construct a broker. "
    ),
    ("scripts/h3-longform-proof.py", "cozy_runtime.author._assets"): (
        "the same driver grants the previous shot's bytes to the next child the way the worker "
        "does. Retire with the same public child-call harness. "
    ),
    ("scripts/sdxl-normalization-proof.py", "cozy_runtime.author._model"): (
        "the normalization driver constructs the exact-checkpoint source Runtime admits; "
        "for_test permits test:// identities only. Retire with a public native job fixture. "
    ),
    ("scripts/sdxl-normalization-proof.py", "cozy_runtime.internal.weights_sink"): (
        "the normalization driver proves custody and process restart through Runtime's native "
        "weights host. Retire when the public fake service can construct this host. "
    ),
    ("scripts/sdxl_normalization_plan.py", "cozy_runtime.author._loader"): (
        "the plan proof uses Runtime's exact constructor census to compare tensor order; "
        "retire when that conformance census has a public driver surface. "
    ),
    ("scripts/h3-adaln-binding-proof.py", "cozy_runtime.author._model"): (
        "Native/broker qualification driver exercises the real Runtime boundary; retire when a "
        "public native invocation harness owns this fixture seam. "
    ),
    ("scripts/h3-adaln-resume-proof.py", "cozy_runtime.author._model"): (
        "Native/broker qualification driver exercises the real Runtime boundary; retire when a "
        "public native invocation harness owns this fixture seam. "
    ),
    ("scripts/h3-adaln-interface-proof.py", "cozy_runtime.author._calls"): (
        "Native/broker qualification driver exercises the real Runtime boundary; retire when a "
        "public native invocation harness owns this fixture seam. "
    ),
    ("scripts/h3-adaln-interface-proof.py", "cozy_runtime.internal"): (
        "Native/broker qualification driver exercises the real Runtime boundary; retire when a "
        "public native invocation harness owns this fixture seam. "
    ),
    ("scripts/h3-adaln-interface-proof.py", "cozy_runtime.internal.discovery"): (
        "Native/broker qualification driver exercises the real Runtime boundary; retire when a "
        "public native invocation harness owns this fixture seam. "
    ),
    ("scripts/h3-repair-proof.py", "cozy_runtime.author._model"): (
        "the repair integration driver constructs the same exact-checkpoint job source record "
        "as Runtime; for_test intentionally permits test:// identities only. Retire when a "
        "public native job fixture factory owns this constructor. "
    ),
    ("scripts/h3-conform.py", "cozy_runtime.internal.residency"): (
        "the warm arms raise Runtime's OWN `ResidencyRefusal` — the class, the typed "
        "`device_shortfall` code and the verbatim detail an H100 produced — so the package's "
        "tolerance of a parked component is proven against what actually raises it rather than "
        "a local look-alike (se-046). Retire when the author surface names the capacity "
        "refusal a `warm` body must tolerate. "
    ),
    ("scripts/h3-conform.py", "cozy_runtime.internal.derive"): (
        "the derive harness is Runtime's CONSTRUCTION plane and cannot become author surface: "
        "`cozy_runtime.author` is torch-free by construction and Runtime's own "
        "`checks/architecture.py` forbids it importing `cozy_runtime.internal`. The supported "
        "spelling is `cozy-model-contract-proof`, which today derives only inside a receipt- "
        "selected sandboxed build seat and has no in-process entry point for one synthetic "
        "model. Retire this entry when it grows one. "
    ),
    ("scripts/native_execution_fixture.py", "cozy_runtime.internal.seam"): (
        "Exercise the actual worker journal and descriptors; retire with a public native fixture. "
    ),
    ("scripts/native_execution_fixture.py", "cozy_runtime.internal.weights_writer"): (
        "Exercise the actual worker journal and descriptors; retire with a public native fixture. "
    ),
    ("scripts/native_execution_fixture.py", "cozy_runtime.internal.worker.grants"): (
        "Exercise the actual worker journal and descriptors; retire with a public native fixture. "
    ),
    ("scripts/native_execution_fixture.py", "cozy_runtime.internal.worker.weights"): (
        "Exercise the actual worker journal and descriptors; retire with a public native fixture. "
    ),
    ("scripts/native_execution_fixture.py", "cozy_runtime.internal.worker.workspace"): (
        "Exercise the actual worker journal and descriptors; retire with a public native fixture. "
    ),
    ("scripts/native_execution_fixture.py", "cozy_runtime.protocol"): (
        "Exercise the actual worker journal and descriptors; retire with a public native fixture. "
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
        if not driver and top in STORE_AND_NETWORK and module != "tensorfs.derived":
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


def local_imports(label: str, source: str) -> list[str]:
    """Every import statement below module scope in ONE file, over its real AST.

    A module-level `try:` or `if TYPE_CHECKING:` block IS module scope: the rule is that the
    file's head declares what the file needs, and both of those still do."""
    bad: list[str] = []

    def visit(node: ast.AST, scope: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if scope and isinstance(child, ast.Import | ast.ImportFrom):
                bad.append(
                    f"{label}:{child.lineno}: import inside `{scope}` — every import goes at "
                    "the top of the file (Paul, 2026-09-08)"
                )
            inner = scope
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                inner = f"def {child.name}"
            elif isinstance(child, ast.ClassDef):
                inner = f"class {child.name}"
            visit(child, inner)

    visit(ast.parse(source, filename=label), None)
    return bad


#: The arm: the spellings the rule exists to catch, and the module-scope ones it must admit.
ARM_LOCAL_IMPORTS = (
    "def f():\n    import torch\n",
    "async def f():\n    from diffusers import X\n",
    "class C:\n    def m(self):\n        import numpy as np\n",
    "def f():\n    if flag:\n        from . import sibling\n",
    "def f():\n    try:\n        import cozy_eval\n    except ImportError:\n        pass\n",
)
ARM_MODULE_IMPORTS = (
    "import torch\n",
    "from x import y\n",
    "try:\n    import cozy_eval\nexcept ImportError:\n    cozy_eval = None\n",
    "if TYPE_CHECKING:\n    from x import Y\n",
)


def fence_top_level_imports() -> Fence:
    """Paul (2026-09-08): never import inside a function body. An import under a def or a
    class hides a dependency from the file's head — and until se-041 it was how package code
    kept torch off a torch-free `describe`, a constraint the production describe (Creator, in
    the package's LOCKED environment) never had. Fired on its own arm first: a rule nobody
    has watched go red is a rule nobody knows works."""
    bad = [
        f"a planted import below module scope does not fail the fence: {source!r}"
        for source in ARM_LOCAL_IMPORTS
        if not local_imports("arm.py", source)
    ]
    bad += [
        f"a module-scope import fails the fence: {source!r}"
        for source in ARM_MODULE_IMPORTS
        if local_imports("arm.py", source)
    ]
    files = sorted(f for f in ROOT.rglob("*.py") if ours(f))
    for path in files:
        bad += local_imports(rel(path), path.read_text())
    return bad, (
        f"every import in {len(files)} modules is at module scope; {len(ARM_LOCAL_IMPORTS)} "
        f"planted spellings fire and {len(ARM_MODULE_IMPORTS)} module-scope ones do not"
    )


#: Direct artifact-loading spellings. Release references are allowed declarative data;
#: Runtime's descriptor validation owns their default-ladder shape and argument binding.
IDENTIFIERS = (
    (re.compile(r"\bsha256:[0-9a-f]{16}"), "a checkpoint digest"),
    (re.compile(r"\bfrom_pretrained\b"), "a from_pretrained call (the loader constructs)"),
    (re.compile(r"\bhf_hub_download\b|\bsnapshot_download\b"), "a weight fetch"),
)


#: Digest spellings a package has EARNED, with the reason. minimax-h3-tools is the H3
#: structural precompute (cr-071): its whole job is attesting the exact plan/config/
#: topology bytes it derives from, so a `sha256:` there is a self-integrity pin over its
#: own committed assets and expected outputs — never a model-selection binding, which
#: lives on the hub. Scoped to the checkpoint-digest shape only; every other
#: identifier rule applies to these files unchanged.
DIGEST_PINNED_PREFIXES = ("minimax-h3-tools/src/h3_tables/",)


def fence_identifiers() -> Fence:
    bad: list[str] = []
    for path in package_modules():
        source = path.read_text()
        # Prose may discuss loaders and digests; executable source stays checked.
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
        source = path.read_text()
        code = _strip_literals(source)
        # Encoding returned latent samples is an output operation, not a model
        # checkpoint reader. Permit only this explicit write-only import; load,
        # safe_open and module imports still trigger the source-format fence.
        serializers = {
            node.lineno
            for node in ast.parse(source).body
            if isinstance(node, ast.ImportFrom)
            and node.module == "safetensors.torch"
            and node.names
            and all(alias.name == "save" for alias in node.names)
        }
        for line_no, line in enumerate(code.splitlines(), 1):
            if line_no in serializers:
                # Remove only the permitted import token, not other operations on
                # the same line (for example a semicolon followed by torch.load).
                line = line.replace("safetensors.torch", "", 1)
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


def fence_no_package_bindings() -> Fence:
    """Model selection is the owner's hub binding, never a default shipped in source."""

    bad: list[str] = []
    manifests = sorted(p for p in ROOT.rglob("package.toml") if ours(p))
    for path in manifests:
        if "bindings" in tomllib.loads(path.read_text()):
            bad.append(
                f"{rel(path)}: [bindings] is not package metadata; bind the slot on the hub with "
                "`cozy package bind <package> <slot> org/model@release --gpu <gpu>=<lane> ...`"
            )
    return bad, f"{len(manifests)} package.toml files carry no [bindings]"


#: The Anima model card's negative prompt, spelled here independently of the package so a
#: drift in either copy is a red rather than two edits agreeing with each other (se-026).
ANIMA_CARD_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, "
    "artist name, blurry, jpeg artifacts, chromatic aberration"
)

#: The card quality prefix the package prepends when the caller does not override it. It
#: carries no rating tag: se-026 proposed `safe` and the owner declined it (2026-09-03), so
#: a tag appearing here would be a silent reversal of a recorded decision.
ANIMA_CARD_QUALITY_PREFIX = "masterpiece, best quality, "
ANIMA_RATING_TAGS = ("safe", "sensitive", "nsfw", "explicit")
#: Anima's render phases, spelled here independently of the package. `cozy run list` renders
#: "<stage> <percent>" verbatim, so a stage name is product copy a person reads.
ANIMA_PHASE_NAMES = ("encoding prompt", "conditioning", "denoise", "decoding", "saving image")


def fence_anima_defaults() -> Fence:
    """The request default negative is the card string that carries this model's quality."""

    bad = _anima_field_default("negative_prompt", ANIMA_CARD_NEGATIVE)
    bad += _anima_field_default("quality_prefix", ANIMA_CARD_QUALITY_PREFIX)
    bad += _anima_no_rating_tag()
    bad += _anima_number_default("cfg_interval_start", "0.15")
    bad += _anima_number_default("cfg_interval_stop", "0.7")
    bad += _anima_number_default("first_block_cache", "0.0")
    return bad, "Anima defaults to the card prompt strings"


def _anima_generate_input() -> tuple[ast.Module, ast.ClassDef | None]:
    """The parsed anima module and its GenerateInput class."""
    tree = ast.parse(
        (ROOT / "anima" / "anima" / "__init__.py").read_text(),
        filename="anima/anima/__init__.py",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GenerateInput":
            return tree, node
    return tree, None


def _anima_field_default(field: str, expected: str) -> list[str]:
    """One `GenerateInput` string default, read off the AST.

    These defaults are the whole of se-026: they are what every request that does not spell
    the field gets, and the owner's labelled bank isolates the negative as the causal
    quality knob. Reading the annotated assignment rather than importing the module keeps
    the fence static, and folding the implicit concatenation here is why the source may stay
    wrapped.
    """
    tree, generate_input = _anima_generate_input()
    if generate_input is None:
        return ["anima/anima/__init__.py declares no GenerateInput"]
    for item in generate_input.body:
        if not (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == field
        ):
            continue
        value = _anima_literal(tree, item.value)
        if value == expected:
            return []
        return [f"anima GenerateInput.{field} default is {value!r}, expected {expected!r}"]
    return [f"anima GenerateInput declares no {field} field"]


def _anima_number_default(field: str, expected: str) -> list[str]:
    """One `GenerateInput` numeric default, spelled as source text.

    `first_block_cache` is the load-bearing one: se-026 measured it extrapolating 44% of
    forwards, smoothing faces and coarsening fine texture, and refusing `device_shortfall`
    at the default 1536 class on an 8 GiB card when `cfg_interval` is not also narrowing the
    live cache states. Turning it on is a decision with a quality bank behind it, not a
    default someone may flip while tuning.
    """
    _tree, generate_input = _anima_generate_input()
    if generate_input is None:
        return ["anima/anima/__init__.py declares no GenerateInput"]
    for item in generate_input.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == field
        ):
            got = ast.unparse(item.value) if item.value is not None else "<none>"
            return (
                []
                if got == expected
                else [f"anima GenerateInput.{field} default is {got}, expected {expected}"]
            )
    return [f"anima GenerateInput declares no {field} field"]


def _anima_no_rating_tag() -> list[str]:
    """No card rating tag rides on a package-supplied default (owner ruling, 2026-09-03).

    The proposal se-026 carried was a `rating` enum prepended server-side; the owner
    declined it in those words. A rating tag reappearing inside a default string is how that
    decision would get reversed without anyone deciding to reverse it, so the fence reads
    the defaults rather than trusting the field name to stay honest.
    """
    tree, generate_input = _anima_generate_input()
    if generate_input is None:
        return ["anima/anima/__init__.py declares no GenerateInput"]
    bad: list[str] = []
    for item in generate_input.body:
        if not (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id in ("prompt", "quality_prefix")
        ):
            continue
        value = _anima_literal(tree, item.value) or ""
        tags = {tag.strip() for tag in value.split(",")}
        found = sorted(tags.intersection(ANIMA_RATING_TAGS))
        if found:
            bad.append(
                f"anima GenerateInput.{item.target.id} default supplies rating tag(s) "
                f"{found}: the owner declined a package-supplied rating (2026-09-03)"
            )
    return bad


def _anima_literal(tree: ast.Module, node: ast.expr | None) -> str | None:
    """A str default written literally, or via ONE module-level str constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        for item in tree.body:
            if (
                isinstance(item, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == node.id for t in item.targets)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ):
                return item.value.value
    return None


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

    bad: list[str] = _anima_progress_ladder()
    required: dict[str, set[str]] = {
        "sdxl/sdxl/__init__.py": {"denoise"},
        "minimax-h3/h3.py": {"denoise"},
    }
    for rel, stages in required.items():
        _, steps, overall = calls(ROOT / rel)
        missing = stages - steps
        if missing:
            bad.append(f"{rel}: no measured step_callback for stages {sorted(missing)}")
        missing_overall = stages - overall
        if missing_overall:
            bad.append(f"{rel}: no overall_range for stages {sorted(missing_overall)}")
    return bad, "generation loops report measured stage steps and overall ranges"


def _anima_progress_ladder() -> list[str]:
    """Anima's phase ladder: named for a person, contiguous, and covering the request.

    Anima names its stages through `_Phase` constants rather than at each call site, because
    one modular-pipeline call spans four of them. This reads the constants, not the calls.

    THE AST ONLY SAYS THE LADDER IS SPELLED RIGHT. That the hooks FIRE is
    `scripts/anima-conform.py`'s job — se-026 shipped a denoise hook installed on a deepcopy
    of the block tree, so a fence like this one was green while the meter sat frozen at
    `conditioning 0%` for the whole render.
    """
    tree, _ = _anima_generate_input()
    phases: dict[str, tuple[str, float, float]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0]
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "_Phase":
            continue
        args = [a.value for a in node.value.args if isinstance(a, ast.Constant)]
        named = isinstance(args[0], str) if args else False
        numbers = [a for a in args[1:] if isinstance(a, (int, float))]
        if isinstance(target, ast.Name) and named and len(numbers) == 2:
            phases[target.id] = (str(args[0]), float(numbers[0]), float(numbers[1]))
    ladder = list(phases.values())
    bad: list[str] = []
    if [name for name, _start, _stop in ladder] != list(ANIMA_PHASE_NAMES):
        return [f"anima phase ladder is {[p[0] for p in ladder]}, expected {ANIMA_PHASE_NAMES}"]
    if (ladder[0][1], ladder[-1][2]) != (0.0, 1.0):
        bad.append(f"anima phase ladder covers {ladder[0][1]}..{ladder[-1][2]}, expected 0.0..1.0")
    for (name, start, stop), (following, next_start, _) in itertools.pairwise(ladder):
        if start >= stop:
            bad.append(f"anima phase {name!r} does not advance: {start}..{stop}")
        if stop != next_start:
            bad.append(
                f"anima phase gap between {name!r} and {following!r}: {stop} != {next_start}"
            )
    measured = {
        phases.get(keyword.value.value.id, ("", 0.0, 0.0))[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step_callback"
        and any(k.arg == "overall_range" for k in node.keywords)
        for keyword in node.keywords
        if keyword.arg == "stage"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
    }
    if "denoise" not in measured:
        bad.append("anima: the denoise phase has no step_callback carrying its overall_range")
    return bad


def fence_interface_format() -> Fence:
    """Every interface is exactly an interface/1 document with the four root fields."""
    expected = {"application", "entrypoints", "format", "jobs"}
    bad: list[str] = []
    for project in projects():
        path = project / "metadata" / "package-interface.json"
        try:
            document = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            bad.append(f"{rel(path)}: unreadable: {exc}")
            continue
        if set(document) != expected or document.get("format") != "cozy.package.interface/1":
            bad.append(f"{rel(path)}: root fields/format are not exact interface/1")
    return bad, f"{len(projects())} interface/1 files carry the four root fields"


def fence_publication_metadata() -> Fence:
    bad: list[str] = []
    expected_projects = {"anima", "sdxl", "minimax-h3", "minimax-h3-tools"}
    actual_projects = {
        rel(path.parent)
        for path in ROOT.rglob("package.toml")
        if ours(path) and not path.is_relative_to(ROOT / "examples" / "client-scripts")
    }
    if actual_projects != expected_projects:
        bad.append(
            f"deployable projects must be {sorted(expected_projects)}; "
            f"found {sorted(actual_projects)}; one-off work belongs in examples/client-scripts"
        )
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
        package_config = tomllib.loads((project / "package.toml").read_text())
        application = package_config.get("application")
        expected = application.get("object") if isinstance(application, dict) else None
        if not isinstance(applications, dict) or list(applications.values()) != [expected]:
            bad.append(
                f'{rel(path)}: [project.entry-points."cozy.application"] must expose '
                f"exactly the package.toml application {expected!r}"
            )
    return bad, f"{len(projects())} packages declare one catalog and installed identity"


FENCES = (
    ("author-surface-only", fence_author_surface),
    ("driver-boundary-armed", fence_driver_arm),
    ("top-level-imports", fence_top_level_imports),
    ("no-direct-artifact-loading", fence_identifiers),
    ("no-memory-choreography", fence_no_choreography),
    ("no-test-suite", fence_no_tests),
    ("h3-media-boundary", fence_h3_media_boundary),
    ("h3-official-hardcut", fence_h3_official_hardcut),
    ("env-free-packages", fence_no_env),
    ("no-package-bindings", fence_no_package_bindings),
    ("anima-defaults", fence_anima_defaults),
    ("step-progress", fence_step_progress),
    ("interface-format", fence_interface_format),
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
