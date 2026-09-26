# Python type checking

Run `sh scripts/typecheck.sh` from this repository. It uses the strict mypy
configuration in pyproject.toml and the repository's .venv interpreter. Set
`COZY_TYPECHECK_PYTHON=/absolute/path/to/python` to check against another prepared
environment, such as the exact worker qualification cohort. Install the typed
first-party dependencies first; a missing Runtime/TensorFS/cozy-eval contract is
an error, not permission to ignore imports.

Use the same environment in BasedPyright (or Pyright); pyrightconfig.json sets
strict checking, .venv selection, and source/stub search paths. Strict editor diagnostics
provide early feedback; the configured mypy command remains the release gate.
Optional command arguments select focused files during development, but do not
replace the full configured check before publication. On the shared developer
machine, use CI or an existing remote worker for broad checks.

New functions require annotations. Use typed JSON envelopes, validated object
narrowing, native stubs, and small Protocols for dynamic integration boundaries.
Do not add blanket ignores, Any casts, or a baseline suppression file to obtain a
green result. Existing exceptions and outstanding errors are recorded in
[the typing rollout](https://github.com/cozy-creator/tracker/blob/master/tracker/cross-cutting/xs-034-strict-python-typing.md).

Adding this configuration does not establish that the entire tree is green.
Record the source commit, interpreter/checker versions, exact command and complete
result before claiming qualification. Generated and attributed upstream code may
have documented exclusions; first-party callers must consume their current stubs.

## Baseline recorded 2026-09-26

The full configured check is **not green**. With Python 3.12.12 / mypy 2.3.1,
private Runtime 0.18.28 SDK6, TensorFS 0.3.54, the verified typed cozy-eval 0.7.2
wheel, and cached Torch 2.13.0, it reports **75 errors in 12 files (107 checked)**.
The five package source trees alone pass all **28 files**. Optional numerical
import exceptions and existing source suppressions remain debt.

The initial lightweight private-cohort check found 48 errors; repairing obsolete
H3 provenance/broker/installation fixtures, typing coverage records, and rejecting
an absent VAE diagnostic reduced that to 35 errors. Installing cached Torch
without downloads exposed additional fixture/example failures; the richer result
above is the current baseline. The five repaired files pass their focused check.
Sibling module search paths now make that focused command resolve author sources.

The unresolved client examples consume generated managed calls through imports
that currently resolve to cozy-eval author implementations. The checker therefore
sees injected services, model objects, and a synchronous quality implementation
where clients require generated awaitable/model-artifact signatures. An obsolete
LoRA scratch example also imports a removed Runtime internal module. These are
recorded failures, not missing-import exceptions.

Actual strict BasedPyright 1.40.1 LSP reported no diagnostics for H3 story and
provenance, the H3-tools source and plan validators, or Qwen's author module.
H3's main module reported 70 diagnostics, including five dynamic `Steps` type
alias errors and unknown numerical types. Existing `call-arg` ignores still hide
the decorator's managed-call typing gap: its ParamSpec removes Context but does
not remove injected services or transform model arguments into artifacts. This
needs a typed managed-call contract rather than more ignores or casts.

Built H3-tools and Qwen wheels include their `py.typed` markers. No full CI,
native execution, publication, global install, or candidate deployment ran.
PR277 remains draft until the complete new gate is qualified.

## Follow-up source repairs

The current branch incorporates the supplied-reference H3 interface and removes
the retired scratch experiment that imported Runtime's deleted private LoRA module.
Historical run artifacts are untouched. Focused checks now pass the H3 conformance,
VAE precision and compile-oracle files (three files), plus the LoRA merge, kernel
quality, first-step and attention-oracle examples (four files).

These checks use existing Torch types. A short explicit list of upstream Torch
APIs without typed signatures is exempt from `no-untyped-call`; this does not
disable checks of first-party functions or hide missing first-party imports.
The deliberately invalid LoRA operand in a negative test has a local `arg-type`
exception. Ordinary variable reuse, model dimensions, optional tensor lifetime,
and compiler reporting are repaired rather than suppressed.

The complete gate remains unqualified: generated managed-call signatures still
need to distinguish injected services/model objects from client arguments, and
the older monolithic interface assertions need reconciliation with current H3.
No full local CI, GPU execution, or installation change ran for these repairs.
