# TensorFS consumer migration (cr-174)

Owner: Codex `/root/tensorfs_scoped_finish`.
Worktree: `~/cozy/.worktrees/packages/cr174-tensorfs-consumers`.
Branch: `refactor/cr174-tensorfs-consumers`.
Base: fetched `origin/master` `c75fb9a89ef4651bdc5e67c2547cd64a03e68539`.

The four core packages and client-script examples are migrating to TensorFS
`Derivation`/`SourceInspection`/`DerivedTransaction` and Runtime Context execution
bindings. Generic quantization remains the shared Runtime operation; family code
continues to provide model policy and mathematical operations.

Runtime draft PR463 owns the adapter. Its real Creator CLI proof covers native
writer descriptors, edited-caller memo reuse, partial checkpoint adoption and
Context-only tensor production/inheritance. These source changes require the
joint cr-170/cr-171/cr-174 SDK, not the currently published 0.17.0 SDK. Candidate
minimum 0.18.0 and TensorFS 0.3.40 remain release-coordination decisions; public
version files are unchanged and proof uses exact development-wheel overrides.
No package will be published from this incomplete migration or with an unqualified
SDK dependency/interface bound. Frozen 0.16.10 proof artifacts are preserved.
