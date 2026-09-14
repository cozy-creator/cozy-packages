# TensorFS consumer migration (cr-174)

Owner: Codex `/root/tensorfs_native_resume`.
Worktree: `~/cozy/.worktrees/packages/cr174-tensorfs-consumers`.
Branch: `refactor/cr174-tensorfs-consumers`.
Base: fetched `origin/master` `c75fb9a89ef4651bdc5e67c2547cd64a03e68539`.

The four core packages and client-script examples use TensorFS
`Derivation`/`SourceInspection`/`DerivedTransaction` and Runtime Context execution
bindings. Generic quantization remains the shared Runtime operation; family code
continues to provide model policy and mathematical operations.

Runtime draft PR463 owns the adapter. Its real Creator CLI proof covers native
writer descriptors, edited-caller memo reuse, partial checkpoint adoption and
Context-only tensor production/inheritance. These source changes require the
joint cr-170/cr-171/cr-174 SDK, not the currently published 0.17.0 SDK. Candidate
Runtime minimum 0.18.0 and the complete TensorFS API remain release-coordination decisions; public
version files are unchanged and proof uses exact development-wheel overrides.
No package will be published from this incomplete migration or with an unqualified
SDK dependency/interface bound. Frozen 0.16.10 proof artifacts are preserved.


All H3 job/lanes/source/AdaLN/turbo consumers, SDXL normalization/assembly, family prepare
handlers and private client producers now use Context plus TensorFS declarations. Source
inspection preserves nonlexical constructor order. Native repair, shared-VAE graft, lane,
AdaLN/turbo and SDXL process-exit/resume proofs preserve bytes, object identities and replay
behavior. The family readback example consumes TensorFS Tensor.parts as a mapping.

The writer/metadata component is public TensorFS0.3.41; the payload reader is qualified
with TensorFS PR170 source03244db and Runtime9b0263c8. The coordinated published Runtime
floor and package release versions remain pending. These library migration proofs do not
claim final H3 rendering or disconnected/private/default-home product qualification.
