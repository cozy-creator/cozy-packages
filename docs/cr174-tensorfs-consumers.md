# TensorFS consumer migration (cr-174)

Owner: Codex `/root/consumer_delivery_finish` (continued from the native reader owner).
Worktree: `~/cozy/.worktrees/packages/cr174-tensorfs-consumers`.
Branch: `refactor/cr174-tensorfs-consumers`.
Base: fetched `origin/master` `c75fb9a89ef4651bdc5e67c2547cd64a03e68539`.
Current merged base: `e73bd86`.

The four core packages and client-script examples use TensorFS
`Derivation`/`SourceInspection`/`DerivedTransaction` and Runtime Context execution
bindings. Generic quantization remains the shared Runtime operation; family code
continues to provide model policy and mathematical operations.

Runtime draft PR463 owns the adapter. Its real Creator CLI proof covers native
writer descriptors, edited-caller memo reuse, partial checkpoint adoption and
Context-only tensor production/inheritance. These source changes require the
joint cr-170/cr-171/cr-174 SDK. Runtime 0.18.0 is reserved and its minimum is
prepared in package and client metadata. TensorFS 0.3.42 is published. Package
versions and frozen locks will be updated against the actual public Runtime release
before activation; proof still uses exact source wheels.
No package will be published from this incomplete migration or with an unqualified
SDK dependency/interface bound. Frozen 0.16.10 proof artifacts are preserved.


All H3 job/lanes/source/AdaLN/turbo consumers, SDXL normalization/assembly, family prepare
handlers and private client producers now use Context plus TensorFS declarations. Source
inspection preserves nonlexical constructor order. Native repair, shared-VAE graft, lane,
AdaLN/turbo and SDXL process-exit/resume proofs preserve bytes, object identities and replay
behavior. The family readback example consumes TensorFS Tensor.parts as a mapping.

The native writer/metadata and reader components are public TensorFS 0.3.41/0.3.42.
The coordinated Runtime 0.18.0 release and package versions/locks remain pending. These library migration proofs do not
claim final H3 rendering or disconnected/private/default-home product qualification.

Merged current package master e73bd86, preserving the new H3 first-step and mixed
checkpoint qualification fixtures. The latter's newly added legacy checkpoint
producer and seed script now use Context plus native TensorFS declarations; no
model mathematics changed. Its exact scale-2/scale-3 f32 bytes and native replay
are checked separately; this does not claim a new GPU qualification.

The isolated followup uses Runtime candidate source 806bf675, public TensorFS
0.3.42, and candidate Eval 908ffd3. Package/example types pass for 85 source files,
Ruff and all 14 fences pass. The new mixed-model fixture now states the generated
client's pending-call type explicitly and confines missing Torch subclass types to
that optional fixture module. SDXL normalization passes its real process-exit,
resume, no-read replay, graft identity, and component assembly controls. H3 order
and staged table/FP8/MXFP8 resume/replay controls also pass. Installed candidate
imports show SDXL generation is not memoized, while normalization and Eval fact
measurements are; generation and quality bind the explicit model and judge slots.

See [native consumer release checks](native-consumer-release-checks.md) for the
six remaining public lock updates and final SDK/CLI qualification. Evidence lives
under `outputs/consumer-delivery-finish` in the Cozy v2 workspace.
