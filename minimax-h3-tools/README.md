# tensorhub/minimax-h3-tools

This is the official H3 producer package. Its stable callable reference is
`tensorhub/minimax-h3-tools@v2/four-lane`.

Publisher ownership is not project metadata. Creator derives it from the current authenticated
Tensorhub account; the official reference above assumes the `tensorhub` account.

The ordinary `four-lane` job consumes two reviewed TensorFS source profiles from the same
pinned MiniMaxAI/MiniMax-H3 provider download:

- `dits` → `hf/minimax-h3/native-dual-bf16/1`: the native FL2VA and Ref2VA
  transformers, converted together by `h3.native/1`;
- `shared` → `hf/minimax-h3/shared-bf16/1`: the text conditioner, video VAE, and audio
  VAE under `diffusers.identity/1`.

Both slots can also bind the same complete `bf16-full` checkpoint. The job reads
each distinct granted checkpoint's structure once and drops only source-only rows
still present. The existing full checkpoint has already removed those rows; its
retained BF16 tensors feed the same table and quantization computations. Required
modulation and quantized replacement tensors remain mandatory, and TensorFS still
checks the complete destination order. This does not accept a pruned or quantized
checkpoint as a substitute for the full BF16 source.

One attempt emits exactly four dual-task checkpoints:

- `bf16-full`
- `bf16-adaln-pruned`
- `fp8-adaln-pruned`
- `mxfp8-adaln-pruned`

The standalone `assemble_full` job remains available for direct use; `four-lane` does not
invoke or nest it and owns the same transformations directly inside one weight-production
attempt. The former per-task table jobs and `assemble_dual` are gone: `retable` replaces
their table pass over an existing pruned checkpoint, and two jobs with byte-identical
descriptors cannot coexist under Runtime 0.4's immutable job plans.

Each lane contains both FL2VA and Ref2VA DiTs plus the shared 902/703/1,087 components.
Full assembly drops the native-only `rope.inv_freq` buffer, leaving the exact 638-row
Diffusers DiTs. It also drops the official text model's layers 50–63, final norm, and
language-model head—156 inherited refs—to produce the reviewed 902-row, 50-layer
pre-norm conditioner without rewriting any retained payload. Each pruned task then
replaces 106 dynamic AdaLN rows with 51 BF16 timestep-table rows carrying the union of
every schedule in the task plan (30, 40 and 50 evaluations: 315 block rows and 207
final-normalization rows, 1,020,515,328 table bytes), so one pruned checkpoint serves every
served step count. FP8 and MXFP8 are independent children of those pruned BF16 task
components and never derive from each other.

The ordinary `retable` job recomputes only those tables for an existing AdaLN-pruned
checkpoint of any encoding: `pruned` (the checkpoint to inherit, BF16, FP8 or MXFP8) and
`full` (the complete BF16 checkpoint whose modulation weights the rows are computed from).
Every non-table tensor is inherited by reference and nothing is requantized, so widening the
plan's schedule set costs table bytes only. It emits two outputs from one table pass:
`adaln-pruned` (the retabled checkpoint, every component inherited from `pruned`) and
`tables` (a two-DiT table bank derived from `full` with every other row dropped) — a
transaction may read only the source components its targets derive from, so the bank
transaction is where the modulation weights are read. The retabled checkpoint commits
first. It refuses before any read unless `pruned` carries table rows and no dynamic
modulation weights for both DiTs and `full` carries the exact modulation weights.

The package declares no GPU, SM, VRAM, or host-RAM guess. Creator derives accelerator-class work
from the typed model inputs; exact artifact residency and measured request/scratch envelopes drive
fit. The job opens the three pruned-output transactions together, computes each task's timestep tables once,
and writes the exact same table bytes to every active output. FP8 and MXFP8 then read the
original BF16 DiTs independently through their own caller-owned transactions. There is no
workflow graph and no nested job invocation. Tensorhub owns final retention or publication;
the job receives no publisher credential or store path.

Each checkpoint commits as soon as its computation finishes: full BF16 first, pruned
BF16 after both table passes, FP8 after both FP8 passes, then MXFP8. A retry replays
retained native receipts and computes only the remaining outputs. Terminal owner
abandonment can release those outputs; a receipt alone is not retained payload custody.
A local commit does not imply remote custody; Creator still verifies uploads before
publishing a release.

The production inputs are package-owned and immutable: exact model config, full DiT shape
contract, construction order, and task plans live as importlib resources in the tools wheel;
the shared `cozy-jobs` wheel owns the exact H3 quantization plan used here and by the standalone
quantization callables. They are not Creator-supplied assets. Run
`scripts/order-proof.py` to recheck their closed census and
`../../proofs/producer-callable.py` to validate the generated graph-free descriptor.

The four-lane result preserves the Runtime quantizer's existing weight measurements in
`weight_fidelity_this_run`, a list of rows naming output slot, component and stats: worst per-tensor relative
Frobenius error, saturated element count, measured tensor count and byte counters. Each
completed component also logs those same values before later stages run. Replayed outputs
are absent from this run's measurements; absence never means zero error. BF16 inheritance
and AdaLN table production do not claim quantizer measurements. Activation fidelity,
matched output distance and output quality require separate inference/evaluation evidence.
