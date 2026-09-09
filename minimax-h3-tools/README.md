# tensorhub/minimax-h3-tools

This is the official H3 producer package. Its stable callable reference is
`tensorhub/minimax-h3-tools@v2/lanes`.

Publisher ownership is not project metadata. Creator derives it from the current authenticated
Tensorhub account; the official reference above assumes the `tensorhub` account.

The ordinary `lanes` job consumes two reviewed TensorFS source profiles from the same
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

## Lanes

A lane is a NAME plus, per component, what this producer does to it. The whole catalogue
lives in `h3_tables/lanes.py`; the request names which of its rows this attempt produces
and defaults to all of them. The recipes are code, never a request field: output slot names
are decorator-time facts, and every lane mints a MiniMax H3 Model Derivative under §I.11(i)
of the community licence, which is a reviewed act rather than a caller choice.

| lane | modulation | per-component treatment |
|---|---|---|
| `bf16-full` | full | none — every component inherited |
| `bf16-adaln-pruned` | AdaLN-pruned | none |
| `fp8-adaln-pruned` | AdaLN-pruned | both DiTs `fp8-rowwise/1` |
| `mxfp8-adaln-pruned` | AdaLN-pruned | both DiTs `mxfp8/1` |

A component a lane does not name is **inherited by reference**: TensorFS copies its tensor
metadata and ObjectRefs unchanged through the zero-read/zero-hash inherit gate, so the
58.2 GiB conditioner/VAE trio is the same stored objects in every lane above. Naming a
component costs its bytes once per lane that names it, which is why the treatment map is
sparse and must stay sparse.

A treatment is a dtype `cast` over the component's float32 rows, an `encode` over its
block-aligned rank-2 float weights, or both; `keep` names exact keys the encoding must not
select. Adding a lane is one catalogue row, one `WeightsOutput` line and one `LaneName`
member — an import-time check proves the three agree, and refuses a lane that treats the
audio VAE, that uses an encoding with no rung on the cards we serve, or whose declared byte
ceiling disagrees with its treatments.

Two components are refused outright. **`mxfp8/1` on any new lane**: `RowwiseNativeLeaf`
predicates `cuda.sm89+`, but `MicroScaledNativeLeaf` predicates the EQUALITY `cuda.sm120`,
so mxfp8 executes on no H100/H200/B200 and degrades to dequant plus a bf16 GEMM;
`mxfp8-adaln-pruned` is grandfathered by name and nothing else may join it. **The
`audio_vae`, by name**: 637 of its 1,087 rows are rank-3 — including the BigVGAN decoder's
344 `weight_norm` `weight_g`/`weight_v` parameters — and the rank-2 encoding cannot
represent any of them; the only six rank-2 float weights the component carries are the
`pre_block` ENCODER attention/MLP linears — so a shape rule does not refuse
the component, it silently quantizes the audio conditioning path and leaves the BigVGAN
decoder untouched.

The standalone `assemble_full` job remains available for direct use; `lanes` does not
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
checkpoint of any encoding, once per admitted table set (`job.TABLE_SETS`): `pruned` (the
checkpoint to inherit, BF16, FP8 or MXFP8), `full` (the complete BF16 checkpoint whose
modulation weights the rows are computed from) and the two PDD acceleration LoRAs
`fl2va_adapter` / `ref2va_adapter` (`alibaba-pai/MiniMax-H3-Acc-LoRAs`, rank 64, alpha 64).
Every non-table tensor is inherited by reference and nothing is requantized, so widening the
plan's schedule set costs table bytes only. Each set emits a retabled checkpoint and a
two-DiT table bank derived from `full` with every other row dropped — a transaction reads
only through source components its targets derive from, so the bank transaction is where
the modulation weights are read; the retabled checkpoint commits first:

- `launch` → `adaln-pruned` / `tables`: the 30/40/50 union plans, unchanged.
- `turbo` → `turbo-adaln-pruned` / `turbo-tables`: PDD-8, eight evaluations on
  `Schedule(9)` at the released shifts 12/3 (the 33-point training grid at its block-4
  boundaries; the pipeline is called with `num_inference_steps = 9` because the scheduler
  counts the terminal sigma), 26 block rows and 17 final-normalization rows per task,
  84,231,168 table bytes. An AdaLN-pruned lane has no `adaln_proj.linear` for an adapter to
  attach to, so each adapter's `adaln_proj.linear` LoRA slice is fused into the block rows in
  the adapter's own inference order (`bf16(W·x + b) + bf16(up(down·x))` at scale alpha/rank);
  the adapter's other six target families and its head bank apply at inference and are not
  read. The turbo bank also carries each adapter's slice by reference (`fl2va_adapter` /
  `ref2va_adapter`): the rows its tables were fused from, and the smallest derivation
  TensorFS admits from a granted source. The turbo checkpoint's config stamps the turbo plan
  digests (`plans.TURBO_PLAN_DIGESTS`).

It refuses before any read unless `pruned` carries table rows and no dynamic modulation
weights for both DiTs, `full` carries the exact modulation weights, and each adapter is one
component carrying the complete bf16 rank-64 slice. `MAX_TABLE_BYTES` is a per-output,
per-task ceiling; the turbo set uses 7.8 % of it beside the launch set's 95.0 %.

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

Two proofs cover the lane catalogue, neither needing a GPU, a rental or one weight byte.
`scripts/lane-proof.py` rebuilds the exact five-component source structure from this
package's own 638-row contract and the banked upstream safetensors HEADERS, then checks
every lane's declaration, the selection each treatment resolves to on the real components,
and every refusal. `../../scripts/h3-lane-store-proof.py` mints a tiny synthetic source in
a real TensorFS store, derives three lanes through the real Runtime `WeightsSink`, and
reads the committed headers back to prove that an untreated component keeps the source's
exact stored objects in every lane — the property that decides whether a per-component lane
is affordable at all.
`scripts/turbo-proof.py` re-derives every committed plan from the package's own composer
(no Diffusers), proves the turbo grid equals PDD's float64 block boundaries at float32, and
checks the fused kernel against the reference `LoRALinear.forward` at every plan row on a
tiny topology; `../../scripts/h3-turbo-store-proof.py` runs the exact `retable` orchestration over a
tiny `full`/`pruned`/adapter set in a real TensorFS store and reads all four outputs back.

The result preserves the Runtime quantizer's existing weight measurements in
`weight_fidelity_this_run`, a list of rows naming output slot, component, the treatment that
produced them and stats: worst per-tensor relative Frobenius error for the encoding and,
separately, for the cast — they measure different carriers against different bounds and a
maximum over both would hide which one moved — plus saturated element count, measured tensor
count and byte counters. Each
completed component also logs those same values before later stages run. Replayed outputs
are absent from this run's measurements; absence never means zero error. BF16 inheritance
and AdaLN table production do not claim quantizer measurements. Activation fidelity,
matched output distance and output quality require separate inference/evaluation evidence.
