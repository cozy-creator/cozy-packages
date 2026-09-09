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

Pruned checkpoint configs describe the ordered AdaLN rows as `cozy_h3.table_keys`:
each final-normalization row names an exact float32 timestep, and each block-modulation
row names a timestep and modality. Sampling-plan hashes remain generation provenance;
they do not decide serving compatibility. Producer and inference use byte-identical
copies of the same small row-label parser, checked by CI.

`restamp` preserves explicit valid row labels and validates them against the stored table
dimensions. For the original checkpoint stamps it first proves the old 345-frame plan
has identical ordered rows, then replaces its legacy digest with those labels. Unknown
or changed historical plans require regenerating the tables. This metadata upgrade
inherits every tensor object and does not normalize video-VAE precision.

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

Two proofs cover the lane catalogue, neither needing a GPU, a rental or one weight byte.
`scripts/lane-proof.py` rebuilds the exact five-component source structure from this
package's own 638-row contract and the banked upstream safetensors HEADERS, then checks
every lane's declaration, the selection each treatment resolves to on the real components,
and every refusal. `../../scripts/h3-lane-store-proof.py` mints a tiny synthetic source in
a real TensorFS store, derives three lanes through the real Runtime `WeightsSink`, and
reads the committed headers back to prove that an untreated component keeps the source's
exact stored objects in every lane — the property that decides whether a per-component lane
is affordable at all.

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


`h3_tables.turbo.prepare_turbo` adds the PDD-8 overlays to one existing AdaLN-pruned
model. It is a memoized Python operation and the `prepare-turbo` job. Pass `source`
(the BF16, FP8 or MXFP8 body being served), `full` (its original full-precision H3
model), and `fl2va_adapter` / `ref2va_adapter` (the two native single-component PDD
artifacts). The released PDD inputs are `MiniMax-H3-FL2VA-Acc-8Step.safetensors` and
`MiniMax-H3-Ref2VA-Acc-8Step.safetensors` from `alibaba-pai/MiniMax-H3-Acc-LoRAs`,
revision `335001fb9e5455d68a0caa18ec2e319072150328`.

```python
from h3_tables.turbo import prepare_turbo

model = await prepare_turbo(
    source=body, full=full, fl2va_adapter=fl2va_pdd, ref2va_adapter=ref2va_pdd,
)
```

The single output retains every base tensor and its construction config, and adds
`fl2va_turbo` and `ref2va_turbo` components. Their six inference LoRA families inherit
source objects. Only adapted AdaLN tables and eight collapsed output heads are
written. The consumed AdaLN factors are absent from the output. One native transaction
checkpoints each completed tensor, so cancellation resumes unfinished work and a
completed replay returns the same artifact. Ordinary `retable` keeps its two inputs
and two outputs; turbo preparation does not change the base schedule.

The output binds to `minimax-h3`'s single `H3TurboModel` slot. Ordinary `H3Model`
continues to consume a checkpoint with only the original five components.

Old `1.0.0-rc.2` AdaLN checkpoints may still carry the retired `frames:345` plan
identity. Before turbo preparation or current ordinary serving, run the `restamp`
job through Creator with `model.lane=paul/minimax-h3@1.0.0-rc.2/fp8-adaln-pruned`.
It resolves only the exact historical frame-only plan or the current known plan,
verifies table geometry, and replaces their opaque stamps with explicit ordered
table-row labels. Already explicit valid layouts are preserved. All tensor objects,
including the video VAE's precision, are inherited unchanged. Other old plans
require real retabling from their generating model; `restamp` refuses them.

The small row-label parser is maintained in `minimax-h3/h3_table_layout.py` and
copied byte-for-byte into the producer. `scripts/sync-h3-table-layout.py` verifies
the copy in CI, avoiding a private package-index dependency for pure validation.

`scripts/h3-turbo-store-proof.py` verifies native inheritance, cancellation, replay,
and component contents. With the pinned upstream `minimax_h3_pdd.py` supplied as an
argument under the H3 environment, it also loads the emitted tensors into the serving
overlay constructor and compares stored modulation tables and all eight output heads
against the actual upstream adapter over Diffusers. These are CPU construction and
numerical checks. They do not measure GPU speed, audio, or video quality.


The tensor-only serving hooks require Runtime's preservation of hooks during encoded
leaf installation and its refusal to prequantize a hooked consumer's operand. Publish
the serving package only with that Runtime fix; the producer itself uses released
Runtime 0.12 APIs.
