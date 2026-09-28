# minimax-h3-tools

The H3 producer package. Every conversion is an ordinary package function with one primary
model input and a destination repository, run on the rental that holds the source:

```sh
cozy run <org>/minimax-h3-tools/<function> <input model ref> <org/model> --rental=NAME
```

Each declared output becomes a checkpoint named by its slot; publish lanes with
`cozy model publish <org/model> --release R --lane <slot>=<checkpoint>`. On the rental that
ingested or already fetched the input, its objects are in the pod Store and nothing moves.

| Function | Input (first positional) | Output | Notes |
|---|---|---|---|
| `bf16-full` | a full-precision H3 checkpoint: the converted `MiniMaxAI/MiniMax-H3` release or a `bf16-full` lane | `bf16-full` | source-only rows dropped, video VAE decode operands f16 |
| `bf16-pruned` | same | `bf16-pruned` | AdaLN tables replace the 106 modulation rows per DiT; CUDA |
| `fp8-pruned` | same | `fp8-pruned` | tables + both DiTs `fp8-rowwise/1`; CUDA; every serving rung |
| `mxfp8-pruned` | same | `mxfp8-pruned` | grandfathered; native only on sm120 |
| `turbo-lora` | the converted `alibaba-pai/MiniMax-H3-Acc-LoRAs` release (one fl2va and one ref2va component) | `pdd8` | `model.base=<full-precision H3>`; reads its modulation weights and heads only |
| `retable` | an AdaLN-pruned lane | `adaln-pruned`, `tables` | `model.full=<full-precision H3>`; recomputes tables only |

Runtime requires a receipt for every declared weights output, so each lane is its own
function. Until Runtime-owned jobs accept a destination, `examples/client-scripts/h3_lanes.py`
runs these functions on the ingest rental and uploads every lane and `pdd8`.

```sh
cozy run fidika/minimax-h3-tools/fp8-pruned fidika/minimax-h3@1.0.0/bf16-full fidika/minimax-h3 \
  --rental=NAME --await
cozy run fidika/minimax-h3-tools/turbo-lora fidika/minimax-h3-acc-loras@1.0.0/original \
  fidika/minimax-h3-turbo-lora model.base=fidika/minimax-h3@1.0.0/bf16-full --rental=NAME --await
```

The lane functions, `turbo-lora` and `retable` are memoized operations. Tables, heads and quantized
tensors checkpoint as they complete; a request re-issued with the same inputs on the same
worker adopts a retained stopped run's completed work (`cozy run pause <run>` retains it) and
computes only the remainder. Quantized tensors encode on every available CPU. The metrics
`h3.reused_tensors`/`h3.computed_tensors` (`h3.turbo.*` for the LoRA) report the split.

The lane functions read the granted checkpoint's structure once and drop only source-only
rows still present (the native `rope.inv_freq` buffer and the text model's layers 50–63,
final norm and head), so the converted upstream release and `bf16-full` produce the same
lanes. It does not accept a pruned or quantized checkpoint as a substitute for the full source.

Pruned checkpoint configs describe the ordered AdaLN rows as `cozy_h3.table_keys`:
each final-normalization row names an exact float32 timestep, and each block-modulation
row names a timestep and modality. Sampling-plan hashes remain generation provenance;
they do not decide serving compatibility. Producer and inference use byte-identical
copies of the same small row-label parser, checked by CI.

## Lanes

A lane is a NAME plus, per component, what this producer does to it. The whole catalogue
lives in `h3_tables/lanes.py`; the request names which of its rows this attempt produces
and defaults to all of them. The recipes are code, never a request field: output slot names
are decorator-time facts, and every lane mints a MiniMax H3 Model Derivative under §I.11(i)
of the community licence, which is a reviewed act rather than a caller choice.

| lane | modulation | per-component treatment |
|---|---|---|
| `bf16-full` | full | none — every component inherited |
| `bf16-pruned` | AdaLN-pruned | none |
| `fp8-pruned` | AdaLN-pruned | both DiTs `fp8-rowwise/1` |
| `mxfp8-pruned` | AdaLN-pruned | both DiTs `mxfp8/1` |

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
`mxfp8-pruned` is grandfathered by name and nothing else may join it. **The
`audio_vae`, by name**: 637 of its 1,087 rows are rank-3 — including the BigVGAN decoder's
344 `weight_norm` `weight_g`/`weight_v` parameters — and the rank-2 encoding cannot
represent any of them; the only six rank-2 float weights the component carries are the
`pre_block` ENCODER attention/MLP linears — so a shape rule does not refuse
the component, it silently quantizes the audio conditioning path and leaves the BigVGAN
decoder untouched.

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
the Runtime wheel carries the reviewed quantization tensor specifications. The ordinary
`h3_tables.operations.quantization_plan()` helper binds their exact two-DiT geometry and
this package's full/pruned construction-order digests for the single Runtime-owned
`cozy_runtime.derive.operations.quantize` operation. Its compact plan contains no
model bytes or executable policy. Noncanonical source order refuses; the lane
functions and AdaLN producers already emit the accepted orders. These are not
Creator-supplied assets. Run
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


`turbo-lora` builds the PDD-8 overlays from the released
`MiniMax-H3-FL2VA-Acc-8Step.safetensors` and `MiniMax-H3-Ref2VA-Acc-8Step.safetensors`
(`alibaba-pai/MiniMax-H3-Acc-LoRAs`, revision `335001fb9e5455d68a0caa18ec2e319072150328`)
and the base model's original modulation weights. The adapter checkpoint must hold exactly two
components whose names contain `fl2va` and `ref2va`. `build_turbo_adapter` composes the same
producer from two single-component adapter checkpoints in a client script.

The output holds only the `fl2va_turbo` and `ref2va_turbo` components. Their six inference LoRA families inherit
source objects. Only adapted AdaLN tables and eight collapsed output heads are
written. The consumed AdaLN factors are absent from the output. One native transaction
checkpoints each completed tensor, so cancellation resumes unfinished work and a
completed replay returns the same artifact. Ordinary `retable` keeps its two inputs
and two outputs; turbo preparation does not change the base schedule.

The output binds to `minimax-h3`'s single `H3TurboModel` slot. Ordinary `H3Model`
continues to consume a checkpoint with only the original five components.

The row-label parser is Runtime's `cozy_runtime.models.minimax_h3.table_layout`;
the producer imports it rather than keeping a copy.

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
