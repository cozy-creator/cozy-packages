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

One attempt emits exactly four dual-task checkpoints:

- `bf16-full`
- `bf16-adaln-pruned`
- `fp8-adaln-pruned`
- `mxfp8-adaln-pruned`

The v2 standalone `assemble_full`, `assemble_dual`, and two timestep-table jobs remain
available for direct use. `four-lane` is available in the current v2 line, but it does not invoke or nest
those jobs; it owns the same transformations directly inside one weight-production attempt.

Each lane contains both FL2VA and Ref2VA DiTs plus the shared 902/703/1,087 components.
Full assembly drops the native-only `rope.inv_freq` buffer, leaving the exact 638-row
Diffusers DiTs. It also drops the official text model's layers 50–63, final norm, and
language-model head—156 inherited refs—to produce the reviewed 902-row, 50-layer
pre-norm conditioner without rewriting any retained payload. Each pruned task then
replaces 106 dynamic AdaLN rows with 51 BF16
timestep-table rows, adding exactly 288,347,136 table bytes. FP8 and MXFP8 are
independent children of those pruned BF16 task components and never derive from each
other.

The package declares no GPU, SM, VRAM, or host-RAM guess. Creator derives accelerator-class work
from the typed model inputs; exact artifact residency and measured request/scratch envelopes drive
fit. The job opens the three pruned-output transactions together, computes each task's timestep tables once,
and writes the exact same table bytes to every active output. FP8 and MXFP8 then read the
original BF16 DiTs independently through their own caller-owned transactions. There is no
workflow graph and no nested job invocation. Tensorhub owns final retention or publication;
the job receives no publisher credential or store path.

The production inputs are package-owned and immutable: exact model config, full DiT shape
contract, construction order, and task plans live as importlib resources in the tools wheel;
the shared `cozy-jobs` wheel owns the exact H3 quantization plan used here and by the standalone
quantization callables. They are not Creator-supplied assets. Run
`scripts/order-proof.py` to recheck their closed census and
`../../proofs/producer-callable.py` to validate the generated graph-free descriptor.
