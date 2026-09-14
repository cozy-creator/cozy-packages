# Native consumer release checks

Owner: Codex `/root/consumer_delivery_finish`; Runtime release integration remains
with `/root`. All consumer work uses the durable worktrees recorded in
`cr174-tensorfs-consumers.md`. No SDK or package publication is implied by a local
candidate build.

## Public dependency sequence

TensorFS 0.3.42 is public and must not be republished. After Runtime 0.18.0 is
genuinely public, update Eval's `uv.lock` from its 0.7.1 candidate metadata, check
the base install remains Runtime-free, and qualify its optional jobs environment.
Then merge/publish Eval 0.7.1 with official artifact readback. The core Eval 0.7.0
parser used by Runtime's capture checks needs no optional jobs migration.

After those public artifacts exist, select the next unused package versions from
current catalog/source state and regenerate these five package-repository locks:

- `anima/uv.lock`
- `sdxl/uv.lock`
- `minimax-h3/uv.lock`
- `minimax-h3-tools/uv.lock`
- `examples/client-scripts/sdxl-assessment/uv.lock`

The separate Eval lock makes six in this release sequence. Use ordinary `uv lock`
against public sources; do not encode local candidate wheels or fabricated hashes
into any published lock. Require `uv lock --check` for every updated project,
`scripts/torch_family.py`, package fences/types/lint, installed-wheel imports and
the existing package interface/conformance jobs. Recheck the four actual core
packages only; private examples are not catalog packages.

## SDXL client preflight and proof

`sdxl_prepare.py` composes Civitai source download, canonical conversion, SDXL
component normalization and generic Runtime quantization. `sdxl_fp8.py` adds an
explicitly prepared Qwen judge, independent reference/repeat/candidate generation,
memoized weight/media/activation/quality facts, the fresh policy fold, report
retention, and publication. Both ordinary `main(ctx)` signatures import under the
candidate cohort. The assessment client declares Eval jobs/weights/judge extras;
the execution image must contain its captured NumPy 2.5.1, Torch 2.13 and
Torchvision 0.28 family along with the model dependencies from the captured closure.

The existing user authorization allows the owner to choose an explicit engineering
policy before this prototype run. Set its exact conditions digest in the one-off
script; that field identifies the selected policy and needs no additional user
approval. Report the measured result with its experimental quality limits. The
designated destination is `paul/sdxl-memoization-proof`, release `0.1.0`, lane `fp8`.
The example defaults leave these fields unset, so the proof owner supplies them in
the new script. A candidate that fails the policy must not reach upload.

The old SDXL rental and its retained bytes are gone. Final qualification therefore
starts with a fresh source download on the explicitly owned machine. Use the
ordinary Creator CLI and its requested home, record installed wheel hashes and the
captured dependency closure, then require all of the following evidence:

- Native source/conversion/normalization/quantization outputs and exact digests.
- An interrupted operation resumed as a new attempt of the same request, with
  completed native parts retained; another ordinary run and a caller-only edit
  reuse eligible completed children as shown by their execution identities.
- Independent reference, repeat and candidate serving request IDs and real images;
  genuine activation captures and a retained judge measurement. Generation itself
  stays uncached. Static imports or small numeric controls cannot prove this.
- Current policy folded from retained facts, report bytes retained independently
  of the producer, and no upload on failure. On PASS, upload/assessment attachment
  precede release publication; read back the exact checkpoint/report/release tuple.
- Interrupted/retried upload and release preserve their accepted effect identity
  and observed outcome. Preserve the rented machine whenever it remains the sole
  retained copy; cleanup is a separate owner decision.

## Reviewed invariants

Candidate Runtime 806bf675 keys memoized model children by callee code/interface,
entrypoint, canonical inputs with exact model manifest identity, capture options
when present, and the numerical environment. Caller identity and model producer
custody are excluded. Byte/tree arguments enter as verified content identities.
Native result delivery acquires recipient retention before optional cache insertion,
and cancellation rechecks the accepted parent before reuse delivery. Consumer
inspection found no new defect in those paths; this is a bounded source review,
not proof of every retry/race arm. Existing targeted Runtime and final ordinary CLI
controls remain required.
