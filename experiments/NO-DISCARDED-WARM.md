# Private no-discarded Model.warm experiment

PR370 prepares two isolated project identities, not public-package/default changes:

| Candidate | Exact control |
|---|---|
| `sdxl-no-discarded-warm` | `sdxl-stable-vae` at37ff2b8ea76da39b8efed30224d04353d1f5f8e1 |
| `anima-no-discarded-warm` | `anima-stage-scopes` at0d76083c0a01d8cb33e44d3715f109903d384373 |

The only authored-code difference is each Model.warm body:

```python
def warm(self, ctx: Context) -> None:
    ctx.raise_if_cancelled()
    return
```

Lifecycle warm is excluded from Model's automatic method-scope wrapping. No empty
uses_components annotation is needed or legal. Runtime fill, capability qualification,
CUDA initialization, overhead accounting, import/warmup hooks and construction remain.
Stable VAE dtype behavior and Anima stage scopes remain exactly those of the controls.
All assets, request math, defaults, CFG/FBC, seed/RNG behavior in request code, tiling,
attention, dtype/cast behavior and dependency pins are preserved. The historical R19
Runtime wheel pin remains an input artifact, not a claim that it is the current R20
candidate: the benchmark coordinator must create matched control/candidate copies with
the same separately selected Runtime/TensorFS cohort. No pin or lock hash was relabeled.
Only each private project name changes coherently in pyproject.toml and uv.lock.

Each project has NO-WARM-PROVENANCE.json containing exact source head/path and every
copied baseline-file hash. FROZEN-SOURCE.json and README.md are retained historical
control records. The new provenance and this document describe the no-warm difference.

## Source and CPU gates

Four targeted guarded tests verify every baseline asset byte, exact whole-source
replacement of only warm's body, private identity-only pyproject/lock changes, and the
actual imported Model methods' cancellation semantics with component scopes set to
fail if entered. Current Runtime static builder control/candidate interfaces are byte
identical; retained metadata matches the current builder semantically.

The separately banked cold-first-failure proof is reused, not rerun or expanded into a
GPU claim: analysis-no-discarded-warm/test_cold_first_failure.py and qualification.json,
three guarded tiny upstream CPU cases. Faulting CLIP capture, Anima VAE cache and Cosmos
FBC calls run before any clean expected forward; actual Planner rollback restores owned
state and global/explicit RNG, and retry matches a fresh twin. Those tests do not prove
full checkpoint CUDA/native first-use safety, every tiled path or real GPU OOM recovery.

No inference fixtures were weakened, and copied tests remain available. No GPU run,
active install, publication or promotion is part of this preparation.

## Matched cold-first-output recipe (requires coordinator approval)

1. Finish the current6GiB baseline unchanged. Review source and CPU evidence independently.
2. Fix one Runtime/TensorFS/agent cohort and the same checkpoints, metadata, inputs and
   capability-cache state for both arms. Use fresh model executors; retain explicit
   records of warmed package environments/native caches. Do not delete shared caches.
3. Run full first requests in counterbalanced control→candidate and candidate→control
   order: unchanged1024²SDXL20steps/CFG7/HiDiffusionoff and Anima30steps with the same
   actual CFG interval/FBC/seed. No proxy request or reduced steps replaces the first output.
4. Follow with repeated requests and the same grouped sequence. Measure all startup,
   qualification/construction, first output and whole sequence wall time, transfer bytes,
   allocator/driver/host pressure peaks, retries, fixed modes and thermal reason flags.
5. Compare decoded RGB exactly against each matched control. Stop and preserve artifacts
   on first-output mismatch or cold OOM/retry-state mismatch; do not change modes to make
   the experiment pass. Separate a needed diagnostic from the scored comparison.
6. Do not subtract R20's5.062+11.211s discarded warm duration from the total as a forecast.
   Library/workspace first-use and learned demand can move into real inference. Promote
   only after actual cold-first-output and full-sequence evidence, ordinary CLI lifetime
   proof and the declared numerical/failure gates.
