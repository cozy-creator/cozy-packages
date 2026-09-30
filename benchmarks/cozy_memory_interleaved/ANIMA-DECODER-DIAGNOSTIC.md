# Private Anima decoder diagnostic: CPU source checkpoint

Owner: `/root/decoder_diagnostic_completion`, tracker #295. Branch
`experiment/295-anima-blend-private-prep-20260930`, base `5001dcb42ca5d6c7731e6bbd86b70fc87144e67f`.

This review checkpoint commits the new diagnostic module, private identity and
generated AST interface, CPU fixtures, strict receipt consumer, and source proof.
The borrowed qualified author, assets, and pure blend projection remain intact in
the durable worktree. Their hashes and original source path are recorded in
`anima-decoder-source-manifest.json`; this PR is not a standalone built package.

The private VAE/model/pipeline are constructed before Runtime adoption. Normal
warm and the baseline full request use scalar blending. Two additional managed
decodes use the captured actual BF16 inverse-normalized argument, with immutable
scalar/broadcast selectors on one VAE. Six original cache bindings and all
diagnostic mode/capture/counter bindings retain public retry-state ownership.

Fourteen guarded CPU fixture groups pass, including actual upstream full1024
tiling with a shape-preserving CPU decoder double. Edge tiles truncate before
concatenation; the result is contiguous `[1,3,1,1024,1024]`, with 6291456 BF16
storage bytes. The unchanged guard still refuses unexpected CUDA layouts before
copying. The proof does not execute real model kernels or establish GPU layout.

The generated single-entrypoint interface declares exactly two raw files, two
RGB files, one combined 128 MiB trace, one 64 MiB image, one 1 MiB argument and
one 1 MiB report: 234 MiB total. The trace cap is an output-file limit; profiler
RAM is not bounded by it. The finite source inventory is baseline plus two
profile decodes; actual invocation/retry/closed-ledger evidence remains required.

Evidence lives at
`/home/fidika/.cozy/outputs/comfy-cozy-memory-20260929/analysis-anima-blend-private-prep-20260930/`.
`REPORT.md`, `OWNER.json`, `DIAGNOSTIC-SOURCE-FACTS.json`, and
`DIAGNOSTIC-CPU-CONTRACTS.json` are the durable handoff. CPU commands must use the
inspected `cpu-no-device.py` Landlock launcher, nice19, and at most two threads.

Remaining gates: root and independent exact-source/cohort review, actual Builder
artifact acceptance, ordinary CPU preflight and accepted Worker output/scope
grants, then explicit authorization for registration, installation, model
imports/weights, and GPU profiling. No release, activation, GPU query, or model
import occurred in this checkpoint. Static typing reports two existing errors in
the unchanged literal projected helper (`super().blend_v/h`); there are no
diagnostic-module typing errors in the pinned-environment check.
