# SDXL absolute activation proposal v2

Status: **PROPOSED, NOT RATIFIED**. This proposal was authored before trained SDXL
measurements. The v1 proposal and all its files remain unchanged. Its zero-repeat-floor
limitation is recorded in ev-016; v2 is a new policy, not a reinterpretation of v1.

`conditions.absolute.proposed.v2.json` uses `cozy-eval/gate-config@3` and has digest
`sha256:5f28766b07b4e9eb149a4915065a346b1accbd5a01786c4e65230115d2feb12e`.
The eight prompts, seeds, steps, capture roster and authored checklists are unchanged.
`proposal.v2.json` records those exact input file digests and the preceding policy.

The explicit activation budgets are worst binding-step sketch relative L2 ≤ 0.05 and
cosine distance ≤ 0.01, at every declared UNet tap across every prompt. They are
engineering hypotheses, with no observed SDXL calibration yet. Missing selected
statistics/taps refuse; candidate NaN/Inf fails. Sketch statistics are observations of
the captured projection, not full activation-tensor norms. No limit is asserted for
an unavailable metric.

The repeat still executes with the same seed, and its measured floors remain in the
report. Absolute mode uses the declared ceilings even when that floor is exactly zero;
it never inserts an epsilon or switches policy based on the observed result. The
existing ratio mode and its `no_floor` result remain unchanged for policies selecting it.
The old ratio caps and PSNR floor budgets are absent from this proposal. Pair distances
remain measured observations; v2 uses the existing image population budgets and quality
budget from v1, still explicitly proposed. Frobenius remains ≤ 0.05; saturation remains
unobserved and is not given a budget input.

Before adoption, the owner must review independently labeled positive/negative judge
controls, same-machine unquantized null/repeat controls, a genuine ×2-weight-scale
negative arm, and separate held-out or leave-one-out image controls. This evidence must
test both selected sketch metrics at every tap, the population imaging measures and
the checklist judge. Record actual checkpoints, exact software/image/driver closure,
workload and report digests, and all unavailable measurements. Preparing the judge
successfully does not calibrate it. A synthetic numeric decision control is not a
trained-model quality control.

If those controls require another bound or different prompt set, freeze a new policy
version before its held-out assessment. Retain earlier failures and provenance; do not
tune a candidate's own admission bands from that candidate. Only an explicitly reviewed
conditions digest plus a complete passing held-out report may enable publication.
The committed script keeps `PUBLISH=False` and `APPROVED_CONDITIONS` empty.

Judge preparation uses the owner's already qualified Qwen 2B source revision, profile,
metadata file list and native normalizer on this execution machine. First use on a new
worker requires approximately 4.25 GB of source download plus converted storage;
normal operation memoization may reuse completed inputs on later runs on that worker.
It produces a genuine ModelArtifact and does not depend on a fabricated receipt, a
repository string, or an uncalibrated instrument release. The owner's uploaded Qwen
checkpoint remains independent. Direct authenticated adoption of an unreleased Hub
checkpoint is a future capability gap, not introduced by this assessment consumer.
