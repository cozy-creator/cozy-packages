# SDXL calibration and held-out controls

These inputs are authored expectations, not reviewed images or independent human
judgments. None has been rendered by this source change. They are disjoint from each
other and from the eight final-admission prompts/seeds in `policy/workloads.json`.
The v1 and v2 proposed policy artifacts remain unchanged.

Run one bounded cell at a time by editing `SPLIT` and `CASE` in
`examples/client-scripts/sdxl_calibration.py`, then use the ordinary local Creator CLI:

```sh
cozy run examples/client-scripts/sdxl_calibration.py --rental-only --await
```

The editable helper is captured automatically; no package publication or manual
installation is part of this workflow. It has one local App to register private control
jobs and no package.toml or deployment. Creator's pyproject-only library capture fix
must be present, together with the helper's pinned Runtime 0.16.10 / TensorFS 0.3.38 /
Eval 0.7.

`null` performs eight prompts × two fresh renders of the same reference checkpoint.
Each pair must have distinct request IDs and the same observed execution environment.
It retains activation-pair, media, image-pair and Qwen checklist facts plus both review
images. These are repeatability observations for review, with no assessment verdict or
publication permission. Eval's paired checkpoint assessment requires distinct checkpoint
identities; manufacturing a different identity would not create a valid null control.

`quantized` and `scale_x2` each perform eight prompts × three fresh arms on the same
worker. Quantized uses the real quantizer. Scale-x2 makes an intentionally damaged
native checkpoint by doubling
every normalized plain F16 UNet tensor, retaining exact other components/configs/order.
Its genuine source-backed writer checkpoints each completed part; it refuses nonfinite
source values and F16 overflow. A zero tensor stays zero, so the weight evidence must
be read before claiming that each tensor changed. The operation is private example code,
outside SDXL's production App. Model generation is never memoized.

`wrong_object` and `wrong_color` each generate eight fresh baseline images plus eight
fresh images from the separately recorded counterfactual prompts. Eval scores each
counterfactual image against the original prompt/checklist, while the evidence names both
prompts and actual request IDs. A model may fail to follow either prompt: the expected
labels are hypotheses until the actual images are reviewed. These controls do not claim
to be paired checkpoint assessments.

`blur` and `flat` each generate eight fresh baseline images and derive eight retained
pixel controls. Blur uses Pillow Gaussian blur radius 8 pixels; flat uses RGB(127,127,127).
The control records its input digest, transformation and output digest. Native media/pair
measurements and explicitly bound Qwen checklist facts are retained. These transformations
are not substitutes for the actual ×2-weight damage arm.

Each cell returns one native Tree bundle with its evidence manifest, exact control policy,
review image files and measured facts or complete assessment report/workload. A separate
bundle manifest records filenames, exact digests and media types. The evidence
contains exact source/candidate/judge receipts, actual request/environment identities,
the base proposed policy digest and the split-specific policy digest. Missing learned
metrics and unknown saturation remain unmeasured. There are no release effects or
automatic threshold fitting, and every label says `authored_expectation_unreviewed`.
If a checkpoint control fails during managed execution or capture comparison, its
manifest records an incomplete control, the actual failure/request identity and any
completed review images. That is not a complete measured FAIL report. Preparation or
other unexpected failures remain normal CLI failed requests for investigation.

Start with calibration cells. A reviewer must inspect the returned images without
treating expected labels as ground truth, record their own judgments, and compare the
judge against those reviewed positives/negatives. Inspect null floors, per-tap sketch
coverage, population imaging changes and the true weight-damage arm. A broken null or
misread image is an observation to investigate, not a reason to silently widen budgets.

Freeze a new versioned Conditions file if calibration changes the numerical proposal;
select it explicitly using `CONDITIONS_FILE`. Do not overwrite v1/v2 or fit the candidate's
own admission thresholds. Before held-out execution, set `FROZEN_HELD_OUT_POLICY` to
that split's exact conditions digest; the selection proof prints the current proposal's
digest without running a model. This guard records preselection, not ratification.
Held-out cells run the same seven cases with their separate eight prompts/seeds. Review
those results before adopting any final-admission policy. Only then may the separate
`sdxl_fp8.py` final prompt set and its explicit publication guard be considered.

Preparation is normal Python composition of memoized native operations on the execution
machine. The first cell needs disk for the pinned Qwen source/converted instrument, SDXL
source/reference/candidate and review images; ×2 adds up to 16 GiB of new native parts
(actual normalized UNet size is recorded by the writer). Subsequent cells on that worker
can reuse completed preparation and immutable measurements. No paid run starts from
this documentation; the rental owner selects the machine and execution order.
