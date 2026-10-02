# Private client scripts

Run a Python file directly with `cozy run ./path/to/script.py --rental-only`.
The client captures the script and its editable dependencies, then executes the
ordinary `main()` function on the private worker. These files are not published
packages and have no `package.toml`, App registry, or request/result classes.
Current private capture and preparation require Runtime 0.16.10 or newer on
both the local client and private worker. The script captures its dependencies.


## Qwen reference-image preparation

Use the standard command to prepare a complete model on an existing rental:

```sh
cozy model upload \
  hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758 \
  paul/reference-image --rental=YOUR_RENTAL --await
```

Runtime 0.18.23 owns the reviewed download, native TensorFS conversion, and required
configuration, tokenizer and license preparation. Upload returns an immutable
checkpoint. Publish its release separately, using the returned checkpoint ID:

```sh
cozy model publish paul/reference-image --release 0.1.0 \
  --lane original=sha256:CHECKPOINT_DIGEST
```

`sdxl_prepare.py` converts any Civitai SDXL checkpoint into its served lanes (`fp16`, plus
the `fp8`/`mxfp8` package functions) and uploads them; `h3_lanes.py` does the same for every
H3 lane and the PDD-8 turbo LoRA. Set their constants (scripts take no scalar arguments), then:

```sh
cozy run examples/client-scripts/sdxl_prepare.py --rental=NAME --allow-publish org/model --await
```

Their downloads and conversions repeat `cozy model upload` of the same pinned sources, so on
the ingest rental they are memo hits. They are the upload path until Runtime-owned jobs accept
a `cozy run <fn> <in> <org/model>` destination. The SDXL conversion itself is servable: there
is no normalization step.

Edit the script and run it again. Each invocation starts at `main()`, while the
library operations look up compatible completed results and validate that their
bytes still exist. A caller-only edit does not invalidate their keys. Changed
operation inputs, parameters, plan or relevant implementation produce a new key.
The native download/conversion/quantization writers can also adopt compatible
partial work; this is explicit library support, not Python line replay. A Runtime
upgrade can conservatively invalidate more work than a caller edit.

The same Runtime/TensorFS code is used locally and on a private worker. Each Store's
`.cozy-workspace/journal.sqlite3` records operation identities, results, progress and
retention; TensorFS stores the artifact bytes. Reuse requires that Store and its
retained bytes. There is no need for a previous run ID or Tensorhub publication.
Inference runs fresh; checkpoint upload and release creation are explicit effects.

Scripts bind models with ordinary `model.<slot>=<ref>` terms. Optional PEP 723
`[tool.cozy.models]` entries can set these defaults; explicit CLI terms take precedence.
Imports and model classes come from the captured library environment. Script-local classes
and annotation expressions with function calls are not capability declarations.

Completed memoized operations and independently retained partial work can be reused
by a new script on the same worker. A script itself starts from `main()` each time;
plain Python lines are not replayed or skipped by a workflow engine.


## SDXL quantization assessment (se-042)

The quantization step calls `cozy_runtime.derive.operations.quantize` with the plain
`QuantizationPlan(components=("unet",))` value. Runtime owns the single memoized
operation; SDXL and Anima callers select components, and H3 supplies exact geometry
and canonical-order fingerprints. A caller or unrelated family edit that leaves
that plan unchanged does not change the shared quantization target.

`sdxl_fp8.py` composes native source/conversion/quantization with
24 fresh 1024² renders (eight prompts × candidate/reference/repeat), activation
capture, memoized weight/media/capture/quality measurements, a fresh Eval fold,
and explicit checkpoint/report/release effects. The local `sdxl-assessment` helper
is an unpublished Python library with no package.toml or deployment. Its local App
registers only the example calibration controls. This consumer requires the coordinated public
Runtime 0.16.10, TensorFS 0.3.38 and Eval 0.7.0 read-only weight cohort.

The committed policy is a **proposal, not ratification**. Read
[the v2 rationale and calibration requirements](sdxl-assessment/sdxl_assessment_client/policy/ABSOLUTE-PROPOSAL-v2.md)
before spending on model runs. Source/code edits retain normal memoization rules;
SDXL inference and its same-seed repeat are never memoized. Missing observations,
missing declared taps and unbound learned metrics cannot become a PASS. V2 explicitly
selects absolute activation budgets; the original v1 floor policy remains archived unchanged.

The script prepares the pinned Qwen 2B judge on its execution machine using ordinary
memoized source, conversion and normalization calls. It passes the genuine retained
ModelArtifact to quality measurements. Confirm disk capacity for its first source
download (approximately 4.25 GB) and converted model, and confirm the image's numerical
closure (NumPy 2.5.1, Torch 2.13.0 and torchvision 0.28.0 for the selected CUDA 13.0 image).
Both scripts and the editable helper carry locks over public Runtime 0.16.10, TensorFS
0.3.38 and Eval 0.7.0; only adjacent authored source remains editable. Keep
`PUBLISH=False` for calibration. Run via
`cozy run examples/client-scripts/sdxl_fp8.py --rental-only --await`; do not invoke
model functions with an external Python harness. The script returns one native Tree
bundle containing the report, workloads, conditions and review images, plus a manifest
of each file's digest and actual media type. Provisional/failed assessments retain the
same review bundle. WebP and PNG bytes keep their original encoding and suffix.

Publication additionally requires an explicitly ratified conditions digest,
destination, release, lane and revision expectation. The client checks this policy
before work when publication is requested, checks the complete report after the
measurements, uploads only the candidate, attaches its actual report/workload bytes,
checks the Hub verdict, then performs revision-guarded release publication. No
publication follows FAIL, INDETERMINATE or an unratified policy. The adapter does
not open a dummy weights transaction; Eval's weight leaf receives actual granted
manifest-only Models through Runtime's read-only WeightsReader service.

`sdxl_calibration.py` runs one explicitly selected calibration or held-out control
cell. Its separate prompt banks cover null, quantized and genuine native UNet ×2
weight arms, wrong-object/color generations and blur/flat pixel controls. See
[control evidence and review procedure](sdxl-assessment/sdxl_assessment_client/controls/README.md).
All expected labels remain authored and unreviewed; its private control jobs have a
local App under examples and are never added to the deployed SDXL App. The driver
returns one native Tree of actual review files/facts, never publishes, and refuses an unfrozen held-out
policy. Calibration, held-out and final-admission prompt/seed sets are disjoint.

Real SDXL source/conversion/normalization and a baseline render are qualified on
a retained private L40S; quantization interruption and edited-caller reuse are
also observed. Full candidate quality and release remain separate tracker gates.
These examples do not claim laptop-disconnected whole-script execution, which is
still being implemented.
