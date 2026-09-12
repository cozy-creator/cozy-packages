# Private client scripts

Run a Python file directly with `cozy run ./path/to/script.py --rental-only`.
The client captures the script and its editable dependencies, then executes the
ordinary `main()` function on the private worker. These files are not published
packages and have no `package.toml`, App registry, or request/result classes.
Runtime 0.7.0 or newer is required for typed model/service
parameters; the dependency is captured with the script.

- `h3_checkpoint_repair.py` is the completed, guarded repair recipe for exactly
  four historical H3 roots. A corrected checkpoint is refused before mutation.
  Its `[tool.cozy.weights]` metadata declares the finite output budget; `main`
  receives the existing Model and WeightsSink capabilities and returns a real
  ModelArtifact. The script performs no automatic upload or release mutation.
- `h3_shared_vae_repair.py` repairs the four pinned rc2 revision 6 roots. It casts
  each selected video VAE decoder operand once and sends identical bytes to four
  native writers. TensorFS shares those objects; each lane inherits its existing
  DiT bytes and order, including FP8/MXFP8 parts and AdaLN tables. Only the legacy
  MXFP8 table metadata changes; existing explicit layouts remain byte-identical.
  The script returns four artifacts in underscore output slots, ready for explicit
  publication under the short lane names. It neither quantizes DiTs nor recomputes
  tables. Retained completed parts resume without source reads, and completed output
  transactions replay; a lost worker cannot supply its unretained local work.
- `h3_vae_roundtrip.py` generates a fixed reference image, uses the existing H3
  VAE encode/decode methods, stages DiT between them, and records resident tensor
  hashes and Cozy Eval grid observations. It returns a video plus source and
  reconstruction PNGs. Change `FRAMES` or `CYCLE` in the file to change the probe.

Supply an exact reviewed model with the ordinary `model.source=<ref>` (repair)
or `model.model=<ref>` (VAE) term. Optional PEP 723 `[tool.cozy.models]` entries
can set these defaults; explicit CLI terms take precedence. Imports and model
classes come from the captured library environment. Script-local classes and
annotation expressions with function calls are not capability declarations.

The VAE example captures the adjacent editable `minimax-h3` library through
`tool.uv.sources`. Its `h3_diagnostics`, `h3_activation_trace`, and
`h3_resident_samples` modules share H3's existing inference and component scopes.
The activation helpers remain ordinary library functions: observers wrap the
original denoise callback and never replace the sampler. Historical fingerprint
expectations identify their original checkpoint and are observations, not universal
requirements for later checkpoints.

Completed memoized operations and independently retained partial work can be reused
by a new script on the same worker. A script itself starts from `main()` each time;
plain Python lines are not replayed or skipped by a workflow engine.


## SDXL quantization assessment (se-042)

The quantization step calls `cozy_runtime.derive.operations.quantize` with the plain
`sdxl.operations.quantization_plan()` value. Runtime owns the single memoized
operation; SDXL and Anima helpers select components, and H3 supplies exact geometry
and canonical-order fingerprints. A caller or unrelated family edit that leaves
that plan unchanged does not change the shared quantization target.

`sdxl_fp8.py` composes native source/conversion/normalization/quantization with
24 fresh 1024² renders (eight prompts × candidate/reference/repeat), activation
capture, memoized weight/media/capture/quality measurements, a fresh Eval fold,
and explicit checkpoint/report/release effects. The local `sdxl-assessment` helper
is an unpublished Python library with no package.toml or deployment. Its local App
registers only the example calibration controls. This consumer requires the coordinated public
Runtime 0.14.2, TensorFS 0.3.36 and Eval 0.7.0 read-only weight cohort.

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
Both scripts and the editable helper carry locks over public Runtime 0.14.2, TensorFS
0.3.36 and Eval 0.7.0; only adjacent authored source remains editable. Keep
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

No SDXL render, model-quality verdict or release is claimed by these source changes.
The producer and full real-model qualification remain separate tracker evidence.
