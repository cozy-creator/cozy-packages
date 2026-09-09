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

`sdxl_fp8.py` composes native source/conversion/normalization/quantization with
24 fresh1024² renders (eight prompts × candidate/reference/repeat), activation
capture, memoized weight/media/capture/quality measurements, a fresh Eval fold,
and explicit checkpoint/report/release effects. The local `sdxl-assessment` helper
is an ordinary unpublished Python library, with no App or package.toml; no new
deployable is introduced. This consumer requires the coordinated public
Runtime0.14.0, TensorFS0.3.34 and Eval0.7.0 read-only weight cohort.

The committed policy is a **proposal, not ratification**. Read
[its rationale and calibration requirements](sdxl-assessment/sdxl_assessment_client/policy/README.md)
before spending on model runs. Source/code edits retain normal memoization rules;
SDXL inference and its same-seed repeat are never memoized. Missing observations,
zero repeat floors and unbound learned metrics cannot become a PASS.

Set the exact owner-held `JUDGE` checkpoint and confirm the image's numerical
closure (the example pins NumPy2.5.1). Keep `PUBLISH=False` for calibration. Run via
`cozy run examples/client-scripts/sdxl_fp8.py --rental-only --await`; do not invoke
model functions with an external Python harness. The script returns completed
native report/workload files even for a provisional/failed assessment.

Publication additionally requires an explicitly ratified conditions digest,
destination, release, lane and revision expectation. The client checks this policy
before work when publication is requested, checks the complete report after the
measurements, uploads only the candidate, attaches its actual report/workload bytes,
checks the Hub verdict, then performs revision-guarded release publication. No
publication follows FAIL, INDETERMINATE or an unratified policy. The adapter does
not open a dummy weights transaction; Eval's weight leaf receives actual granted
manifest-only Models through Runtime's read-only WeightsReader service.

No SDXL render, model-quality verdict or release is claimed by these source changes.
The producer and full real-model qualification remain separate tracker evidence.
