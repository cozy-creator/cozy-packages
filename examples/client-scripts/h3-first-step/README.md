This optional private diagnostic runs normal FL2VA Turbo preparation, text/keyframe
conditioning, and one scheduler step through the existing H3 entrypoint. It stops
before decoding. It saves a bounded JSON trace and original-dtype refiner tensors,
never a generated video.

It is useful only if full-video comparisons still differ after a candidate fix.
Compare the candidate and old code using the same prompt, seed, immutable base and
adapter checkpoints, worker software, and attention route.

Generate the diagnostic from the exact captured H3 wheel being investigated:

```sh
python ./examples/client-scripts/h3-first-step/prepare_h3_first_step.py \
  /absolute/path/to/minimax_h3.whl /absolute/path/to/probe/project
uv lock --project /absolute/path/to/probe/project
cozy package install /absolute/path/to/probe/project --editable --no-model-download
cozy run local/h3-first-step/probe \
  'prompt=A woman sings beside a pianist in a softly lit jazz club.' \
  seed=7101 duration_s=5 \
  model.base_model=paul/minimax-h3@1.0.0-h3-audit.1/fp8-adaln-pruned \
  model.turbo_lora=paul/minimax-h3-turbo-lora@1.0.0-audit.1/pdd8 \
  --rental=OWNED_RENTAL --await --out=./first-step-output
```

Read back both aliases before submitting: the audited base is
`sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275`
and the adapter is
`sha256:9921c17df2db67e8350769c490249b43659603ad780a67fc713d847a2ff194b0`.
The example uses an existing owned rental and never acquires a machine. Use the
same ordinary per-request GPU selection as the full-generation qualification.

The generator copies unchanged H3 source into a separate private project beside
the diagnostic module, preserves the wheel's dependency ranges, and records
digests outside the project. Actual Creator capture refused a separate App
importing foreign H3 model classes, despite isolated static extraction passing
with an explicit dependency environment. Keeping definitions inside the captured
source boundary succeeded on real H100s.

Freeze the generated project before submission and store logs outside it. No
production package pin changes; this is not an additional catalog application.

Run `python scripts/h3-first-step-compare.py ONE_GPU.json TWO_GPU.json FOUR_GPU.json`
from the worktree to compare the retained JSON trace assets. The comparator
refuses different prompts, seeds, durations, checkpoint identities, coordinate
layouts, or trace boundaries. It reports equality of exact original-dtype hashes
and numeric sample errors. A differing hash with zero sample error means the
difference lies outside the small numeric sample.

Whole-tensor observations cover DiT inputs, input projections, the token refiner,
its first attention Q/K/V and output projections, the input to transformer block
zero, both gathered head outputs, and final DiT outputs. Intermediate observations
cover only global packed rows 0 through 7 on rank zero. The initial full-input
hashes include noise, text conditioning, timestep/position tensors and modality
indices. The hooks rely on the current H3 CP plan, which splits at block zero and
gathers both projection outputs before ordinary PyTorch post-hooks run.

The safetensors output retains text/refiner input and output, first-refiner
attention input/output, Q/K/V, and the attention result before output projection.
Each tensor is limited to two MiB and the combined output to nine MiB; this
bounded probe is intended for short prompts.

The trace locates the first *observed boundary* that differs; it does not prove
the precise first arithmetic operation. Equal prefixes do not prove equality
elsewhere in the intermediate activation. If needed, expand the observation
around the first differing boundary instead of dumping all 52 layers.

Local checks: `scripts/h3-first-step-proof.py` verifies unchanged output/RNG and
hook restoration on a tiny real H3 forward, exact scalar/noncontiguous hashes,
and the explicit first-step stop. Runtime static interface extraction also
resolves both model bindings with degrees 2 and 4. These are supplementary.
The source-complete diagnostic was also executed through the ordinary CLI on
real H100s and identified the first-refiner attention result as the first
differing boundary. Always record actual capture and execution for the exact
candidate being investigated.
