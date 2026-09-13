This optional private diagnostic runs normal FL2VA Turbo preparation, text/keyframe
conditioning, and one scheduler step through the existing H3 entrypoint. It stops
before decoding. Its sole output is a bounded JSON trace, never a generated video.

It is useful only if full-video comparisons still differ after a candidate fix.
Compare the candidate and old code using the same prompt, seed, immutable base and
adapter checkpoints, worker software, and attention route.

From the packages worktree containing the candidate H3 source:

```sh
cp ./examples/client-scripts/h3-first-step/package.toml.example \
  ./examples/client-scripts/h3-first-step/package.toml
cozy package install ./examples/client-scripts/h3-first-step --editable --no-model-download
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

The project resolves its H3 dependency from the source in the same worktree.
Freeze that source before capture. Changing only an installed global H3 package
does not change this explicit local dependency. Its request and result schemas
are local, so static capture need not resolve foreign payload aliases.
The generated `package.toml` is local capture configuration. Keep it out of the
commit: this example is not a fifth published application in the package catalog.

Run `python scripts/h3-first-step-compare.py ONE_GPU.json TWO_GPU.json FOUR_GPU.json`
from the worktree to compare the retained JSON trace assets. The comparator
refuses different prompts, seeds, durations, checkpoint identities, coordinate
layouts, or trace boundaries. It reports equality of exact original-dtype hashes
and numeric sample errors. A differing hash with zero sample error means the
difference lies outside the small numeric sample.

Whole-tensor observations cover DiT inputs, the input to transformer block zero,
both gathered head outputs, and final DiT outputs. Intermediate observations
cover only global packed rows 0 through 7 on rank zero. The initial full-input
hashes include noise, text conditioning, timestep/position tensors and modality
indices. The hooks rely on the current H3 CP plan, which splits at block zero and
gathers both projection outputs before ordinary PyTorch post-hooks run.

The trace locates the first *observed boundary* that differs; it does not prove
the precise first arithmetic operation. Equal prefixes do not prove equality
elsewhere in the intermediate activation. If needed, expand the observation
around the first differing boundary instead of dumping all 52 layers.

Local checks: `scripts/h3-first-step-proof.py` verifies unchanged output/RNG and
hook restoration on a tiny real H3 forward, exact scalar/noncontiguous hashes,
and the explicit first-step stop. Runtime static interface extraction also
resolves both model bindings with degrees 2 and 4. These are supplementary;
actual Creator capture and full real-model execution must still be recorded.
