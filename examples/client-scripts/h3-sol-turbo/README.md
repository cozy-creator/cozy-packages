# Sol with H3 Turbo and Ulysses

This private qualification package exposes an ordinary Python `main()` serving
entrypoint. It binds the existing H3 base checkpoint and independent PDD adapter,
runs the official eight-step workflow, and reuses H3's decoding/output checks.
It contains no attention kernel or sampling implementation. There are no optional
Wushu/Spatial adapters in this comparison.

`sol_dense_steps=3` keeps the first three steps dense; `4` keeps four dense; `8`
is an all-dense control. The first two transformer blocks, refiner, and protected
conditioning/audio tokens retain Runtime's dense policy. Select the backend
independently through `--attention-kernel`. A dense-step setting alone does not
select Sol. Inspect effective backend counts before calling a clip sparse.

## Capture and worker requirements

This helper needs Runtime PR531's `sol_dense_steps` model method and the separate
Sol Ulysses integration. Released Runtime 0.18.2 does not contain these APIs.
The dependency floor is not a qualification claim: stage a private copy with
`tool.uv.sources.cozy-runtime` pointing to the exact reviewed development wheel.
Capture and the remote worker must use the same source/wheel.

The `minimax-h3` dependency must be the migrated workflow importing Runtime's
model classes (Packages PR220/221). The prior qualified workflow is at Packages
commit `9798fa332736b4fe8fe8019f58bf4241c88a6438`, directory `minimax-h3`.
Point the staged copy's `tool.uv.sources.minimax-h3` to that exact local package.
The helper refuses an older workflow with separate model classes. Do not copy
model inference back into this package to accommodate an old Runtime.

For the qualified experimental-wire55 worker family, keep its Host, protocol,
native extension, TensorFS 0.3.42, Torch 2.13.0+cu130, torchvision 0.28.0+cu130,
Diffusers, CuTe and Triton pins. Add only the reviewed Python Sol/CP/policy fixes
onto Runtime `8cd889922e04695d3127252ba5990c3c0b9d3bc9`; do not replace all of
`official.py` with modern source because the paid branch carries additional
physical-weight validation. The existing optional-kernel image is
`sha256:f54ce14ec5aafd864ab136940d44ca44967da0d104e3d772cec6c4b2526fbd5f`.
Build/register a new passive child image containing the fresh dev wheel before
renting, rather than paying while the environment is built.

Keep capture default groups empty; no new Ruff, PEFT, TorchAO or conversion
dependency is needed by this experiment. Existing image-owned libraries retain
their original compatible pins. Never replace the shared CLI with a modern
wire57 build as part of this attention experiment.

## Ordinary CLI

Install the staged package, then render through normal `cozy run`:

```sh
uv sync --no-dev --no-default-groups --project /absolute/path/to/staged/h3-sol-turbo
cozy package install /absolute/path/to/staged/h3-sol-turbo --editable --no-model-download
cozy run local/h3-sol-turbo/main \
  model.base_model=paul/minimax-h3@1.0.0-h3-audit.1/fp8-adaln-pruned \
  model.turbo_lora=paul/minimax-h3-turbo-lora@1.0.0-audit.1/pdd8 \
  --in=/absolute/path/to/fight15.json \
  --attention-kernel=fl2va_dit=sol-attn \
  --rental=<owned-rental> --idempotency-key=<unique-arm-key> \
  --await --json --out=/absolute/path/to/arm-output
```

Make separate input files with `sol_dense_steps` set to 3, 4 and 8; hold every
other field fixed. Use `flash-attn3` instead of `sol-attn` for the independent
dense backend control. The component-only override is intentional: the private
wire55 CLI cohort predates the complete model-qualified override fix. Only the
base model has `fl2va_dit`; the separately bound PDD adapter does not.

Keep capture logs and generated comparison files outside the staged package
directory. Creator correctly refuses an editable tree that changes during capture.

A standalone `cozy run script.py` always becomes a job in the current CLI.
Jobs cannot use a Ulysses serving group. This is why this helper uses
`App.entrypoint` and a local package, while keeping its implementation a normal
Python `main()` function.

The installed CLI selects the whole rented GPU group. It has no per-request
`--gpus` flag. A no-CP model declaration is rejected on a multi-GPU rental; it
does not select one of its cards. Thus ordinary CLI cross-degree comparisons
currently need separate 1-, 2- and 4-GPU rentals. Root records ownership and
explicitly releases each rental after its results are retained.

## Qualification order

1. Before renting: format/type/static-interface checks against the paired SDK;
   ordinary CLI capture; immutable base/PDD header resolution; image readback.
2. On an owned multi-GPU worker: the Runtime agent's bounded NCCL primitive
   tests, including uneven sequence lengths and protected prefixes. Record
   per-rank repeatability, padding exclusion and degree-one reference error.
   These tests supplement ordinary CLI proof; they are not videos.
3. One 15-second Turbo clip per arm on the same rental: FA3 dense, Sol dense8,
   Sol dense4, Sol dense3. Record cold preparation/JIT separately from denoising
   and total execution time. Dense8 must contain zero sparse calls. Compare
   both numerical digests and the final visual/audio outputs.
4. Repeat the best quality candidate and dense control at the other GPU
   degrees. Use actual effective per-rank counts and output digests; do not
   promise cross-degree byte identity merely from a fixed random seed.
5. Run a regular 30-step FA3/Sol Ulysses pair. Its first ten steps remain dense;
   a Turbo policy must not alter standard generation or the next request.

For eight evaluations, 50 main blocks and two dense refiner paths, expected
per-rank Sol counts are:

| Dense steps | Sparse | Protected dense prefix | Dense step | Dense path |
| --- | ---: | ---: | ---: | ---: |
| 3 | 240 | 240 | 144 | 32 |
| 4 | 192 | 192 | 192 | 32 |
| 8 | 0 | 0 | 384 | 32 |

Do not sum duplicated telemetry snapshots, or silently compare all-rank totals
with single-rank counts. Keep Sol opt-in until actual clips establish the chosen
quality/performance policy. A five-second smoke does not qualify the requested
15-second fight scene.
