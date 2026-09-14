# Matched H3 attention videos

This private project calls the unchanged published H3 1.14.3 FL2VA function for a full standard30-step video. Choose `fa3_bf16`, `fa3_tile128_fp8` (the corrected two-level PV accumulation candidate with BlockN128), or `sage2_sm90`. It changes only the50 main FL2VA DiT attention blocks; token-refiner, text, VAE, numerical checks, media encoding and normal outputs remain on the original path. It makes no production-default changes and does not qualify multiple GPUs.

Prepare with the exact released H3 wheel and the reviewed tile128 oracle project:

```sh
python3 prepare_h3_kernel_quality.py \
  /home/fidika/cozy_v2/outputs/h3-turbo-determinism-20260913/release/published-minimax_h3-1.14.3-py3-none-any.whl \
  /home/fidika/cozy_v2/outputs/h3-attention-oracle-20260914/project-tile128 \
  /home/fidika/cozy_v2/outputs/h3-kernel-quality-20260914/project
```

The destination must be empty. The preparer retains every serving source/data byte from the wheel, copies the existing attention helpers without edits, and uses a relative local dependency for the exact private CUDA candidate wheel. Sage2 must already be available on the selected Hopper worker. The tile128 wheel was built for Torch2.13; do not use it in an unqualified Torch2.14 image.

Install that explicit local directory with `cozy package install --no-model-download /home/fidika/cozy_v2/outputs/h3-kernel-quality-20260914/project`. Then use its installed `local/h3-kernel-quality/generate` target through the ordinary default-home CLI on an already owned single-H100 rental, pinning BF16 FA3 for preparation and all untouched attention:

```sh
cozy run local/h3-kernel-quality/generate \
  model.model=paul/minimax-h3@1.0.0-h3-audit.1/fp8-adaln-pruned \
  --rental=NAME --attention-kernel=flash-attn3 \
  --in=/absolute/path/input.json --idempotency-key=UNIQUE-COHORT-BACKEND \
  --await --json --out=/absolute/path/results
```

An input is `{"prompt":"A martial artist performs fluid wushu movements in a sunlit courtyard.","seed":7101,"duration_s":5,"backend":"fa3_bf16"}`. Reuse the exact prompt, seed, duration and optional keyframe assets across backends; choose5 or15 seconds. Record the exact CLI/image/Runtime cohort separately. Reattach interrupted watchers with the same idempotency key and preserve previous event logs.

Successful output retains the ordinary video, continuation frame and warnings, plus a small JSON report containing source hashes, actual backend provenance, exactly1500 applied calls (30 per block), GPU, Torch and generation/denoise timings. Exceptions restore hooks and the dispatch registry; an unavailable backend or missed call refuses rather than falling back. Existing H3 numerical checks still guard every denoising step and decoded media. Timing includes the private wrapper and helper setup, so compare full generation and denoise times under the same project and worker state; this is a quality experiment, not a production speed guarantee.
