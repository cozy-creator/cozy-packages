# One-step H3 LoRA profile

This private diagnostic leaves the ordinary H3 generator, attention selection and
generic Runtime adapter stack unchanged. It profiles the second DiT step after
the first step has warmed kernels, then stops explicitly after two denoising steps
of the standard30-step5- or15-second schedule. It returns one bounded compressed
Chrome trace and operator counts/shapes/CPU/device times, not a completed video.
It does not merge or modify weights.

Prepare from the same exact published H3 1.14.3 wheel:

```sh
python3 prepare_h3_lora_profile.py /absolute/path/published-minimax_h3-1.14.3-py3-none-any.whl /absolute/path/profile-project
uv lock --project /absolute/path/profile-project
cozy package install --editable --no-model-download /absolute/path/profile-project
```

Submit through ordinary default-home CLI on an explicitly owned single-H100 rental,
after the already-authorized video comparisons finish:

```sh
cozy run local/h3-lora-profile/probe \
  model.model=paul/minimax-h3@1.0.0-h3-audit.1/fp8-adaln-pruned \
  --lora=model:fl2va_dit=paul/minimax-h3-spatial-physics-lora,0.6 \
  --rental=OWNED_RENTAL --in=/absolute/path/matched-input.json \
  --idempotency-key=UNIQUE-PROFILE-COHORT --await --json --out=/absolute/path/results
```

The input contains `prompt`, `seed` (default7381), `duration_s` (5 or15, default15)
and `profile` (defaulttrue). Final LoRA qualification uses the fixed15-second
complex-fight input, with each LoRA independently compared to the same baseline.
Use the exact same input once without`--lora` and once with the selected adapter.
No global attention pin is valid across all H3 components. The Runtime execution
observation must confirm actual LoRA calls and FP32 factor precision on the adapted
run. `profile=false` supplies an unprofiled two-step control; an empty trace is
explicitly reported in that case.

The profiler changes execution overhead. Use its operator shapes and self device
times to identify costs, not its two-step wall time as a generation speed result. FP32
LoRA tile GEMMs have4096 input rows and low inner/output rank; distinguish these
from encoded base GEMMs and existing activation quantization before attributing
cost. Merge arithmetic and cache lifecycle are a separate experiment; no production
behavior, format choice or quality promise follows from this trace alone.

## Actual-layer merge probe

`prepare_h3_lora_merge_probe.py WHEEL ARITHMETIC_FILE DESTINATION` stages a separate
`local/h3-lora-merge-probe/probe` project. The preparer pins the reviewed helper's
exact bytes; it does not install or execute anything. Lock/install it through the
same ordinary CLI flow above, then supply the identical fight input (omit fixed
`steps`, select optional `block=0|25|49`) and a nonzero generic `--lora` binding.

The probe runs one actual denoising step, observes one Runtime-owned encoded Q
projection and its already validated active factors, then stops before video
decode. It takes at most the first4,096 packed input rows and compares separate
candidate weight buffers. It never patches the live module, subtracts an update,
writes a checkpoint or implements a cache. The temporary ownership bound is1GiB,
checked against tensor shapes and actual device headroom; one candidate is held at
a time and all original source/factor hashes must remain unchanged.

Controls compare rank-ordered FP32 and highest-mode FP32 GEMM preparation under
preserved, grow-only and recalibrated row scales. Original-grid overflow is a
recorded refusal. A BF16 merged-weight control changes the activation route too;
the report says so. Forced zero-update codec controls expose gratuitous grid
changes, while actual zero strength remains an exact original-part bypass.
Autocast is disabled during measurements without changing process-global math
settings. The report includes original-scale diversity, code maxima, preparation
and forward times, sampled FP64 weight/forward errors, and error relative to the
LoRA update's norm. Per-row numerical diagnostics are part of prototype preparation
time. These are one-layer measurements, not full-model quality or cached-serving
qualification.
