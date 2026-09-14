# One-step H3 LoRA profile

This private diagnostic leaves the ordinary H3 generator, attention selection and
generic Runtime adapter stack unchanged. It profiles the second DiT step after
the first step has warmed kernels, then completes the standard30-step5- or15-second
video. It returns that ordinary video/frame, one bounded compressed Chrome trace,
and operator counts/shapes/CPU/device times. It does not merge or modify weights.

Prepare from the same exact published H3 1.14.3 wheel:

```sh
python3 prepare_h3_lora_profile.py /absolute/path/published-minimax_h3-1.14.3-py3-none-any.whl /absolute/path/profile-project
uv lock --project /absolute/path/profile-project
cozy package install --editable --no-model-download /absolute/path/profile-project
```

Submit through ordinary default-home CLI on an explicitly owned single-H100 rental,
after the already-authorized video comparisons finish:

```sh
cozy run local/h3-lora-profile/generate \
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
run. `profile=false` supplies an ordinary unprofiled control; an empty trace is
explicitly reported in that case.

The profiler changes execution overhead. Use its operator shapes and self device
times to identify costs, not its overall generation time as a speed result. FP32
LoRA tile GEMMs have4096 input rows and low inner/output rank; distinguish these
from encoded base GEMMs and existing activation quantization before attributing
cost. Merge arithmetic and cache lifecycle are a separate experiment; no production
behavior, format choice or quality promise follows from this trace alone.
