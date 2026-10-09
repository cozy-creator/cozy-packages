# H3 attention selection

H3 uses the attention policy in the worker's Runtime. The package imports Runtime's
`H3Model`, `H3TurboBase`, and `H3TurboLoRA` directly; it does not detect GPU names or
pin a backend. Updating Runtime can therefore change the default for an existing
H3 package release without changing its inputs, checkpoints, or LoRA.

The approved RTX 5090 policy uses **Sage3 NVFP4 dense attention followed by Kitchen
INT8 Sol attention**. This is a hybrid: the sparse kernel is INT8, not FP4.

- Turbo keeps eight evaluations: four dense, then four sparse.
- Regular keeps its requested evaluation count: the first ten are dense, followed
  by sparse attention. The normal 30-step request therefore uses ten plus twenty.
- Token refiners and the first two transformer blocks remain dense in every step.
  The protected text/audio prefix within an otherwise sparse block also receives
  dense attention, using the sparse implementation's prefix backend.

The RTX 5090 quality comparison approved this hybrid and found visible artifacts
in the all-eight-dense Sage3 FP4 clips. All-dense execution remains an explicit
experiment; it is not the new default.

## Hardware support

The default promotion is implemented in [Runtime PR1263](https://github.com/cozy-creator/cozy-runtime/pull/1263).
The default requires both SM120/121 and a compatible reviewed Linux Kitchen wheel.
Runtime checks the wheel's Python, ABI, and platform tags as well as the device,
attention site, degree, and companion kernels. Older Runtime releases can retain
their previous policy.

| Hardware and placement | Preferred implementation |
| --- | --- |
| SM120/121 with the compatible reviewed Linux wheel, one GPU, including RTX 5090 | Sage3 FP4 dense + Kitchen INT8 Sol with shared QKV production |
| SM120/121 with the compatible reviewed Linux wheel, multiple GPUs | Head-local Sage3 FP4 dense + Kitchen INT8 Sol through Ulysses |
| H100, RTX 4090, Ampere, and other devices | Existing Runtime attention preferences remain unchanged pending matched benchmarks |

The package adds no GPU-name switch. A Blackwell family name alone is not enough:
B200/SM100 retains its existing policy, as do platforms without a compatible
reviewed wheel. Those platforms do not gain a new H3 preference for the hybrid.
Explicit kernel pins remain available subject to their own capability and build
checks. Expanded Kitchen INT8 routes do not become universal defaults without
matched measurements.

The single-5090 Turbo matrix qualifies that measured route. Current-candidate
ordinary CLI runs also verified automatic Turbo selection, restoration after a
Kitchen override, and regular 30-step sampling with ten dense plus twenty sparse
steps. These checks do not establish multi-GPU or other-device performance.

## Explicit overrides

An explicit CLI attention pin takes precedence over the default policy:

```sh
cozy run paul/minimax-h3/fl2va_turbo --input=clip.json --rental=my-machine \
  --attention-kernel=fl2va_dit=kitchen-sol-producer-fp4-lowmem-shared-qkv
```

Use the package account on your configured Hub. Selecting a dense-only backend
explicitly keeps every evaluation dense; it does not reduce the eight or thirty
evaluations. An unsupported explicit pin fails instead of silently substituting
another kernel. The run's observed attention implementation and dense/sparse call
counts are the execution evidence; a requested selector alone is not proof.

## Release ownership

This policy uses the existing Runtime model API. It needs a Runtime release and
worker update, not a duplicate package-side selector or an arbitrary package
dependency bump. Raise H3's Runtime floor only when the package requires a newer
API or intentionally stops supporting the older policy. Publish and verify that
Runtime release before committing such a floor.
