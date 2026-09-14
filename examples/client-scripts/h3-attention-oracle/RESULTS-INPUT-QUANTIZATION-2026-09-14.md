# H3 input quantization and key-centering controls

K quantization accounts for most of the remaining error at the tested last H3
block. V-only error is much smaller. Removing a shared key mean reduces the
error only slightly, so this experiment does not establish a quality-preserving
FP8 replacement and changes no production default.

The ordinary default-home CLI captured unchanged published H3 1.14.3 inputs,
jazz prompt, seed 7101, requested 15 seconds and standard 30-step schedule.
Capture sites were step 0, blocks 0 and 49, with full `[1,109104,56,128]` Q/K/V.
Each roundtrip uses the actual Runtime quantizer, restores selected inputs to
BF16 and calls the same BF16 FA3 kernel. Unselected inputs remain unchanged.
Times include all preprocessing, with five timed calls after warmup. Each
backend passed its fixed-input `torch.equal` repeat comparison; source input
hashes were unchanged. These are operation errors, not video quality scores.

## Q/K/V isolation

| Quantized inputs | First-block full-output L2 vs BF16 | Last-block full-output L2 vs BF16 | First / last median ms |
|---|---:|---:|---:|
| None | 0 | 0 | 496.854 / 482.311 |
| Q only | 0.01140063 | 0.02078226 | 494.365 / 488.435 |
| K only | 0.02528468 | 0.17272067 | 505.187 / 489.074 |
| V only | 0.00656274 | 0.01035144 | 505.022 / 486.641 |
| Q, K and V | 0.02832690 | 0.17445564 | 526.002 / 504.128 |

Sampled FP32 attention over 16 evenly spread queries, retaining every key/value,
gives last-block relative L2 of 0.0015990 for BF16, 0.0176734 for Q only,
0.1804863 for K only, 0.0102458 for V only and 0.1791987 for all three. Errors
do not add linearly because changing Q/K changes the softmax weights.

Runs 759/760 were `req-e65ae0e1bd5058203c5b55d9#1` and
`req-e231b7b5c46cff97967032a4#1`. CLI execution was 45.3 / 73.5 seconds;
queue/preparation was 1303.7 / 1362.0 seconds. Measurements are respectively
`dc9778c0d59a316833edb53fd27ca07998569865e2f4f25b3ce5a12dd3e612ab` and
`cabc7f5c4807ea63529f1a7d36ee790a6cc432445062938493914f6b5ac6a257`.

## One focused last-block key-centering follow-up

A FP32 mean over sequence is subtracted separately per batch/head/channel.
Ideal softmax is invariant to this shared key shift; the BF16-only arm measures
the rounding effects. The corrected FP8 arm reuses the existing tile128 kernel
source `76aebdf701653cb41036c1523cb6c03606d22a10`, with original Q/V and centered K
quantized through the production quantizer. No new attention kernel was built.

| Arm | Median ms | Full-output L2 vs original BF16 | Sampled-FP32 L2 |
|---|---:|---:|---:|
| Original BF16 | 480.790 | 0 | 0.00159899 |
| Centered K, BF16 attention | 494.420 | 0.00372618 | 0.00253192 |
| Centered K roundtrip, BF16 attention | 496.735 | 0.16803982 | 0.16535307 |
| Centered K, corrected tile128 FP8 | 401.636 | 0.17100712 | 0.16753057 |

The centered K roundtrip retains almost all of the uncentered K-only error.
This is a negative result for shared-mean subtraction as the solution. No full
video or additional kernel experiment was launched from it.

Run 763 was `req-2d7e1379305f1d10f94cc791#1`: 66.2 seconds CLI execution,
291.5 seconds queue/preparation, completed at 10:37:23 UTC. Measurement SHA256
is `18b8a29fb95feec0773a893b5f3f729147b85b25d28324f7186d47bb1afe3ad5`.

## Provenance and interpretation

Root-owned jinwoo retained Runtime source 381d8077, Torch 2.13.0+cu130,
TensorFS 0.3.42 and Triton 3.7.1. The exact checkpoint was
`sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275`.
Q/K/V hashes match the prior Torch 2.13 candidate captures and each other at the
same site. This does not assert equivalence to the separate Torch 2.14 cohort.

Frozen wheels, all 20 unchanged published H3 members, inputs/commands, CPU
control proofs, JSONL receipts, verified measurements and CUDA traces are in
`outputs/h3-attention-oracle-20260914/{qkv-isolation,k-centering}/`. The broader
current report is `outputs/h3-attention-oracle-20260914/ATTENTION-RESULTS.md`,
which also links the completed coherent Torch 2.14/cuDNN 9.24 benchmark and
matched full 15-second video comparison. This bounded input study stops here;
finer K precision schemes need separate design and validation.
