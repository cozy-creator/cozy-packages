# H3 four- and eight-GPU benchmark captures

`prepare.py` makes local H3 packages from this checkout. The original functions,
weights, geometry and sampling schedules are unchanged. Each capture's base and
Turbo LoRA ladders have only one width (4 or 8), for H100 and RTX 5090. This matters
on an eight-card machine: the shipping ladder currently stops at four cards.

The captures pin Runtime 0.22.0. They do not publish a package or change the user's
Hub. Obtain the configured account index from `cozy package lock`, then:

```sh
python research/h3-scaling/prepare.py /absolute/output/native \
  --account-index=http://127.0.0.1:8819/v1/index/paul/simple/
cd /absolute/output/native/package-degree-4
cozy package lock --upgrade-package=cozy-runtime
cozy run ./fl2va_turbo --describe --json
```

Repeat the lock and describe for `package-degree-8`. The explicit index in the
capture is required by `uv export --locked`; it must match the configured Hub's
account index. The script preserves the original model references:

- `minimax-h3@1.0.0-rc.3/fp8-pruned`
- `minimax-h3-turbo-lora@1.0.0-audit.1/pdd8`

The current rc.3 base differs from the prior rc.2 campaign: 350 text-encoder
decoder weights now use rowwise FP8. Header comparison proves both DiTs and both
VAEs retain exactly the same tensor descriptors and content hashes; only those
350 text-encoder rows change. Comfy's prepared default uses its BF16 text encoder.
Report this precision difference and retain fresh quality videos; the older
attention quality approval does not establish approval of the new text encoder.

Invoke the local directory through ordinary `cozy run`, with an existing rental
and a unique idempotency key. The input is the full 15-second request. Turbo uses
8 steps (4 initial dense, 4 sparse); regular uses 30 (10 dense, 20 sparse).

An accepted request is not a degree proof. Before accepting a timing, retain
`cozy run show <id> --json --full` and verify the reported degree, each participating
GPU UUID and rank, attention recipe and Sol counts, all denoising steps, and the
362-frame 1344×768 24-fps video with audio. Record compile/download warmup separately
from repeated steady runs. Do not overlap Cozy and Comfy inference.

On compatible 5090 systems, degree 4/8 uses the head-local Sage3 NVFP4 dense plus
Kitchen INT8 Sol recipe. The shared-QKV producer is single-GPU only. H100 defaults
remain BF16 Sol; its dense reference is selected from the ready Sage/FA3/SDPA
preferences and must be recorded from the actual run, not assumed.

Only the DiT is sequence parallel. Sampling and the audio/video VAEs remain on the
leader, so a fourfold DiT improvement does not imply a fourfold full-video speedup.
H3 has 56 attention heads: degree 4 assigns 14 heads/rank and degree 8 assigns 7.
Eight-rank source support and CPU tests are not yet an eight-GPU benchmark result.
