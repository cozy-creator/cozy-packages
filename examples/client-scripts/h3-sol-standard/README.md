# Standard H3 attention qualification

This ordinary Python `main()` delegates the existing H3 `fl2va` workflow with
30 steps, one base model and no PDD or other LoRA. It follows the previously
qualified `h3_attention_oracle.generate` body; no sampler, decoder or kernel is
copied. The model selects Sol for `fl2va_dit` during preparation on every rank.
All other components use the normal Runtime policy.

Stage two separate packages from this source. Keep `h3-sol-standard` and its
`ATTENTION_BACKEND="sol-attn"` literal for Sol. Name the other package
`h3-fa3-standard` and change only that literal to `flash-attn3`. Each App has
one entrypoint and exactly one model. Request attention pins verify the prepared
backend rather than changing a running Ulysses group.

Use the same migrated workflow at Packages commit
`9798fa332736b4fe8fe8019f58bf4241c88a6438` as the earlier 30-step comparison.
The helper rejects a workflow that defines a different H3 model class. In staged
copies, point `tool.uv.sources.minimax-h3` to that exact local workflow, and
`cozy-runtime` to the reviewed development wheel. The version floor alone does
not prove the builtin model or Sol/Ulysses APIs are present in a public release.

For this comparison, capture with Runtime source
`dd7d11346b4356031099c8d912d19b7775a69ad1`, wheel SHA256
`05e29a094d4e118d3d87e1498a6d40b15f113c2cecdc3772fa94ea05d5de8ab4`,
and the same image-owned TensorFS 0.3.42, Torch 2.13.0+cu130 and torchvision
0.28.0+cu130 cohort as the Turbo controls. Exclude default/dev dependency groups.
Use a fresh source directory and keep logs, inputs and outputs outside it.

The matched input is the existing 15-second rain-soaked courtyard fight,
seed 7101, with no supplied keyframes. Bind the fixed FP8 AdaLN-pruned base
checkpoint; do not add a Turbo adapter. The standard Sol policy keeps the first
ten evaluations, first two main blocks and both refiner blocks dense, while
protecting conditioning and audio tokens throughout.

Expected Sol counts per rank over 30 evaluations are sparse960,
dense-prefix960, dense-step480 and dense-path120. FA3 should have no Sol calls.
Record actual GPU count, selected backend, preparation/JIT separately from
execution, denoising time, per-rank counts, output video/audio digests and quality.
Do not infer equal quality or cross-degree byte identity from matching seeds.

The current CLI uses the rental's full declared group. Separate owned rentals
are needed for ordinary 1/2/4-GPU qualification; do not invent a `--gpus` request
flag or use another person's rental. Capture and concrete run-command records
are prepared outside this source tree; root owns installation and submission.
