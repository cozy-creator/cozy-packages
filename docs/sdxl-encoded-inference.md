# SDXL encoded linear inference

Owner: `/root/h3_longform_finish`, continuing the existing `fix/sdxl-encoded-inference`
branch in `~/cozy/.worktrees/packages/sdxl-encoded-inference`. Original change
`84f2c45d` was resumed from a clean, inactive worktree and merged with fetched
`origin/master` at `8a187443`.

The shared Runtime quantizer can produce encoded UNet linear weights, but SDXL's
model declaration refused them before inference. `SdxlModel` now accepts Runtime's
encoded linear replacements. Its construction topology and forward calls stay the
same; Runtime owns storage and execution of encoded weights. Plain checkpoints
remain accepted. No per-family quantizer or new package is added.

The generated interface changes only the two SdxlModel encoded-leaf declarations.
Current SDXL 2.3.6, native TensorFS normalization, source provenance, dependency lock
and package metadata are preserved. PR204 previously reproduced the refusal through
the ordinary Creator CLI after quantization and a caller-edit memo hit. Its older
green CI is historical evidence; this resumed head requires fresh CI.

This change removes the admission refusal. Actual inference, output quality and the
full download/convert/quantize/evaluate/publish memoization qualification remain under
se-042 and the root-owned rented CLI proof. No GPU-quality claim is made here.
