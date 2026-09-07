# Repair the four original H3 checkpoints

`paul/checkpoint-repair/h3-swiglu` is an ordinary CPU job with one model input, `source`,
and one checkpoint output, `checkpoint`. It repairs the104 fused FC1 matrices in
both DiTs, including their token-refiner blocks. The original conversion copied
`gate,value` rows into a Diffusers layout that requires `value,gate`.

The package imports no Torch, NumPy or inference code. This migration accepts only the four immutable checkpoints produced by that old
converter. It refuses corrected or unknown roots, preventing a second swap from
undoing the repair. BF16, FP8 and MXFP8 data/scale rows are permuted exactly;
quantization and AdaLN table computation are not repeated. All other tensors and
configs are inherited through the existing TensorFS transaction.

Repair reads into one bounded role buffer (largest production role308,281,344B),
then writes through `add_part`; it creates no temporary weight files. Reading from
the same transaction inside an `add_part` stream callback would re-enter the native
writer, so those operations remain sequential. Completed roles and tensor-boundary
checkpoints resume through Runtime0.2.32/TensorFS0.3.18; completed output replay reads
no payload. The native writer may allocate additional working memory.

The repair performs no inference and makes no output-quality approval claim.
Publish its uploaded checkpoint through the ordinary release flow, with separate
matched-input inference/evaluation evidence. Run `scripts/h3-repair-proof.py` from
the repository root for actual native plain/FP8/MXFP8 repair, interruption/replay,
unchanged-reference checks and the failing identity-copy control.
