# BiRefNet background removal

`remove_background` keeps the supplied foreground and composites it onto white.
It uses the standard BiRefNet checkpoint and upstream architecture, torchvision's
input transforms, and Pillow compositing. The image geometry is preserved. Runtime
owns weight loading, placement and residency; the model never fetches a backbone.

```sh
cozy run ./birefnet/normalize model.source=hf://ZhengPeng7/BiRefNet@e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 --tensorhub local
cozy run upload <run> paul/birefnet --tensorhub local
cozy model publish paul/birefnet --release 0.1.0 --commit main --tensorhub local
cozy run paul/birefnet/remove_background --asset image=./character.png
```

The callable returns a lossless WebP. Its deterministic result is memoized by the
normal Runtime mechanism, including the input asset and model identity. H3's
long-form workflow calls it for supplied character references once before the
video segments. Supplied scenes and generated references are not processed.

The source weights retain their original FP16 values and integer buffers in
CozyTensors. The CPU `normalize` function checks the upstream architecture's tensor
names, shapes and types, then adds its constructor configuration without rewriting
weights. The machine downloads and converts the Hugging Face source and uploads
the result through the ordinary run command. Inference uses one GPU at 1024×1024;
its predicted soft mask is resized
to the input dimensions. This selects the salient foreground, so it can retain
multiple people or objects when several appear in one reference.

Upstream code and weights are MIT licensed. The attributed architecture is under
`birefnet/upstream`; see its `NOTICE` and `LICENSE`. No custom image codec or
segmentation architecture is introduced.
