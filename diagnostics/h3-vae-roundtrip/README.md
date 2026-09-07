# H3 VAE reconstruction diagnostic

This package encodes a still reference repeated for 22 or 345 frames and decodes
it through the actual H3 VAE. The image is letterboxed onto 1344×768. Its output
is a reconstruction diagnostic, not a generated scene or an inference quality pass.

It imports the existing `minimax-h3` package as a normal local wheel dependency,
subclasses its model and uses Runtime's ordinary component scopes. It does not
copy the inference implementation, access the Store directly, or download model
files itself. Runtime 0.2.29 is pinned to match the worker being diagnosed.

Publish from this directory, install without prefetch, then invoke on an existing
rental using the already published candidate checkpoint:

```sh
cozy package publish
cozy package install paul/h3-vae-diagnostic --no-model-download
cozy run paul/h3-vae-diagnostic/roundtrip frames=345 \
  model.model=paul/minimax-h3@1.0.0-rc.1/bf16-adaln-pruned \
  --asset image=/absolute/path/reference.png --rental-only --await --out ./result
```

The result includes the source image, reconstruction, playable video and a grid
metric. Compare both temporal lengths; the shorter test does not establish
correctness of long-video VAE windows. Do not infer DiT or sampler correctness
from a successful round trip.
