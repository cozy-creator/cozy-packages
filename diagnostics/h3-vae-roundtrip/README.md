# H3 VAE reconstruction diagnostic

This package encodes a still reference repeated for 22 or 345 frames and decodes
it through the actual H3 VAE. The image is letterboxed onto 1344×768. Its output
is a reconstruction diagnostic, not a generated scene or an inference quality pass.

It imports the existing `minimax-h3` package as a normal local wheel dependency,
subclasses its model and uses Runtime's ordinary component scopes. It does not
copy the inference implementation, access the Store directly, or download model
files itself. The lock selects Runtime 0.2.29 to match the worker being diagnosed.

Build the dependency wheel from the existing inference source, then publish from
this directory, install without prefetch, and invoke on an existing
rental using the already published candidate checkpoint:

```sh
uv build --project ../../minimax-h3 --wheel --out-dir dependencies
uv lock
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

Version 1.0.1 defaults to `cycle=true`: encode in the VAE scope, enter the Ref2VA
DiT scope to hash selected resident tensors, then decode in a new VAE scope.
The Runtime residency log establishes whether this caused eviction and reloading.
`cycle=false` performs encode and decode within one uninterrupted VAE scope.
Both modes return small VAE fingerprints before and after; the cycling mode also
returns DiT fingerprints. These hashes preserve the loaded dtype and compare with
the sampled BF16-pruned checkpoint. They are diagnostics, not checkpoint IDs.
