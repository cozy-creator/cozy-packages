# Reference image

`generate` creates one PNG from text using Qwen-Image-2.1. The model checkpoint is
selected independently from the package. TensorFS owns model downloads and bytes;
Runtime owns construction, memory residency and component execution.

Use `background=white` for a character reference or `background=normal` for a scene.
Native RGBA decoding is composited over white because the v1 API produces RGB PNGs;
it does not promise transparent outputs. `seed` is optional and the chosen seed is
returned. Width and height are multiples of 32, default 1024; steps default 40.

The upstream model is under the Qwen Research License, for noncommercial research
and evaluation. This wrapper is not an authorization for commercial model use.
See the checkpoint's licence/Notice and upstream source attribution in Runtime.


## Prepare the checkpoint

With Runtime 0.18.23 or newer, use an existing rental to download, convert, attach
required metadata, and upload the model:

```sh
cozy model upload \
  hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758 \
  paul/reference-image --rental=YOUR_RENTAL --await
cozy model publish paul/reference-image --release 0.1.0 \
  --lane original=sha256:CHECKPOINT_DIGEST
```

Replace `YOUR_RENTAL` and `CHECKPOINT_DIGEST` with the selected rental name and
returned checkpoint digest. Upload creates the checkpoint; the separate publish
command creates the release used by the package's default binding. The optional
[client script](../examples/client-scripts/prepare_reference_image.py) shows how to
compose the same native operations in Python. Preparation and descriptor checks
alone do not qualify GPU inference or the generated image.
