# Reference image

`generate` creates one PNG from text using Qwen-Image-2.1. The model checkpoint is
selected independently from the package. TensorFS owns model downloads and bytes;
Runtime owns construction, memory residency and component execution.

Use `background=white` for a character reference or `background=normal` for a scene.
Native RGBA decoding is composited over white because the v1 API produces RGB PNGs;
it does not promise transparent outputs. `seed` is optional and the chosen seed is
returned. Width and height are multiples of32, default1024; steps default40.

The upstream model is under the Qwen Research License, for noncommercial research
and evaluation. This wrapper is not an authorization for commercial model use.
See the checkpoint's licence/Notice and upstream source attribution in Runtime.
