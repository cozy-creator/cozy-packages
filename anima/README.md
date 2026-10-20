# Anima

`generate` makes one anime-style image from a text prompt with Anima Base 1.0. The model's
licence is non-commercial.

```sh
cozy run fidika/anima/generate prompt='1girl, silver hair, rain, neon city at night' seed=11
cozy run fidika/anima/generate prompt='a cat samurai under cherry blossoms' aspect_ratio=16:9 megapixels=1
```

Inputs: `prompt`, `negative_prompt`, `quality_prefix` (put before the prompt), `aspect_ratio`
(1:1, 16:9, 9:16, 4:3, 3:4), `megapixels` (1 or 2, default 2), `steps` (8–50, default 30),
`guidance` (1–10, default 4.5), `cfg_interval_start`/`cfg_interval_stop` (the part of the
schedule that uses guidance, default 0.15–0.70), `first_block_cache` (0 = off) and `seed`. The
image is a WebP in `~/.cozy/outputs`.

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.2 GiB (`fidika/anima@1.0.0`); the
bf16 lane is 5.2 GiB. A 1024² image at 30 steps takes about 8.4 s on an RTX 4090 and 70 s on an
8 GB RTX 4070 Laptop. Without an NVIDIA GPU, add `--on new` to rent one.

`fp8` and `int8` write the model's 8-bit lanes; they are for model authors.
