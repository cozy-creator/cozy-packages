# SDXL

`generate` makes one image from a text prompt with Stable Diffusion XL 1.0 (VAE fix).

```sh
cozy run fidika/sdxl/generate prompt='a lighthouse at dusk, oil painting' seed=7
cozy run fidika/sdxl/generate prompt='a fox in fresh snow' aspect_ratio=16:9 megapixels=2 steps=30
```

Inputs: `prompt`, `negative_prompt`, `aspect_ratio` (1:1, 16:9, 9:16, 4:3, 3:4), `megapixels`
(1, 2, 4, 8 or 16; above 1 uses HiDiffusion unless `hidiffusion=false`), `steps` (1–50,
default 20), `guidance` (0–20, default 7) and `seed`. The image is a WebP in
`~/.cozy/outputs`.

GPU: any NVIDIA card with 8 GB. The default 8-bit lane is 3.8 GiB (`fidika/sdxl@1.0.0`);
`--lane 16` loads the 6.5 GiB bf16 lane. A 1024² image at 20 steps takes about 2.6 s on an
RTX 4090 and 17 s on an 8 GB RTX 4070 Laptop. Without an NVIDIA GPU, add `--on new` to rent
one.

`fp8` and `int8` write the model's 8-bit lanes; they are for model authors
(`cozy model quantize`).
