# Qwen Image 2

`generate_image` creates or edits one PNG using Qwen-Image-2.1, and `generate_image_turbo` does
the same with Qwen-Image-2.1-Turbo. Both take the same inputs. The model checkpoint is selected
independently from the package. TensorFS owns model downloads and bytes;
Runtime owns construction, memory residency and component execution.

`prompt` reaches the model exactly as given. Native RGBA decoding is composited over white because the v1 API produces RGB PNGs;
it does not promise transparent outputs. `seed` is optional and the chosen seed is
returned. Seeds range from 0 through 9007199254740991 (the interoperable 53-bit
integer range); omitting one chooses a random seed in that range. Steps default to 40; a checkpoint
that names its own schedule (Turbo's `sample_sigmas`, 8 steps) runs it, and a different `steps` is
reported as a `clamp` adjustment.
Unknown fields, such as the removed width/height inputs, are ignored with a warning.

`aspect_ratio` is any `W:H` and `megapixels` is 1, 2 or 4 (1024² pixels each). The size is
upstream's `calculate_dimensions` at that area, in multiples of 32 px; text-to-image defaults to
`1:1` at 4 MP, the model's native 2048×2048. A ratio past the model's rotary range (a 32768 px
side) or its tier's area is served at the nearest size it admits, reported as a `clamp`
adjustment. Width and height are output fields only.

## Image editing and references

Set `reference_images` to one through ten images and describe the desired edit or
composition in `prompt`. Omit it for ordinary text-to-image generation. The same
Qwen-Image-2.1 checkpoint handles both modes; editing runs its complete instruction
sampling schedule, with no image-to-image strength parameter.

```sh
cozy run fidika/qwen-image-2/generate_image \
  prompt='Change the background to a sunset beach; preserve the person and clothing.' \
  --asset reference_images.0=portrait.png seed=42

cozy run fidika/qwen-image-2/generate_image \
  prompt='Place the person from image 1 in the room from image 2.' \
  --asset reference_images.0=person.png --asset reference_images.1=room.png \
  seed=42
```

References are ordered. Each accepts PNG, JPEG or WebP up to 32 MiB encoded; Runtime
decodes it upright (EXIF orientation) and scales anything larger than 4 MP down to 4 MP. This package currently uses RGB input and output; transparent layer
editing is not exposed. Each reference is resized to the upstream 1 MP conditioning
area independently of the output's `megapixels` tier.

With references, an omitted `aspect_ratio` takes the last reference's exact ratio (as upstream) and an omitted `megapixels` the tier nearest its pixel count.

Reference VAE encoding and multimodal prompt encoding each run once before
denoising, in separate component residency leases. Denoising reuses their tensors
and the upstream prefix KV cache across steps. Reference images add tokens and
memory demand, so compare editing benchmarks separately from text-only runs.

The upstream model is under the Qwen Research License, for noncommercial research
and evaluation. This wrapper is not an authorization for commercial model use.
The licence and Notice ship in this package (`qwen_image_2/LICENSE.model.txt`, `qwen_image_2/Notice`).
The architecture is Diffusers' (Apache-2.0), at the release `pyproject.toml` pins.


## Prepare the checkpoint

Use an existing rental to download, convert and upload the model. The checkpoint
carries the upstream `processor/` files (tokenizer, chat template, image and video
processor configs) as assets; the package builds its processor from them:

```sh
cozy model upload \
  hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758 \
  fidika/qwen-image-2.1 --on YOUR_MACHINE
cozy model publish fidika/qwen-image-2.1 --release 0.1.0 --commit main
cozy model upload \
  hf://Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b \
  fidika/qwen-image-2.1-turbo --on YOUR_MACHINE
cozy model publish fidika/qwen-image-2.1-turbo --release 0.1.0 --commit main
```

Replace `YOUR_MACHINE` with the selected machine name. Upload makes a commit and moves
the private tag `main` to it; publish releases that commit as the package's default. Preparation and descriptor checks
alone do not qualify GPU inference or the generated image.

## The int8 lane

`int8` writes the fast lane for consumer cards: every linear of the 32 DiT blocks and of the
text encoder's 36 language layers as `int8-convrot256/1` (each 256 input channels rotated by a
Hadamard, int8 per output row), served as W8A8 on the int8 tensor cores. Embeddings, the vision
tower, the input/output projections, the modulation and the VAE keep their source bytes.

```sh
cozy model quantize fidika/qwen-image-2.1@0.1.0/bf16 --int8 --on YOUR_MACHINE
cozy model publish fidika/qwen-image-2.1 --release 0.1.0 --lane int8=sha256:CHECKPOINT_DIGEST
```

## Python dependency

This is an ordinary Python distribution (`qwen-image-2`) published on the
publishing account's Tensorhub package index. A package published by the same
account names that index in its `pyproject.toml` and never spells its URL:

```toml
[project]
dependencies = ["qwen-image-2>=0.9.0"]

[tool.uv.sources]
qwen-image-2 = { index = "tensorhub" }
```

Creator writes the `tensorhub` index (`<hub>/v1/index/<account>/simple/`, explicit)
for the command's Hub and account, so lock with `cozy package lock`, not raw
`uv lock`. uv resolves this distribution only from that index. Import its generated
managed caller:

```python
from qwen_image_2 import Megapixels, generate_image

image = await generate_image(
    prompt="An explorer in a blue coat",
    aspect_ratio="3:4",
    megapixels=Megapixels.MP1,
)
```

The Hub package is `<account>/qwen-image-2`, e.g. `paul/qwen-image-2` on the local
Hub and `fidika/qwen-image-2` on tensorhub.com; its callables are `generate_image` and
`generate_image_turbo`. Their default models are the org-relative `qwen-image-2.1@0.1.0` and
`qwen-image-2.1-turbo@0.1.0`: the publishing account's models.
Installing the wheel alone does not turn GPU execution into an in-process call;
Cozy supplies the managed caller when composing package functions.
