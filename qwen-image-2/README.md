# Qwen Image 2

`generate_image` creates or edits one PNG using Qwen-Image-2.1. The model checkpoint is
selected independently from the package. TensorFS owns model downloads and bytes;
Runtime owns construction, memory residency and component execution.

Use `background=white` for a character reference or `background=normal` for a scene.
Native RGBA decoding is composited over white because the v1 API produces RGB PNGs;
it does not promise transparent outputs. `seed` is optional and the chosen seed is
returned. Seeds range from 0 through 9007199254740991 (the interoperable 53-bit
integer range); omitting one chooses a random seed in that range. Select `aspect_ratio` and `megapixels` (1, 2 or 4); the default is `1:1` at 1 MP,
exactly 1024×1024. Steps default to 40. Width and height are output fields only.
Unknown ratios, tiers, and the removed width/height inputs are refused.

## Image editing and references

Set `reference_images` to one through ten images and describe the desired edit or
composition in `prompt`. Omit it for ordinary text-to-image generation. The same
Qwen-Image-2.1 checkpoint handles both modes; editing runs its complete instruction
sampling schedule, with no image-to-image strength parameter.

```sh
cozy run paul/qwen-image-2/generate_image \
  prompt='Change the background to a sunset beach; preserve the person and clothing.' \
  --asset reference_images.0=portrait.png seed=42 --await

cozy run paul/qwen-image-2/generate_image \
  prompt='Place the person from image 1 in the room from image 2.' \
  --asset reference_images.0=person.png --asset reference_images.1=room.png \
  seed=42 --await
```

References are ordered. Each accepts PNG, JPEG or WebP up to 32 MiB encoded and
20 MiB decoded RGB. Runtime verifies and decodes them through the ordinary input
asset path. This package currently uses RGB input and output; transparent layer
editing is not exposed. Each reference is resized to the upstream 1 MP conditioning
area independently of the output's `megapixels` tier. Output dimensions still come
from `aspect_ratio` and `megapixels`.

Reference VAE encoding and multimodal prompt encoding each run once before
denoising, in separate component residency leases. Denoising reuses their tensors
and the upstream prefix KV cache across steps. Reference images add tokens and
memory demand, so compare editing benchmarks separately from text-only runs.

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

## Resolution buckets

Megapixels are nominal area classes, as in Anima and SDXL. Every dimension is a
multiple of the Qwen latent patch stride (32 pixels), and every image stays below
5 million pixels and the 20 MiB decoded RGB limit. The 4 MP standard buckets are
[Qwen2.1 native recommendations](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios);
lower tiers scale those sizes and snap to 32 pixels. The 21:9 and 9:21 buckets
extend the supported grid at comparable area; GPU quality at those extremes
is not established by the request-contract checks.

| Aspect ratio | 1 MP | 2 MP | 4 MP |
| --- | --- | --- | --- |
| 1:1 | 1024×1024 | 1440×1440 | 2048×2048 |
| 4:3 | 1216×896 | 1696×1280 | 2400×1792 |
| 3:4 | 896×1216 | 1280×1696 | 1792×2400 |
| 3:2 | 1280×864 | 1792×1216 | 2528×1696 |
| 2:3 | 864×1280 | 1216×1792 | 1696×2528 |
| 16:9 | 1376×768 | 1952×1088 | 2752×1536 |
| 9:16 | 768×1376 | 1088×1952 | 1536×2752 |
| 21:9 | 1568×672 | 2208×960 | 3136×1344 |
| 9:21 | 672×1568 | 960×2208 | 1344×3136 |

## Python dependency

This is an ordinary Python distribution (`qwen-image-2`) published on the
publishing account's Tensorhub package index. A package published by the same
account names that index in its `pyproject.toml` and never spells its URL:

```toml
[project]
dependencies = ["qwen-image-2>=0.2.0"]

[tool.uv.sources]
qwen-image-2 = { index = "tensorhub" }
```

Creator writes the `tensorhub` index (`<hub>/v1/index/<account>/simple/`, explicit)
for the command's Hub and account, so lock with `cozy package lock`, not raw
`uv lock`. uv resolves this distribution only from that index. Import its generated
managed caller:

```python
from qwen_image_2 import AspectRatio, Megapixels, generate_image

image = await generate_image(
    prompt="An explorer in a blue coat",
    aspect_ratio=AspectRatio.PORTRAIT,
    megapixels=Megapixels.MP1,
)
```

The Hub package is `<account>/qwen-image-2`, e.g. `paul/qwen-image-2` on the local
Hub and `fidika/qwen-image-2` on tensorhub.com; its callable is `generate_image`. The
default model is the org-relative `reference-image@0.1.0/original`: the publishing
account's `reference-image` model.
Installing the wheel alone does not turn GPU execution into an in-process call;
Cozy supplies the managed caller when composing package functions.
