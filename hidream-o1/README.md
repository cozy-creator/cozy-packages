# HiDream-O1-Image

HiDream.ai's HiDream-O1-Image: an 8B pixel-space Unified Transformer for text-to-image generation
and editing. It is Qwen3-VL-8B with a 32 px patch embedder, a timestep token and a pixel head, and
has no VAE and no separate text encoder. Text and condition tokens attend causally, and the noisy
patches attend to everything (tech report arXiv:2605.11061 §3.3).

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | HiDream-O1-Image | 50 | `guidance` 5.0 (CFG) |
| `generate_turbo` | HiDream-O1-Image-Dev (distilled) | 28 (its only schedule) | none |

One function does both jobs: with `reference_images`, it edits them, or composes several in order.
HiDream recommends the full model, `generate`, for editing.

```sh
cozy run fidika/hidream-o1/generate_turbo prompt='A dog holds a sign that says "HiDream-O1-Image release."'
cozy run fidika/hidream-o1/generate prompt='remove the earphones' --asset reference_images.0=photo.jpg
```

Fields: `prompt`, `reference_images` (up to 10), `aspect_ratio` (any `W:H`), `megapixels`, `steps`, and
`seed` (random when omitted, returned). `generate` also takes `guidance` and `negative_prompt` (default:
upstream's unconditional caption, a single space). Both take `prompt_upsampling` (default `false`).
Unknown fields are dropped with a warning.

The result has the image, its `width` and `height`, the `seed`, and `prompt`: the caption used.

## Sizes

The model was refined at 2048² (report §4.1), and all of upstream's sizes are about 4 MP. Text-to-image
defaults to 1:1 at 4 MP, 2048×2048. `megapixels` is held to 1–4, where 1 MP is 1024² pixels and the
earlier pre-training stage ran at that size. Sides are 32 px multiples from 256 to 4096, and a ratio
past that is clamped. Each clamp is reported as an adjustment.

With references, any size input you leave out comes from the first reference: its ratio, its pixel
count, or both. With one reference and neither input, the output takes the reference's own size on the
32 px grid. This is upstream's `keep_original_aspect`: the reference is resized to that area, and the
output matches it. Otherwise each reference is resized to the area of a square on the output's longer
side, smaller when there are more references (upstream's `generate_image`).

## Sampling

All recipes are upstream's latest (HiDream-ai/HiDream-O1-Image `main` at 2c2d29f, `inference.py`):

- **`generate`:** Wan's flow UniPC (bh2, order 2) at shift 3, with CFG against a single-space caption.
  With two or more references it uses shift 1, as upstream's README runs subject-driven generation.
  The initial noise is 8 standard normals per pixel.
- **`generate_turbo`:** the 28 fixed Dev timesteps. Text-to-image and two or more references use the
  flash sampler: predict the clean image, then noise it again to the next sigma, at noise scale 7.5
  clipped at 2.5 standard deviations. A one-reference edit uses flow-match Euler, upstream's editing
  scheduler for Dev since 2026-05-12.

As in upstream, the noise comes from `seed + 1`, and the state between steps is bf16. The vision tower
encodes each reference at 384² (288² for 5–8 references, 192² for more). Its tokens go into the prompt,
and its deepstack features are added after the first three layers, as upstream added on 2026-05-13.
ComfyUI's port drops them. The text and condition tokens' keys and values are computed once per request.

`scripts/hidream-o1-model-proof.py` checks the cached forward against Transformers' own Qwen3-VL text
model over the whole sequence under the hybrid mask. It also checks the UniPC timesteps against
upstream's.

## Prompt upsampling

`prompt_upsampling=true` runs HiDream's Reasoning-Driven Prompt Agent on a text-to-image prompt.
The agent is Gemma-4-31B-it with upstream's system prompt (`prompt_agent.py`): no thinking, sampling at
temperature 1.0 (Gemma's top-k 64, top-p 0.95), seeded by `seed`. It rewrites the request as one dense
English paragraph, the kind of prompt the model was trained on (report §3.2). A reply that does not
parse falls back to the request plus upstream's quality suffix. The agent reads text only, so an
edit's prompt is passed as typed, with a warning. The agent's token loop declares
`min_bits=4, phase="decode"`, so it serves the nvfp4 lane.

## Prepare the models

Upstream ships f32 weights (35 GB per checkpoint). Upload each repository as it is:

```sh
cozy model upload hf://HiDream-ai/HiDream-O1-Image@0b0901d99f200389e138c61946af1185f5f49a13 <account>/hidream-o1 --on <machine>
cozy model upload hf://HiDream-ai/HiDream-O1-Image-Dev@c0bada0e15c54a9f96a6d1ecc35575b32bc21544 <account>/hidream-o1-dev --on <machine>
cozy model upload hf://google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475 <account>/gemma-4-31b-it --on <machine>
```

Each lane is written from the f32 upload. `bf16` rounds every f32 tensor to bf16. `int8` (the default
lane, `int8.convrot`, W8A8 on int8 tensor cores) takes the attention and MLP projections of the 36
decoder layers, 6.9B of the 8.2B weights read per step. The patch embedder, timestep MLP, pixel head,
embeddings and vision tower stay bf16. `nvfp4` takes the same projections. For the agent, `int8` and
`nvfp4` take Gemma's 410 language projections.

```sh
cozy model quantize <account>/hidream-o1 --recipe bf16 --on <machine>
cozy model quantize <account>/hidream-o1 --int8 --on <machine>
cozy model quantize <account>/gemma-4-31b-it --nvfp4 --on <machine>
```

## Licence

HiDream-O1-Image and its code are MIT (`hidream_o1/LICENSE.model`, Copyright (c) 2026 HiDream.ai).
Gemma-4-31B-it is Apache-2.0. The UniPC solver follows Wan's (Apache-2.0).
