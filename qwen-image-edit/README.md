# Qwen Image Edit

Qwen-Image-Edit-2511 (the Edit Plus line): Qwen-Image's 20B dual-stream DiT with a Qwen2.5-VL-7B
text encoder. One function edits or composes when given `reference_images` and generates from text
otherwise.

| function | weights | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | `qwen-image-edit-2511@0.1.0` | 40 | true CFG 4.0, `negative_prompt` " " |
| `generate_turbo` | the same, with `qwen-image-edit-2511-lightning-lora@0.1.0` | 8 | none |

```sh
cozy run fidika/qwen-image-edit/generate prompt='Make him play a violin instead of the guitar' \
  --asset reference_images.0=guitarist.png seed=42
cozy run fidika/qwen-image-edit/generate_turbo \
  prompt='The cat from Picture 1 wearing the space suit from Picture 2' \
  --asset reference_images.0=cat.jpg --asset reference_images.1=astronaut.jpg
cozy run fidika/qwen-image-edit/generate prompt='A red vintage bicycle against a pastel wall' \
  aspect_ratio=3:2
```

Inputs: `prompt`, `reference_images` (ordered, up to ten; Qwen recommends one to three, and the
prompt calls them "Picture 1", "Picture 2", ...), `aspect_ratio` (any `W:H`), `megapixels` (1, 2 or
4; 1024² pixels each), `steps`, `seed` (random when omitted, returned). `generate` also takes
`guidance` and `negative_prompt`. Unknown fields are ignored with a warning. Outputs: `image` (PNG),
`width`, `height`, `seed` and `prompt`, the prompt as given (there is no upsampler).

## Size

With references and neither size input, the output takes the first reference's aspect ratio and
pixel count. Upstream takes the last reference's ratio at 1 MiP; this package uses the first on
purpose. An explicit `aspect_ratio` or `megapixels` replaces only that one. Text-to-image defaults to
1:1 at 1 MiP (1024×1024), the area upstream edits at. Sides are rounded to the 16 px grid. The model
serves 256² to 2 MiP (it is trained up to 1328²), with sides from 128 to 4096 px. Anything outside
that is clamped and reported as a `clamp` adjustment, for example `megapixels=4` or a 12 MP photo.

## Fidelity

Conditioning and sampling are Diffusers' `QwenImageEditPlusPipeline` (0.41.0), Qwen's reference
code. Each reference goes to the vision encoder at a 384² area and to the VAE at a 1024² area, and its
latents sit beside the target at t = 0 (`zero_cond_t`). Noise is seeded on the CPU, the timesteps are
`linspace(1, 1/steps)` with the checkpoint's dynamic shift, and true CFG keeps the conditional
prediction's norm. References are read upright, with any transparency composited over white.
`generate_turbo` binds lightx2v's Lightning 8-step LoRA (V1.0, fp32) in its own slot, `turbo_lora`, at
the file's `alpha / rank` (8/64), and samples at
lightx2v's constant shift of 3. Runtime bakes the LoRA into the weights of whichever lane serves the
base model; the merged weights exist only on the GPU. `scripts/qwen-image-edit-model-proof.py` checks
the package against Diffusers on tiny real modules: the same pixels for editing with and without
CFG, and Diffusers' own reading of a Lightning-format LoRA.

## Quality and speed (A40, warm)

Same seed and size as Diffusers' bf16 pipeline: the bf16 lane is the same image (61.0 dB on a one-reference
edit, 55.4 dB on a two-reference compose); the turbo function's per-step LoRA is 30.5 dB from Diffusers'
8-step Lightning, and the default bake is visually the same image. The int8 lane re-samples the same image
(18–26 dB from bf16) and keeps every reference's identity in two-reference composes.

| request | bf16 | int8 |
| --- | --- | --- |
| `generate`, one 0.5 MP reference, 40 steps | 3.1 s/step | 2.7 s/step |
| `generate`, two references, 0.9 MP out, 40 steps | 6.6 s/step | 5.5 s/step |
| `generate_turbo`, 8 steps | | 13–23 s sampling |

## Lanes

`int8` writes the default lane: every linear of the 60 DiT blocks, their modulation heads included,
as `int8.convrot` (W8A8). The text encoder stays bf16, as in v1's 8-bit lane: with it int8 too, a
two-reference compose lost the first reference's identity. `nvfp4`
(`--lane 4`, Blackwell) uses NVFP4 for the blocks' attention and MLP linears and int8 for the
modulation heads. Both lanes are made from the bf16 upload. The VAE keeps its stored bf16, as
upstream runs it.

```sh
cozy model upload hf://Qwen/Qwen-Image-Edit-2511@6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
  <account>/qwen-image-edit-2511 --on <machine>
cozy model upload hf://lightx2v/Qwen-Image-Edit-2511-Lightning@107d238d68b8387e32843d0b4a2e24ceb0d3ad86/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors \
  <account>/qwen-image-edit-2511-lightning-lora --on <machine>
cozy model quantize <account>/qwen-image-edit-2511#<checkpoint> --int8 --on <machine>
cozy model quantize <account>/qwen-image-edit-2511#<checkpoint> --nvfp4 --on <machine>
```

## Licence

Qwen-Image-Edit-2511 and the Lightning LoRA are Apache-2.0. The architecture is Diffusers'
(Apache-2.0), at the release `pyproject.toml` pins.
