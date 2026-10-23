# FLUX.3 Image Klein

Black Forest Labs' FLUX.3 Image Klein: a 5.6B DiT for text-to-image generation and editing with up
to ten reference images. Its text encoder is Qwen3-VL-4B.

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | Klein Base | 50 | `guidance` 4 (CFG) |
| `generate_turbo` | Klein (distilled flow map) | 16 | none |

Each makes an image from the prompt, or edits and composes up to ten ordered `reference_images`.

```sh
cozy run fidika/flux-3-klein/generate_turbo prompt='A lighthouse on a basalt cliff in a winter storm' \
  aspect_ratio=16:9 seed=7
cozy run fidika/flux-3-klein/generate prompt='Make it night, the lamp lit' \
  --asset reference_images.0=lighthouse.png
```

Fields: `prompt`, `reference_images`, `aspect_ratio` (any `W:H`), `megapixels` (1, 2 or 4; 1024² pixels
each), `seed` (random when omitted, returned), `steps`, and `prompt_upsampling` (default `true`). Without
references the default is 1:1 at 1 MP (1024×1024, BFL's `default_size`); the size is the tier's area at
that ratio, rounded to 32 px. With references, what is not given is the first reference's: given neither,
its own size (fitted to at most 4 MP, generated on the 16 px grid and cropped back); given one, its ratio
or its pixel count. Sides stay within BFL's 64–4096 px; a size moved to fit is reported as a `clamp`
adjustment. The result returns the image, its size, the seed and `prompt`: the caption the image was made
from.

## Prompt upsampling

BFL's prompt upsampler is a Qwen3.5-9B fine-tune. It turns a short request, plus the references when
editing, into the full caption Klein is trained on. It runs first, as its own memoized stage, with
BFL's system prompts and decoding: greedy, presence penalty 1.5 for text-to-image and 0 for edits.
Text-to-image also tells it the canvas (`TARGET CANVAS: 16:9`). Set `prompt_upsampling=false` to send
the prompt as typed, which then should be a full scene caption. Its token loop declares
`min_bits=4, phase="decode"`, so it serves the release's `nvfp4` lane where there is one.

## Sampling

The sampling follows BFL's reference code (`black-forest-labs/flux-3-image-klein` at `bfc8889`): Euler
steps with a constant shift (alpha) of 16, CFG 4 for the base model, and noise from the seed in bf16
with an fp32 state across steps. BFL raised alpha from the release's 6.93 and CFG from 3.5 on
2026-10-17; Diffusers and ComfyUI still default to 6.93.
The distilled model also embeds the timestep each step jumps to. The base model runs CFG with an empty
negative prompt. A lone reference is padded to the 16-pixel grid at up to 4 MP, with no upscaling.
With several references, each is cropped to the grid at up to 1 MP. The modules are Diffusers'
(main `7ecb1e1b`; Flux3 is in no release yet). `scripts/flux-3-klein-model-proof.py` checks the
package's constants and denoised tokens against BFL's code, and its images against Diffusers' Flux3
modular pipelines, on tiny real modules.

## Prepare the models

Upload each repository as it is. The served checkpoint is the upload: TensorFS takes a Diffusers repo's
index components (not BFL's `variants/`) and a Transformers repo's config, tokenizer, processor and chat
template. Publish each as lane `bf16`.

```sh
cozy model upload hf://black-forest-labs/flux-3-image-klein@8454017f9ebaada30c998fccd9f67a80c7580f68 \
  <account>/flux-3-klein --on <machine>
cozy model upload hf://black-forest-labs/flux-3-image-klein-base@1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4 \
  <account>/flux-3-klein-base --on <machine>
cozy model upload hf://black-forest-labs/flux-3-image-klein-pup@8b069751eb5dcab23829d9c5eccb470eb8b8a4ed \
  <account>/flux-3-klein-upsampler --on <machine>
```

Klein never builds Qwen3-VL's vision tower, its last four layers or its head, so it never loads those
tensors either. BFL's upsampler system prompts ship in `flux_3_klein/system_prompts.json`.

## The int8 lane

`int8` writes the default lane: every linear of the 43 DiT blocks, the modulation heads, the context
embedder, the output norm and the text encoder's 32 language layers as `int8.convrot`, served
W8A8 on the int8 tensor cores. On real Klein activations each linear is 0.5–1.3% from bf16 (fp8
row-scaled: 1.5–2.8%), and the images match bf16's. The VAE stays f32. As in BFL's int8 release,
the image and joint blocks declare 8-bit attention (`attention_bits`; Runtime's `kitchen-int8` on
RTX 30/40); the text and reference early blocks, and `--lane 16`, keep exact attention.

```sh
cozy model quantize <account>/flux-3-klein@0.1.0/bf16 --int8 --on <machine>
cozy model publish <account>/flux-3-klein --release 0.1.0 --lane int8=sha256:<checkpoint>
```

Do the same for `flux-3-klein-base` and `flux-3-klein-upsampler`: each lane is one recipe that reads
which checkpoint it was given (`--nvfp4` writes the 4-bit lane the same way). Warm 1 MP seconds per step (16-step distilled; 50-step base
with CFG), and the weights the lane holds:

| card | distilled bf16 → int8 | base bf16 → int8 | weights |
|---|---|---|---|
| RTX 5090 | 0.245 → 0.146 | 0.496 → 0.295 | 17.6 → 9.4 GiB |
| RTX 4090 | 0.390 → 0.255 | 0.781 → 0.506 | |

## Licence

FLUX.3 Image Klein is under the FLUX Kommunity License v1.0 (`flux_3_klein/LICENSE.model.md`,
`NOTICE.model.md`): non-commercial use, plus commercial outputs for Qualifying Users under the terms
there. Its Qwen3-VL text encoder and the prompt upsampler are Apache-2.0. This wrapper grants no model
rights.
