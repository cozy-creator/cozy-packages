# FLUX.3 Image Klein

Black Forest Labs' FLUX.3 Image Klein: a 5.6B DiT for text-to-image generation and editing with up
to ten reference images. Its text encoder is Qwen3-VL-4B.

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | Klein (distilled flow map) | 16 | none |
| `edit` | Klein (distilled flow map) | 16 | none |
| `generate_base` | Klein Base | 50 | `guidance` 3.5 (CFG) |
| `edit_base` | Klein Base | 50 | `guidance` 3.5 (CFG) |

Use the distilled functions for generation. The base checkpoint is meant for fine-tuning and research.

```sh
cozy run fidika/flux-3-klein/generate prompt='A lighthouse on a basalt cliff in a winter storm' \
  aspect_ratio=16:9 seed=7
cozy run fidika/flux-3-klein/edit prompt='Make it night, the lamp lit' \
  --asset reference_images.0=lighthouse.png
```

Fields: `prompt`, `seed` (random when omitted, returned), `steps`, and `prompt_upsampling`
(default `true`). Text-to-image takes any `aspect_ratio` `W:H` (default `1:1`) and `megapixels`
(1, 2 or 4; 1024² pixels each). The default is 1024×1024, the size BFL's reference code makes
(`default_size`); the model card names no native size, and BFL's README says up to 4K. The size is
the tier's area at that ratio, rounded to 32 px. Sides stay within BFL's 64–4096 px. A ratio past that
is served at the nearest size it allows, reported as a `clamp` adjustment. Editing takes
`reference_images` in order. As in BFL's code, with no size given the output is the first prepared
reference's own size: a lone reference fitted to at most 4 MP, or the first of several at up to 1 MP on
the 16 px grid. An `aspect_ratio` or `megapixels` sets the size instead; a missing one comes from the
first reference (its ratio, or the tier nearest its pixel count). The result returns the image, its size, the seed and `prompt`: the caption
the image was made from.

## Prompt upsampling

BFL's prompt upsampler is a Qwen3.5-9B fine-tune. It turns a short request, plus the references when
editing, into the full caption Klein is trained on. It runs first, as its own memoized stage, with
BFL's system prompts and decoding: greedy, presence penalty 1.5 for text-to-image and 0 for edits.
Text-to-image also tells it the canvas (`TARGET CANVAS: 16:9`). Set `prompt_upsampling=false` to send
the prompt as typed, which then should be a full scene caption. Its token loop declares
`min_bits=4, phase="decode"`, so it serves the release's `nvfp4` lane where there is one.

## Sampling

The sampling follows BFL's reference code (`black-forest-labs/flux-3-image-klein` at `45e57c9`): Euler
steps with a constant shift of 6.93, and noise from the seed in bf16 with an fp32 state across steps.
The distilled model also embeds the timestep each step jumps to. The base model runs CFG with an empty
negative prompt. A lone reference is padded to the 16-pixel grid at up to 4 MP, with no upscaling.
With several references, each is cropped to the grid at up to 1 MP. The modules are Diffusers'
(main `7ecb1e1b`; Flux3 is in no release yet). `scripts/flux-3-klein-model-proof.py` checks the
package against Diffusers' Flux3 modular pipelines on tiny real modules.

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

Do the same for `flux-3-klein-base`. Warm 1 MP seconds per step (16-step distilled; 50-step base
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
