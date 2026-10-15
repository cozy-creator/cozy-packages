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
  aspect_ratio=16:9 seed=7 --await
cozy run fidika/flux-3-klein/edit prompt='Make it night, the lamp lit' \
  --asset reference_images.0=lighthouse.png --await
```

Fields: `prompt`, `seed` (random when omitted, returned), `steps`, and `prompt_upsampling`
(default `true`). Text-to-image takes `aspect_ratio` (default `1:1`) and `megapixels` (1, 2 or 4),
which use the Qwen Image 2 buckets (1 MP square is 1024×1024). Editing takes `reference_images` in
order. Its output takes the first reference's aspect ratio at the `megapixels` area, unless
`aspect_ratio` is given. The result returns the image, its size, the seed and `prompt`: the caption
the image was made from.

## Prompt upsampling

BFL's prompt upsampler is a Qwen3.5-9B fine-tune. It turns a short request, plus the references when
editing, into the full caption Klein is trained on. It runs first, as its own memoized stage, with
BFL's system prompts and decoding: greedy, presence penalty 1.5 for text-to-image and 0 for edits.
Text-to-image also tells it the canvas (`TARGET CANVAS: 16:9`). Set `prompt_upsampling=false` to send
the prompt as typed, which then should be a full scene caption.

## Sampling

The sampling follows BFL's reference code (`black-forest-labs/flux-3-image-klein` at `45e57c9`): Euler
steps with a constant shift of 6.93, and noise from the seed in bf16 with an fp32 state across steps.
The distilled model also embeds the timestep each step jumps to. The base model runs CFG with an empty
negative prompt. A lone reference is padded to the 16-pixel grid at up to 4 MP, with no upscaling.
With several references, each is cropped to the grid at up to 1 MP. The modules are Diffusers'
(main `7ecb1e1b`; Flux3 is in no release yet). `scripts/flux-3-klein-model-proof.py` checks the
package against Diffusers' Flux3 modular pipelines on tiny real modules.

## Prepare the models

Upload each repository as it is (no TensorFS profile matches it yet), then `normalize` it into the
served checkpoint. That is a zero-copy derivation: transformer, text encoder and VAE, without BFL's
`variants/` folder or the text-encoder tensors Klein never runs.

```sh
cozy model upload hf://black-forest-labs/flux-3-image-klein@8454017f9ebaada30c998fccd9f67a80c7580f68 \
  <account>/flux-3-klein --rental=<rental> --await
cozy run <account>/flux-3-klein/normalize model.source=<account>/flux-3-klein#sha256:<raw> \
  --upload-to <account>/flux-3-klein --rental=<rental> --await
```

Do the same for `flux-3-image-klein-base@1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4`. A raw upload of the
upsampler, `flux-3-image-klein-pup@8b069751eb5dcab23829d9c5eccb470eb8b8a4ed`, carries no configuration
or tokenizer. `normalize_upsampler` adds BFL's files, which ship in `flux_3_klein/upsampler/`. Publish
each normalized checkpoint as release `0.1.0`, lane `original`.

## Licence

FLUX.3 Image Klein is under the FLUX Kommunity License v1.0 (`flux_3_klein/LICENSE.model.md`,
`NOTICE.model.md`): non-commercial use, plus commercial outputs for Qualifying Users under the terms
there. Its Qwen3-VL text encoder and the prompt upsampler are Apache-2.0. This wrapper grants no model
rights.
