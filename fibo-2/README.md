# Fibo 2.0

Bria's Fibo 2.0: one 8B flow-matching DiT for text-to-image and image editing, trained on structured
JSON captions. It has 44 single-stream blocks over the FLUX.2 VAE's latents. Qwen3-VL-4B encodes the
prompt and the images to edit, and a Perceiver turns it into 256–512 gist tokens; five blocks also
cross-attend to Qwen3-VL's inner layers. It ranked #1 for editing and #2 for text-to-image among
open-weights models on Artificial Analysis (2026-10-20).

| function | checkpoint | default steps | guidance |
| --- | --- | --- | --- |
| `generate` | Fibo 2.0 Dev | 50 | `guidance` 5.0 (CFG), `negative_prompt` |
| `generate_turbo` | Fibo 2.0 Edit Turbo, distilled from Dev | 4 | none |

Both generate from text, or edit when `reference_images` are given (up to four, `<image_1>`…`<image_4>`).
Bria ships Edit Turbo for editing; it also renders text-to-image.

```sh
cozy run fidika/fibo-2/generate prompt='An origami whale swimming through paper clouds' aspect_ratio=16:9 seed=7
cozy run fidika/fibo-2/generate_turbo prompt='Replace the jacket with a red leather one' \
  --asset reference_images.0=photo.jpg
```

Fields: `prompt`, `reference_images`, `aspect_ratio` (any `W:H`), `megapixels`, `steps`, `seed` (random
when omitted, returned) and `prompt_upsampling` (default `true`); `generate` also takes `guidance` and
`negative_prompt`. Fibo 2.0 was trained at about 1 MP on a 16 px grid (model card). Text-to-image
defaults to 1:1 at 1024×1024. With references, the output takes the first one's aspect ratio and pixel
count; `aspect_ratio` or `megapixels` replaces only that one. The size is held to 0.25–1 MP and
64–8192 px sides, reported as a `clamp` adjustment. A lone reference is cropped to the output's size;
several are each cropped to about 1 MP at their own ratio (Diffusers). The result returns `image`, its
size, the `seed` and `prompt`: the prompt the image was made from.

## Prompts

Fibo 2.0 reads a JSON caption (Bria's VGL), a short description, or for an edit the instruction alone.
A JSON `prompt` is used as given; an edit's needs an `edit_instruction`. Otherwise
`prompt_upsampling` writes the caption with Bria's prompt-to-JSON VLMs, the local route Diffusers
points to (Bria's own API path, `bria-vgl`, calls Gemini):

- text-to-image: `briaai/FIBO-vlm` (`<generate>`);
- one reference: `briaai/FIBO-edit-vlm` (`<edit>`, the image thumbnailed to 1024 px), whose restated
  request (`reworded_prompt`) becomes the caption's `edit_instruction`;
- several references: neither VLM reads more than one image, so the instruction goes as typed, with a
  warning.

Both are Qwen3-VL-4B fine-tunes, decoded greedily (Bria samples at temperature 0.2/0.4), each fetched
on first use. `prompt_upsampling=false` sends the prompt as typed. The returned `prompt` reproduces
the image with `prompt_upsampling=false` and the same seed.

## Sampling

Diffusers' `BriaFibo2Pipeline` (main `48cdbaef`, #14991, verified by Bria against its reference code),
run per component. Euler from sigma 1 to 1/steps, shifted by the image size; float32 latents; noise
from the seed on the CPU. Dev runs CFG 5 at 50 steps (the model card; Diffusers' default is 30), with
the card's text-to-image negative prompt, or the empty caption the model was trained with for edits.
Turbo runs 4 steps at guidance 1. The caption's text is also drawn and shown to Qwen3-VL, in the fonts
training used (`fibo_2/fonts`). `scripts/fibo-2-model-proof.py` checks that the package makes
Diffusers' image on tiny real modules.

## Prepare the models

Upload each repository as it is, and publish each as lane `bf16`. The Fibo repositories are gated:
accept Bria's licence on Hugging Face first.

```sh
cozy model upload hf://briaai/Fibo-2.0-dev@<rev> <account>/fibo-2-dev --on <machine>
cozy model upload hf://briaai/Fibo-2.0-edit-turbo@<rev> <account>/fibo-2-edit-turbo --on <machine>
cozy model upload hf://briaai/FIBO-vlm@bd13da5e0b73c808014d6260963e311c8cdf0d8b <account>/fibo-vlm --on <machine>
cozy model upload hf://briaai/FIBO-edit-vlm@61d852ebd380ac79e04ec171552f5d7cca1a04cc <account>/fibo-edit-vlm --on <machine>
```

## Quantized lanes

`int8` writes the default lane from a 16-bit release: the DiT's attention linears and FFN gate/up
projections (blocks, noise refiner, Perceiver) and Qwen3-VL's language linears as `int8.convrot`, W8A8
on the int8 tensor cores. The FFN down projections (9,536 columns, not a multiple of ConvRot's 256),
adaLN modulation, embedders, final layer, vision tower and VAE keep their bytes. For a VLM it takes the
language linears. `nvfp4` puts the DiT's 44 blocks, down projections included, in NVFP4 (Blackwell),
and a VLM's linears in NVFP4 (W4A16 decode).

```sh
cozy model quantize <account>/fibo-2-dev@0.1.0/bf16 --int8 --on <machine>
cozy model publish <account>/fibo-2-dev --release 0.1.0 --commit sha256:<commit>
```

## Licence

Fibo 2.0, FIBO-vlm and FIBO-edit-vlm are Bria AI's, under CC BY-NC 4.0 (`fibo_2/LICENSE.model.md`):
non-commercial use; commercial use needs an agreement with Bria. This wrapper grants no model rights.
