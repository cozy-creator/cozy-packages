# HunyuanImage 3.0 Instruct

Tencent's HunyuanImage 3.0 Instruct: one 80B mixture-of-experts transformer (32 layers, 64 experts with 8
routed per token plus one shared, about 13B active) that reads text and images and makes images. It first
writes its chain of thought and a rewritten prompt token by token, then denoises the image (flow matching on
32-channel VAE latents) in the same transformer. One function makes images from text, or edits and fuses up to
three ordered reference images.

| function | checkpoint | default `steps` | `guidance` |
| --- | --- | --- | --- |
| `generate` | Instruct | 50 | 2.5, classifier-free (both halves run every step) |
| `generate_turbo` | Instruct-Distil | 8 | 2.5, the value the distilled model is conditioned on |

```sh
cozy run fidika/hunyuan-image-3/generate_turbo prompt='A paper boat sailing down a rain gutter at dusk' seed=7
cozy run fidika/hunyuan-image-3/generate prompt='Put the cat in a tiny astronaut suit' \
  --asset reference_images.0=cat.png
```

Fields: `prompt`, `reference_images` (0–3, in order), `aspect_ratio` (any `W:H`), `megapixels`, `steps`
(omitted: the default above), `guidance`, `seed` (random when omitted, returned) and `prompt_upsampling`
(default `true`). The result is the `image`, its `width` and `height`, the `seed`, and `prompt`: the caption the
image was made from, which is the model's rewrite when `prompt_upsampling` is on. There is no negative prompt:
the model's unconditional branch is the prompt dropped.

## Sizes

The model makes about 1 MP, at Tencent's sizes for `image_base_size` 1024: 33 ratios from 1:4 to 4:1 on a
64 px step (width + height = 2048), plus 1024×768, 1280×720 and their portraits. A request gets the size
nearest its ratio (Tencent's `get_target_size`): the `aspect_ratio` given, else the first reference's, else 1:1.
A ratio past 1:4 or 4:1, or a pixel count (`megapixels`, else the first reference's) more than 10% off the size
made, is reported as a `clamp` adjustment. Tencent's edit postprocess, which resizes the output to the
reference's exact ratio, is not applied: the output is the model's own size.

## Prompt upsampling

`prompt_upsampling` is the Instruct models' own think-then-recaption stage (`bot_task="think_recaption"`, Tencent's
default): the model thinks about the request and its references, rewrites the prompt, and draws from both. It
samples as Tencent's `generation_config.json` says (temperature 0.6, top-k 1024, top-p 0.95, at most 2048
tokens), from a generator seeded with `seed`, so a seed repeats the same thought. Off (`bot_task="image"`), the
prompt goes straight to the image stage.

## Implementation

`hunyuan_image_3/tencent/` is Tencent's code (`Tencent-Hunyuan/HunyuanImage-3.0` at `6e9113a6`, the same as the
checkpoints' remote code), vendored because it is not importable as a package (its `__init__` imports a
top-level `utils`, and its requirements pin Gradio). The changes are in its git history: no Transformers
model/generation/cache classes, the token and denoising loops written out over Tencent's own input preparation
and logits processors, the next token's logits only, and expert routing that gathers each expert's rows with
one sort instead of a mask per expert. As Tencent's `from_pretrained(torch_dtype="auto")` builds it, every module
runs in bfloat16, the VAE included, except the expert routers (float32). `scripts/hunyuan-image-3-model-proof.py`
checks the port against Tencent's code on CPU with a small random-weight model.

## Prepare the models

Upload each checkpoint as it is (168.5 GB each; neither repository is gated):

```sh
cozy model upload hf://tencent/HunyuanImage-3.0-Instruct@2ec2c78bee7d4b94157341fba86c4c2c7b1858b2 \
  <account>/hunyuan-image-3-instruct --on <machine>
cozy model upload hf://tencent/HunyuanImage-3.0-Instruct-Distil@c8ffd07206f1b843697606968196e8f59f8ff38c \
  <account>/hunyuan-image-3-instruct-distil --on <machine>
```

## Lanes

`int8` writes the default lane from the 16-bit checkpoint: every linear of the 32 decoder layers (attention, the
shared and the 64 routed experts: 99% of the parameters) and the LM head as `int8.convrot`, about 83 GB. `nvfp4`
puts the routed experts in NVFP4 (Blackwell), about 45 GB. The routers, embedding, vision tower, VAE and image
layers keep their source bytes.

```sh
cozy model quantize <account>/hunyuan-image-3-instruct@0.1.0/bf16 --int8 --on <machine>
cozy model quantize <account>/hunyuan-image-3-instruct@0.1.0/bf16 --nvfp4 --on <machine>
```

## Licence

Tencent Hunyuan Community License (`hunyuan_image_3/LICENSE.model.txt`, `NOTICE.model.md`). It does not apply in
the European Union, the United Kingdom or South Korea, and a licensee with more than 100 million monthly active
users needs Tencent's licence. This wrapper grants no model rights.
