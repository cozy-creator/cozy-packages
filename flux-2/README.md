# FLUX.2

Black Forest Labs' FLUX.2 open weights in one package: [dev] (32B, Mistral-Small-3.2 text encoder), fal's
[dev] Turbo LoRA, and [klein] 9B and 4B (Qwen3 text encoders), each distilled or Base. One function per
speed generates an image, or edits when `reference_images` are given (up to ten, in order).

| function | default model | other models (`model.model=`) |
| --- | --- | --- |
| `generate` | [dev]: 50 steps, guidance 4 | [klein] Base: `flux-2-klein-9b-base`, `flux-2-klein-4b-base` (50 steps, CFG 4) |
| `generate_turbo` | [dev] + fal's Turbo LoRA: 8 steps, guidance 2.5 | [klein]: `flux-2-klein-9b`, `flux-2-klein-4b` (4 steps, no guidance) |

The checkpoint sets the defaults, so any model runs correctly in either function. Klein 4B (Apache-2.0,
8 GB of weights at int8) is the choice for consumer cards:

```sh
cozy run fidika/flux-2/generate_turbo model.model=fidika/flux-2-klein-4b \
  prompt='A lighthouse on a basalt cliff in a winter storm' aspect_ratio=16:9
cozy run fidika/flux-2/generate_turbo model.model=fidika/flux-2-klein-9b \
  prompt='Make it night, the lamp lit' --asset reference_images.0=lighthouse.png
cozy run fidika/flux-2/generate model.model=fidika/flux-2-klein-4b-base prompt='...' negative_prompt='blurry'
cozy run fidika/flux-2/generate prompt='...'          # [dev]
cozy run fidika/flux-2/generate_turbo prompt='...'    # [dev] Turbo
```

Fields: `prompt`, `reference_images`, `aspect_ratio` (any `W:H`), `megapixels` (1, 2 or 4; 1024² pixels
each), `steps` (default as above; Turbo always runs its 8), `seed` (random when omitted, returned),
`guidance` ([dev]'s embedded guidance, or [klein] Base's CFG scale), `negative_prompt` ([klein] Base) and
`prompt_upsampling` (default off). Unknown fields are dropped with a warning. The result is `image`,
`width`, `height`, `seed` and `prompt`, the prompt actually used.

Text-to-image defaults to 1:1 at 1 MP (1024×1024). With references and no size, the output takes the first
reference's ratio and pixel count; `aspect_ratio` or `megapixels` overrides only itself. Sizes snap to
16 px; sides stay within 64–4096 px and the area within 4 MP, reported as a `clamp` adjustment.

"FLUX.2 [dev] Flash" is fal's hosted endpoint (`fal-ai/flux-2/flash`, "timestep distillation"); its
weights are not published, so it is not here.

## Pipeline

It follows BFL's reference code (`black-forest-labs/flux2` 50fe516): the text encoders' hidden states
10/20/30 (Mistral) and 9/18/27 (Qwen3) over 512 tokens; references cropped to the 16 px grid at up to
2024² pixels for one, 1024² each for several, at time offsets 10, 20, …; Euler steps in bf16 on BFL's
schedule (linear times shifted by the token count's empirical mu); [klein] Base's CFG against
`negative_prompt` (BFL's is empty). Turbo runs fal's card as written: its sigmas
`1, 0.6509, … 0.00031` through Diffusers' `Flux2Pipeline`, which shifts them by mu as well, at guidance
2.5. The modules are Diffusers' (main 7ecb1e1b). `scripts/flux-2-model-proof.py` checks every checkpoint
kind against Diffusers' pipelines on tiny real modules.

`prompt_upsampling` is BFL's local upsampler: [dev]'s Mistral-Small-3.2 as the full vision-language model,
with BFL's system prompts, used for [klein] too. It decodes greedily (BFL samples at temperature 0.15).
Its first use fetches [dev]'s text encoder.

## Prepare the models

Upload each repository as it is, and publish it as lane `bf16`:

```sh
cozy model upload hf://black-forest-labs/FLUX.2-dev@26afe3a78bb242c0a8bb181dcc8937bb16e5c66c fidika/flux-2-dev --on <machine>
cozy model upload hf://fal/FLUX.2-dev-Turbo@9ee51cd87578162cf8d02355a870bc5f4570045c fidika/flux-2-dev-turbo --on <machine>
cozy model upload hf://black-forest-labs/FLUX.2-klein-9B@92196c8e11f7b6cf2b7493e037d8c5345c559216 fidika/flux-2-klein-9b --on <machine>
cozy model upload hf://black-forest-labs/FLUX.2-klein-base-9B@32773329fbe7e81a90ef971740e8ba4b0364ecf3 fidika/flux-2-klein-9b-base --on <machine>
cozy model upload hf://black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294 fidika/flux-2-klein-4b --on <machine>
cozy model upload hf://black-forest-labs/FLUX.2-klein-base-4B@a3b4f4849157f664bdbc776fd7453c2783562f4d fidika/flux-2-klein-4b-base --on <machine>
```

The Turbo LoRA stays one separate 16-bit model: Runtime bakes it into [dev]'s resident weights at load.

## Lanes

`int8` (the default lane) writes every linear of the DiT's blocks, its modulation heads, context embedder
and output norm, and the text encoder's layer projections ([dev]'s LM head too) as `int8.convrot`, W8A8
on int8 tensor cores. `nvfp4` (`--lane 4`, Blackwell) puts the DiT blocks in NVFP4 beside the same int8
rest. The VAE, embedders and output projection keep their bytes. Each lane is written from the 16-bit
upload:

```sh
cozy model quantize fidika/flux-2-dev@0.1.0 --int8 --on <machine>
cozy model quantize fidika/flux-2-dev@0.1.0 --nvfp4 --on <machine>
```

Measured on an A40 (int8 W8A8), seconds per step at 1 MP: [dev] 1.93, [klein] 4B 0.82 (bf16) → 0.62,
[klein] 9B Base with CFG 1.20. Against Diffusers' own bf16 pipeline on the same weights and noise, this
package's bf16 is 40.6 dB (4B edit) and 30.9 dB (9B, two references), int8 31.6 and 27.0 dB: the same
images. A 4-step [klein] text-to-image differs more (21 dB: the same composition, slightly shifted):
it embeds the timestep as BFL does, in bf16, where Diffusers rounds `t·1000` first.

## Licence

FLUX.2 [dev], [klein] 9B and 9B Base are under the FLUX Non-Commercial License
(`flux_2/LICENSE.model.md`, `NOTICE.model.md`); so is fal's Turbo LoRA. [klein] 4B and 4B Base are
Apache-2.0. Generated outputs are yours to use under those licences' terms; this wrapper grants no model
rights.
