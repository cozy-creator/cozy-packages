# Higher-resolution H3 video

Direct `ref2va` and `ref2va_turbo` calls also accept the long-form `references`
syntax. A reference with an `image` is passed through as supplied by default;
`prepare_references: true` runs BiRefNet for supplied character images, and
`describe_references: true` adds a one-sentence description of each undescribed
supplied character, as in long-form. The prompt is then compiled like one long-form
segment. Descriptions without images are generated once with Qwen Image 2.1. The resolved
assets are then handed to the H3 child, so preprocessing is not repeated. Send
either `references` or attached `assets`; a references-only request sends `"assets": []`.

```json
{
  "prompt": "<Rover> waves in a bright studio.",
  "duration_s": 6,
  "references": [
    {"name": "Rover", "kind": "character", "image": "./rover.png"},
    {"name": "Studio", "kind": "scene", "description": "A bright studio."}
  ],
  "assets": [],
  "prepare_references": true,
  "describe_references": true
}
```

FL2VA and Ref2VA, including their Turbo variants, accept `resolution` as a closed enum:
`768p`, `1080p`, `1440p`, or `2160p` (4K UHD). The default remains the original 1344×768
output. Higher tiers deliver 1920×1080, 2560×1440, or 3840×2160 with aspect-preserving
fitting and narrow letterboxing where necessary; the original 7:4 frame is never stretched.

Choose `upscale_method: seedvr2` (default, SeedVR2 3B) or `seedvr2_7b` (SeedVR2 7B) for
video restoration, or `upscale_method: h3_latent` for learned latent upscaling followed by
two standard H3 refinement steps. Original audio and frame timing are retained. H3 refinement uses a
suffix of the existing standard 30-step schedule, even when the original was Turbo.
It does not apply arbitrary extra steps to the PDD-8 adapter.

Higher-resolution calls publish the original 768p video before the upscale finishes.
CPU orchestration runs each model as a separate GPU child. Omit `--gpus` so the single-GPU
upscalers and multi-GPU H3 stages can each use their supported count. An explicit count
applies to every GPU child and is never silently reduced.

```sh
cozy run paul/minimax-h3/ref2va_turbo --asset character.png \
  prompt='<Picture 1> waves in a bright studio.' duration_s=15 seed=581203 \
  resolution=1440p upscale_method=h3_latent --await
```

`compare_fl2va_upscalers` and `compare_ref2va_upscalers` reuse one original generation
for every selected method and resolution. Supply the same inputs plus a required seed,
optionally `resolutions: [1080p, 1440p, 2160p]` (the default),
`methods: [h3_latent, seedvr2, seedvr2_7b]` (the default), and `mode: standard` (default
`turbo`). They publish `original` and one video per method and resolution, labeled by both. Every result uses the exact same original
generation. A failed variant records its error and leaves the other comparisons running.

The requested 15 seconds follows H3's native frame grid (362 frames at 24 fps, about
15.083 seconds). All three comparison videos retain the same frames and audio clock.
Upscaling can restore or synthesize texture; it cannot guarantee correction of wrong
motion or character identity. Review moving faces, hands and clothing at normal speed.

Qualified 2026-10-13 on one RTX Pro 6000 with 15 s Turbo originals (seed 581203; FL2VA and
Ref2VA). Minutes per upscale of the 362-frame clip, and down-consistency PSNR (output area-scaled
back to 1344×768 against the original) for FL2VA / Ref2VA:

| method | 1080p | 1440p | 2160p |
|---|---|---|---|
| `h3_latent` | 3.1–3.6 min, 33.7 / 35.4 dB | 6.9–7.3 min, 35.1 / 35.6 dB | fails on one GPU |
| `seedvr2` (3B) | 6.8–8.0 min, 33.4 / 33.0 dB | 11.7–13.1 min, 36.2 / 34.7 dB | 26–30 min, 38.0 / 36.4 dB |
| `seedvr2_7b` | 7.4–8.2 min, 34.5 / 34.1 dB | 12.5 min, 38.1 / 36.2 dB | 27.6–27.7 min, 39.3 / 37.6 dB |

`seedvr2_7b` keeps faces closest to the original while adding detail; `seedvr2` is sharpest
but raises contrast; `h3_latent` is soft and can drift a reference identity slightly. 2160p
`h3_latent` needs more than one GPU: a 96 GB RTX Pro 6000 refuses its first DiT block and an
H200 faults in attention at about 866k tokens. Use `seedvr2` or `seedvr2_7b` for 2160p.
