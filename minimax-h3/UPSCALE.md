# Higher-resolution H3 video

Direct `ref2va` and `ref2va_turbo` calls also accept the long-form `references`
syntax. A reference with an `image` is passed through as supplied by default;
`prepare_references: true` runs BiRefNet for supplied character images.
Descriptions without images are generated once with Qwen Image 2.1. The resolved
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
  "prepare_references": true
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

This source is under qualification; the 1080p, 1440p and 2160p GPU comparison results
are not established until the accompanying benchmark receipts are complete.
