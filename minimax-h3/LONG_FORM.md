# Long-form video

Use `long_form_cuts` for a story with new camera setups between shots. Use `long_form`
for deliberate continuation of one camera take.

## Stories with camera cuts

`long_form_cuts` first generates fixed reference images with Qwen-Image-2.1 through
`paul/qwen-image-2/generate_image`, then renders independent H3 shots using those images.
Define one to nine named characters and scenes. Characters receive a plain white
background; scene references describe the empty location. Each shot selects only the
names relevant to it. There is no feedback from generated shots, no previous-frame
conditioning, and no first/last-frame anchor. Every generated frame is retained at cuts.

```json
{
  "prompt": "A cinematic action film at dusk. Realistic photography. Dialogue is in English.",
  "references": [
    {"name": "Lena", "kind": "character", "prompt": "An adult woman with short black hair, a blue motorcycle jacket, black trousers and worn boots."},
    {"name": "Omar", "kind": "character", "prompt": "An adult man with curly dark hair, a green bomber jacket, grey trousers and brown boots."},
    {"name": "Depot", "kind": "scene", "prompt": "An abandoned rain-soaked train depot, rusted blue train cars, concrete pillars and yellow overhead lights at dusk."}
  ],
  "shots": [
    {"references": ["Lena", "Omar", "Depot"], "duration_s": 10, "prompt": "Low tracking shot in {Depot}. {Lena} ducks under {Omar}'s swinging arm, slides over a bench and turns to face him. Rain splashes under their boots."},
    {"references": ["Lena", "Depot"], "duration_s": 5, "prompt": "New tight side angle in {Depot}. {Lena} catches her breath behind a concrete pillar, glances left and says, <d>[English] Your move.</d>"}
  ]
}
```

```sh
cozy package install paul/qwen-image-2
cozy package install paul/minimax-h3
cozy run paul/minimax-h3/long_form_cuts --input story.json --rental=your-rental --await --out ./film
```

The two model packages run on the selected machine. Installing a package alone does
not download model weights. Explicit model preparation can warm the machine; inference
also downloads missing models as a fallback. The CPU parent awaits all reference images
first, then awaits each video shot. Runtime handles model placement and memory between
the image and video models.

Use `{name}` in shared and shot prompts to link actions to the exact Subject label.
Reference names begin with a letter and contain only letters, digits, `_` or `-`.
A shot must select every name it uses; the shared prompt must therefore avoid names
absent from any shot. All references, names and compiled prompt lengths are checked
before the first image generation. The reference `prompt` is used only for image generation. An optional short
`description` (up to 256 characters) can identify the appearance to preserve in H3;
studio posing and image-generation instructions are not repeated in video prompts.
The compiler defines each character/environment separately and retains its visual
identity; it does not preserve the still image's pose, white backdrop or viewpoint.

Turbo is the default. `mode="standard"` enables the existing 30/40/50 step schedules.
Reference generation uses 40 steps and 1024×1024 PNGs. Reference and shot seeds are
optional and stable across attempts of the same request. There is no eight-shot cap;
each shot has H3's native 5–15 second bound and the assembled MP4 has a 256 MiB limit.

Only the assembled MP4 and final PNG are downloaded. Fixed reference images and
intermediate clips remain retained child results on the worker. Logs record reference
names and digests, selected subjects, seeds and native render paths. The parent reports
reference generation, individual shot progress, assembly and overall progress. A later
shot failure delivers the completed portion with `complete=false`; reference-generation
failure, first-shot failure and cancellation stay terminal.

The previous `history_frames` and uploaded-image reference fields are removed from this
API. Use ordinary `ref2va` for direct authored-image conditioning, or `long_form` below
for deliberate continuation. Fixed references avoid recycling generated defects but do
not guarantee visual quality or identity; real inference remains a separate quality gate.

## Continuous takes

`long_form` renders 1–8 sequential segments using the same generated character and
scene references throughout. Each continuation also receives a completed audio/video
tail from its predecessor. The tail is decoded media, never a mid-denoising checkpoint.

Only the assembled MP4 and final PNG are delivered. Reference images, per-segment clips,
and bounded private context assets remain under Runtime's ordinary child-result custody.
Inference is not memoized. A later-segment failure returns the completed portion with
`complete=false`; first-segment failure and cancellation remain terminal.

```json
{
  "references": [
    {"name": "rover", "kind": "character", "prompt": "A small red rover with four black wheels."},
    {"name": "stream", "kind": "scene", "prompt": "A woodland stream beside a small wooden bridge."}
  ],
  "prompt": "A continuous tracking shot. Flowing water, light wind and quiet wheels. No music or voices.",
  "context_frames": 22,
  "shots": [
    {"prompt": "{rover} travels beside {stream}."},
    {"prompt": "{rover} approaches the bridge beside {stream}."},
    {"prompt": "{rover} crosses the bridge at {stream}."},
    {"prompt": "{rover} follows {stream} into a clearing."},
    {"prompt": "{rover} slows beside {stream}."}
  ]
}
```

Run with `cozy run paul/minimax-h3/long_form --input shots.json --rental=your-rental
--await --out ./video`. Each shot accepts `prompt`, optional `seed`, optional `duration_s`
(default 10), and optional `references` (defaults to all declared names in declaration
order). Explicit reference selections control per-shot picture slots. `mode` defaults to
`turbo`; `standard` accepts 30, 40 or 50 `steps`. Turbo fixes eight PDD evaluations.

`duration_s` is new delivered time at 24 fps. Context and native alignment padding consume
additional sampled frames and are excluded from the delivered clip. The first 10-second
segment samples 243 frames and delivers 240. Subsequent 10-second segments sample:

| Context frames | Context seconds | Sampled frames | Delivered frames |
| --- | --- | --- | --- |
| 22 | 0.917 | 277 | 240 |
| 39 | 1.625 | 294 | 240 |
| 56 | 2.333 | 311 | 240 |

Five default segments deliver exactly 1,200 frames (50 seconds), without replay-frame
removal or visual crossfade. Every prompt and frame plan is validated before Qwen runs.
The native generation limit is 362 frames: 13 new seconds plus 56 context frames cannot
fit and is rejected; 13 seconds plus 39 context frames fits. The default 22 is an
experiment setting, not a quality claim. Longer context may improve continuity, but this
requires matched-source visual and listening comparisons. Cuts or identity/audio drift
remain possible; the API does not guarantee seamless video.

Context compatibility records the selected base model and adapter manifests. It does
not inspect package source or SDK versions. Runtime's existing
request and child records own recovery; this API does not accept an opening-frame-only
resume or export its private AV context.

Progress weights sampled frames × steps, reference generation, assembly frames and the
final image save. The assembled MP4 has a 256 MiB encoded-byte limit; assembly decodes
bounded media events rather than the complete film at once.
