# Long-form video

Use `long_form_cuts` for a story with new camera setups between shots. Use `long_form`
for deliberate continuation of one camera take.

## Stories with camera cuts

Both workflows use supplied reference images or generate missing images with Qwen-Image-2.1
through `paul/qwen-image-2/generate_image`. H3 receives every image plus its description.
`long_form_cuts` renders independent H3 shots using those images.
Define one to nine named characters and scenes. Characters receive a plain white
background; scene references describe the empty location. Each shot selects only the
names relevant to it. There is no feedback from generated shots, no previous-frame
conditioning, and no first/last-frame anchor. Every generated frame is retained at cuts.

```json
{
  "style": "Photorealistic live action with restrained color grading and natural skin texture.",
  "soundscape": "Rain and footsteps. English dialogue only where specified.",
  "music": "N/A",
  "references": [
    {"name": "Lena", "kind": "character", "description": "An adult woman with short black hair, a blue motorcycle jacket, black trousers and worn boots.", "image": "./characters/lena.png"},
    {"name": "Omar", "kind": "character", "description": "An adult man with curly dark hair, a green bomber jacket, grey trousers and brown boots."},
    {"name": "Depot", "kind": "scene", "description": "An abandoned rain-soaked train depot, rusted blue train cars, concrete pillars and yellow overhead lights at dusk."}
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
cozy run paul/minimax-h3/long_form_cuts --input story.json --rental=your-rental --await
```

The two model packages run on the selected machine. Installing a package alone does
not download model weights. Explicit model preparation can warm the machine; inference
also downloads missing models as a fallback. The CPU parent awaits all reference images
first, then awaits each video shot. Runtime handles model placement and memory between
the image and video models.

Use `{name}` in style, audio and shot prompts to link actions to the exact Subject label.
Reference names begin with a letter and contain only letters, digits, `_` or `-`.
Names and selections are case-insensitive; `Lena` and `lena` cannot be declared twice.
A shot must select every name it uses. All references, names and compiled prompt lengths
are checked before the first image generation. `description` (up to 1024 characters)
describes appearance and is shared with both Qwen and H3. Character studio instructions
are appended only to Qwen's prompt. With `image`, generation is skipped and the description
is optional. Without `image`, a nonempty description is required.

Creator resolves relative `image` filenames beside the `--input` JSON file, and also
accepts absolute paths and `~/`. PNG, JPEG and WebP use the ordinary verified asset
transfer path with 64 MiB encoded and 256 MiB decoded limits. HTTP URLs are not accepted:
download the file first. Alternatively omit `image` and use
`--asset references.0.image=/path/to/lena.png`; supplying both forms is an error.
Package code never opens client filenames or downloads supplied references itself.

`style` controls shared visual treatment; location appearance belongs in a scene
reference, and events/camera actions belong in shot prompts. Optional `soundscape`
and `music` provide shared audio direction. Set `music` to `N/A` for no score.
When omitted, audio sections defer to explicit shot instructions rather than inventing
speech or a score. Keep synchronized dialogue inside the shot prompt, using stable
speaker IDs and the official form `{Lena} (S1) says, <d>[English] Your move.</d>`.

The compiler emits all six official sections: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`, and
`non_diegetic_music`. Style opens the detailed description before `[Shot 1]`.
It preserves authored text; it does not run an LLM rewrite or infer speakers.
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

The previous `history_frames` approach is removed. Supplied images remain fixed across
shots; they are not sampled from generated video. Fixed references avoid recycling generated defects but do
not guarantee visual quality or identity; real inference remains a separate quality gate.

## Continuous takes

`long_form` renders one or more sequential segments using the same supplied/generated character and
scene references throughout. Each continuation also receives a completed audio/video
tail from its predecessor. The tail is decoded media, never a mid-denoising checkpoint.

The first segment can deliver 15 seconds. Later segments deliver at most 14 seconds
with 22 context frames, 13 seconds with 39, or 12 seconds with 56. Longer requested
durations are shortened automatically to fit; progress and result warnings report
the adjustment. `delivered_frames / fps` reports the actual total duration.
There is no padding or repetition added to make a shortened segment look longer.

This is a new input contract: reference `prompt` becomes `description`, and the
top-level `prompt` becomes `style`. Existing installed packages are unchanged. JSON
filenames require the matching Creator update; `--asset` uses the established path.

Only the assembled MP4 and final PNG are delivered. Reference images, per-segment clips,
and bounded private context assets remain under Runtime's ordinary child-result custody.
Inference is not memoized. A later-segment failure returns the completed portion with
`complete=false`; first-segment failure and cancellation remain terminal.

```json
{
  "references": [
    {"name": "rover", "kind": "character", "description": "A small red rover with four black wheels."},
    {"name": "stream", "kind": "scene", "description": "A woodland stream beside a small wooden bridge."}
  ],
  "style": "Photorealistic nature photography with soft natural light.",
  "soundscape": "Flowing water, light wind and quiet wheels. No voices.",
  "music": "N/A",
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
fit and is shortened to 12 seconds; 13 seconds plus 39 context frames fits. The default 22 is an
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


## Exact dialogue at an action point

Both `long_form` and `long_form_cuts` accept an optional `dialogue` list on each shot.
Each line names a selected **character** reference, an explicit language label, and
exact spoken text. Optional `delivery` stays outside the dialogue tags; `voiceover`
adds the official off-screen direction and a lips-closed instruction. Put each line's
one-based `{dialogue:N}` marker exactly once in the shot prompt where it should occur.
Marker position expresses narrative order, not a guaranteed timestamp.

[The complete two-shot garden example](examples/dialogue-garden.json) uses Mara and
a garden keeper. Its first prompt places `{dialogue:2}` before `{dialogue:1}`: Mara
therefore receives `(S1)` and Guard `(S2)`. Those IDs remain stable when the second
shot changes reference order and hence its local `<Subject N>` labels.

```sh
cozy run paul/minimax-h3/long_form --input examples/dialogue-garden.json --rental=your-rental --await
```

This is the proposed source contract, not an instruction to replace a running local
candidate. Public release installation remains held until its dependencies are published.
Generated media uses the normal package output folder.

Words, whitespace, punctuation and language are never translated or rewritten. Finish
the line with punctuation before any closing quotation mark; invalid lines, missing or
repeated markers, unknown/unselected speakers, and scene references used as speakers
refuse before any reference image is generated. Keep H3 tags out of structured `text`
and `delivery`. Quotes in visual prose are not guessed to be speech: the garden sign
in the example remains an on-screen sign.

Manual H3 dialogue remains supported with an empty `dialogue` list. A story must use
either structured dialogue or manual `<d>`/`(S1)` markup, so automatic IDs cannot collide
with hand-authored IDs. Manual `<d>` bodies are preserved, including literal braces.
Keep dialogue in the detailed shot description, not `soundscape` or `music`; the latter
sections describe ambience and audience-only score. The default direction follows explicitly described vocal cues and keeps ambience and
physical sounds between lines.
An empty dialogue list does **not** mean total silence. Only explicit soundscape `N/A`
requests total silence; music `N/A` requests no score.

### Raw markup for the currently installed older prompt API

This shot prompt needs no new `dialogue`, `style`, `soundscape` or `music` fields. Use it
inside an existing shot with selected character references named Mara and Guard:

```text
The camera follows {Mara} toward {Guard} beside the garden gate. A sign reads "Please close the gate." {Mara} (S1), in a warm natural voice, says: <d>[English] Good morning. May I come in?</d> {Guard} pauses. {Guard} (S2), in a low calm voice, says: <d>[English] Good morning. The garden is open.</d> They walk along the path. Leaves rustle and footsteps continue between lines. No background music and no additional speech.
```

Keep the same speaker IDs in later shots. This example does not modify the active
candidate or an existing request. Formatting matches the official
[base guide](https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/skills/h3-prompt-writing/references/base-en.txt)
and [reference guide](https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/skills/h3-prompt-writing/references/ref-en.txt).
It corrects an authoring mismatch; intelligible delivery and exact transcript fidelity
still require a real video/audio assessment. No prompt-only cure is claimed.
