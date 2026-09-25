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

`long_form` renders 1–8 shots in sequence and assembles them on the rental. Each shot
starts from the previous shot's final frame. Shared subject and sound descriptions
are repeated in every prompt.

The delivered files are the final assembled MP4 and final continuation PNG. Intermediate
shot clips and frames remain internal to the serving calls; no prefix directory is
produced or downloaded.

Four 15-second shots produce 1,445 frames at 24 fps (60.208 seconds); eight produce
2,889 frames (120.375 seconds). One repeated frame is removed at each join. This
is shot-to-shot continuation: appearance and sound can still drift between shots.

Save a request such as this as `shots.json`:

```json
{
  "shots": [
    {"prompt": "A red rover travels beside a woodland stream."},
    {"prompt": "The rover approaches a small wooden bridge."},
    {"prompt": "The rover crosses the bridge, with water below."},
    {"prompt": "The rover follows the stream into a clearing."}
  ],
  "subject_definitions": "A small red autonomous rover, with four black wheels.",
  "overall_soundscape": "Flowing water, light wind and quiet wheel noise. No voices.",
  "non_diegetic_music": "N/A"
}
```

Run it on an existing rental:

```sh
cozy run paul/minimax-h3/long_form --input shots.json --rental=your-rental --await --out ./video
```

`complete`, `delivered`, and `requested` report whether all requested shots finished.
Each shot's seed is optional; an omitted or null seed derives from the request and shot
index and stays stable when that request retries. Explicit zero is a valid seed.

A later-shot failure returns the assembled completed portion and its final frame with
`complete=false` and failure details. Failure on the first shot fails the request;
cancellation remains cancellation.

To generate a new sequence from the delivered endpoint, put only the new shots in
`next-shots.json` and attach the final frame:

```sh
cozy run paul/minimax-h3/long_form --input next-shots.json --rental=your-rental \
  --asset opening_frame=/path/to/continuation.png --await --out ./next-video
```

The new request renders its own shots and returns its own assembled video. Runtime's
existing request and child records own execution recovery; the package does not export
intermediate state as an output or memoize model inference.

Progress identifies the current shot and child phase. Overall progress weights planned
frame × step work, assembly frames, and saving the final image. It is not an elapsed-time
estimate. A partial delivery remains below 100% of the requested work.

The assembled video has a 256 MiB encoded-byte limit. Runtime decodes bounded media
events during assembly instead of holding the entire decoded video in memory.
