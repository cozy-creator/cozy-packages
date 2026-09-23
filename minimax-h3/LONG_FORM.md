# Long-form video

Use `long_form_cuts` for a story with new camera setups between shots. Use `long_form`
for deliberate continuation of one camera take.

## Stories with camera cuts

`long_form_cuts` generates each shot with native Ref2VA appearance references. References
do not become pinned first or last frames. Every generated frame is retained at the cut.
The worker assembles the film and returns only the final MP4 and last PNG.

```json
{
  "prompt": "A small red rover with four black wheels travels beside a woodland stream. Natural light, realistic textures, water and wheel sounds, no speech.",
  "shots": [
    {"prompt": "Wide view across the stream as the rover approaches a wooden bridge."},
    {"prompt": "Low side angle beside the bridge. The rover climbs onto the boards."},
    {"prompt": "Overhead camera looking straight down as the rover crosses the bridge."},
    {"prompt": "Close view from ahead of the rover as it reaches a sunlit clearing."}
  ]
}
```

```sh
cozy run paul/minimax-h3/long_form_cuts --input story.json --rental=your-rental --await --out ./film
```

`references` optionally supplies up to nine stable image references. Each item has an
`image`, a required `description` of relevant appearance, an optional `subject` name,
and optional `fidelity` (`auto`, `low`, `medium`, `high`). Give pictures of the same person,
prop or location the same subject name so they define one subject from several views.
Descriptions should identify the appearance to preserve; each shot prompt specifies its
own current action, pose, camera and scene state. The wrapper compiles ordered native
Picture numbers inside Subject definitions. Fidelity controls image presentation size,
not reference strength. A photograph containing two named subjects occupies one Picture
slot with two Subject definitions. Avoid unnecessary global references: they may constrain
a scene even when its prompt does not mention them.

`history_frames` defaults to one and accepts zero through six. Stable references reserve
slots first. History selects recent clips' middle frames before extra quarter and
three-quarter views, with exact duplicate and coarse thumbnail similarity suppression.
This is a provisional selection policy, not an image quality or semantic relevance judge.
The nine-image ceiling and Runtime's reference token budget still apply. At nine stable
slots, no history is captured. With no supplied images, the first shot uses the existing
text-only path; later shots can use its generated references. With no images and
`history_frames=0`, every shot is an independent text-only generation.

Turbo is the default. `mode="standard"` enables the existing 30/40/50 step schedules.
Seeds are optional and reproducible across attempts of the same request. The parent
reports each shot's child progress and overall work. A later failure delivers the
completed portion with `complete=false`; first-shot failure and cancellation stay terminal.

Reference PNGs, selection records and intermediate clips stay internal on the worker.
Shot logs record reference digests, source frame indices, subject roles, seeds and native
render paths. Three candidate frames come directly from each needed shot's existing RGB8
decode, without decoding its compressed MP4 again. Generated references can still carry
artifacts or stale state. Shared voices, music and identity are not guaranteed across
independent generations. CPU contract checks do not establish improved H3 video quality.

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
