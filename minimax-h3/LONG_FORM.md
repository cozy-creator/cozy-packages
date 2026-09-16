# Long-form video

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
