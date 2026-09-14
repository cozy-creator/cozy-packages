# Long-form video

The long-form entrypoint renders 2–8 shots in sequence. Each shot starts from the
previous shot's final frame. Shared subject and sound descriptions are repeated in
every prompt. It returns a playable video and a prefix directory containing the
original clips, continuation frames, and their recorded rendering inputs.

Four 15-second shots produce 1,445 frames at 24 fps (60.208 seconds); eight produce
2,889 frames (120.375 seconds). One repeated frame is removed at each join. This
is shot-to-shot continuation: appearance and sound can still drift between shots.

Save a request such as this as `shots.json`:

```json
{
  "shots": [
    {"prompt": "A red rover travels beside a woodland stream.", "seed": 101},
    {"prompt": "The rover approaches a small wooden bridge.", "seed": 102},
    {"prompt": "The rover crosses the bridge, with water below.", "seed": 103},
    {"prompt": "The rover follows the stream into a clearing.", "seed": 104}
  ],
  "subject_definitions": "A small red autonomous rover, with four black wheels.",
  "overall_soundscape": "Flowing water, light wind and quiet wheel noise. No voices.",
  "non_diegetic_music": "N/A"
}
```

Run it on an existing rental:

```sh
cozy run paul/minimax-h3/long_form --in shots.json --rental=your-rental --await --out ./video
```

`complete`, `delivered`, and `requested` report whether all requested shots finished.
If a later shot fails, the result contains a playable partial video, its prefix, and
the failure details. Failure on the first shot fails the request. Cancellation
remains cancellation.

To extend or recover a delivered prefix, keep its complete directory, set
`resume_from` to a reference such as `"retained-prefix"` in the new request, and
bind that reference to the collected prefix directory:

```sh
cozy run paul/minimax-h3/long_form --in extended.json --rental=your-rental \
  --input-tree retained-prefix=/path/to/collected/prefix --await --out ./extended-video
```

Include the original completed shots at the start of `extended.json`. Their
prompts, seeds, durations, shared descriptions, step count and opening frame must
match. You may edit the remaining shots or append more shots, up to eight total.
The response's `reused` count identifies shots that were kept. Supplying a complete
prefix with the same shot list assembles it again without rendering any shot.

The prefix records the exact model manifest, package code, and rendering software.
An extension refuses a different rendering cohort instead of describing old clips
as if they were newly generated. This is explicit reuse of completed clips;
ordinary inference is not memoized. Keep the matching package/runtime cohort when
extending a prefix.

The final video and the prefix each have a 256 MiB encoded-byte limit. Runtime
decodes bounded media events during assembly; it does not hold a complete decoded
multi-minute video in memory.
