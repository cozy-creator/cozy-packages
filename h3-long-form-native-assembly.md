# H3 native long-form delivery

The CPU parent awaits ordinary internal serving calls and assembles their clips on the
rental with Runtime MediaDecoder and Outputs. It returns the final MP4 and the last
segment's continuation PNG, forwarded unchanged, plus delivery status. No intermediate directory or shot receipt list
is part of the public result.

One validation worker scans each completed clip's soundtrack while the next shot renders.
Cuts copy every clip's H.264 packets through Runtime's `save_video_concat` and encode only
the joined soundtrack, trimmed or padded per clip to the exact frame clock; Runtime checks
the frame clock and codec parameters on the packets. A continuous join removes one
replayed frame at each seam and re-encodes; so do a master track and unjoinable tracks. A later child failure
assembles the completed portion; first-child failure and cancellation remain terminal.

Runtime's existing child records own execution recovery. There is no package prefix archive,
resume bundle, additional storage authority, workflow DSL, or inference memoization.

The CPU broker/codec proof checks a seven-shot video, exactly one parent-owned output
asset, partial delivery, frame continuation, progress, seeds,
and cancellation. It does not establish H3 generation quality or GPU timing.
