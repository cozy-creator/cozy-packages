# H3 native long-form delivery

The CPU parent awaits ordinary internal serving calls and assembles their clips on the
rental with Runtime MediaDecoder and Outputs. It returns the final MP4 and final
continuation PNG, plus delivery status. No intermediate directory or shot receipt list
is part of the public result.

One validation worker scans completed clips while the next shot renders. The assembler
preserves the frame/sample clock, validates common media formats, removes one replayed
frame at each continuous join, and checks the encoded output. A later child failure
assembles the completed portion; first-child failure and cancellation remain terminal.

The final image can be supplied as `opening_frame` for a new shot sequence. Runtime's
existing child records own execution recovery. There is no package prefix archive,
resume bundle, additional storage authority, workflow DSL, or inference memoization.

The CPU broker/codec proof checks a seven-shot video, exactly two parent-owned output
assets, final-frame identity, partial delivery, frame continuation, progress, seeds,
and cancellation. It does not establish H3 generation quality or GPU timing.
