# H3 assembly overhead

Long-form currently decodes every completed video to validate frame/sample clocks and
measure audio levels, then decodes them again to encode the joined MP4. All of this
begins after the final GPU child completes.

The first change moves the bounded validation scan of each completed shot onto one
worker while the next child runs. Assembly consumes those private scan results without
rescanning. Retained prefixes take the same validation path. Corrupt input, mismatched
frames, changed intent and cancellation must remain failures; later child failure still
assembles every completed prefix segment.

Encoding still needs all audio seam levels and a global peak bound. Preserving that
soundtrack requires either deferring encoding or a Runtime writer supporting independently
spooled video and audio. Package code must not bypass Runtime output custody or substitute
an unbounded packet buffer.

Acceptance: native codec and child-broker fixtures prove identical frame/sample clocks,
encoded bytes, bounded scan memory, rejection arms and scan/child overlap. These CPU
fixtures do not establish real H3 rendering or the end-to-end 35-minute target.
