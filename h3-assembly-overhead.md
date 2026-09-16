# H3 assembly overhead

Long-form currently decodes every completed video to validate frame/sample clocks and
measure audio levels, then decodes them again to encode the joined MP4. All of this
begins after the final GPU child completes.

The first change moves the bounded validation scan of each completed shot onto one
worker while the next child runs. Assembly consumes those private scan results without
rescanning. Every completed shot takes the same validation path. Corrupt input, mismatched
frames and cancellation must remain failures; later child failure still
assembles every completed prefix segment.

Encoding still needs all audio seam levels and a global peak bound. Preserving that
soundtrack requires either deferring encoding or a Runtime writer supporting independently
spooled video and audio. Package code must not bypass Runtime output custody or substitute
an unbounded packet buffer.

Acceptance: native codec and child-broker fixtures prove identical frame/sample clocks,
encoded bytes, bounded scan memory, rejection arms and scan/child overlap. These CPU
fixtures do not establish real H3 rendering or the end-to-end 35-minute target.

Implemented and locally qualified:

- One validation worker overlaps completed-shot scans with subsequent broker children,
  and the scope joins its worker on cancellation or failure.
- Final assembly consumes the scans once. Native 1/2/4/8-shot outputs remain byte-identical
  to serial assembly, with the same exact video and audio clocks and bounded audio windows.
- Malformed clips are refused. Partial child failure, common renderer provenance,
  and cancellation retain their contracts.
- Unity-gain audio chunks keep their original PCM bytes; peak/window scans avoid redundant
  Python conversion. A scalar reference proves gain output bytes remain unchanged.

The local comparison using two already-delivered H3 clips repeated into a four-shot
60.2-second video produced the same MP4 digest before and after. Serial elapsed time was
96.5 seconds before and 102.1 seconds after; scan time was 26.3 and 28.4 seconds. Other proof
processes shared the CPU, so this is byte-integrity evidence, not an intrinsic speedup
claim. The measured structural improvement is scan/child overlap; the final encode still
waits for all soundtrack levels. GPU workflow timing remains unqualified.

The same workflow also retained Qwen's image-position `rope_deltas` after a complete
conditioning forward. This autoregressive cache has no H3 consumer (`use_cache=False`)
and causes Runtime's next-sample residency check to reject reuse. `condition_text` now
clears it in `finally`, including failed conditioning. The native package conformance arm
runs the actual truncated 50-layer Qwen model four times, proves every hidden state remains
bit-identical to upstream, and verifies cleanup on success and failure. Runtime separately
measures the 512-byte CUDA allocator residue and its removal.
