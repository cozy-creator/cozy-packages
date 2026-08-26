"""THE ONE PIXEL CONVERSION. Float decode to display bytes, applied exactly once.

se-002's first render applied `x / 2 + 0.5` to a tensor the video VAE had ALREADY returned
in [0, 1] (`video_vae._finalize_pixels` un-normalizes by the ImageNet statistics and clamps,
which is upstream's convention and diffusers' too). The second rescale is not a rounding
difference: it folds the whole range into [0.5, 1.0], so BLACK BECOMES MID-GREY and no
correct render can ever be dark (#522d). It survived because the conversion was an inline
expression at the call site with nothing to check it against.

So the conversion is a NAMED function with a stated input range, and it is written in the
array-API subset that numpy and torch spell identically — `.clip(lo, hi)`, `*`, `.round()`.
That is not a stylistic choice: it is what lets the conformance arm exercise THIS function
on a real array with no torch installed, instead of a paraphrase of it that can drift.

The cast to uint8 stays with the caller, because that is the one operation the two
libraries genuinely spell differently, and because the caller is also the one that knows
whether it wants a tensor or an ndarray back.
"""

from __future__ import annotations

from typing import Any

#: What the video VAE's decode returns. Not a preference — `_finalize_pixels` clamps to it.
VAE_OUTPUT_RANGE = (0.0, 1.0)

#: Display full scale. Black is 0 and white is 255, and the arm checks both ends.
PIXEL_FULL_SCALE = 255.0


def pixel_bytes(clip: Any) -> Any:
    """[0, 1] float pixels -> [0, 255] float pixels, rounded to whole bytes.

    Round rather than truncate: truncation costs half a level of accuracy everywhere and
    buys nothing, and the two endpoints (0 and 255) land identically either way."""
    low, high = VAE_OUTPUT_RANGE
    return (clip.clip(low, high) * PIXEL_FULL_SCALE).round()
