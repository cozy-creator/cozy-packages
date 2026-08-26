"""The VISION SEAM — reference pixels to the text encoder's patch/grid/index triple.

`presentation.build` emits a `VisionBlock` row standing in for a whole block of vision
tokens, because how many tokens a block becomes is a fact of its RESOLVED GRID and only a
patchifier can state it. This file is that patchifier, plus the two things that bracket it:
the normalization upstream puts a reference image through before it is ever patchified, and
the expansion that turns a presentation carrying block rows into the flat token id sequence
the text encoder is actually called on.

THE PREPROCESSING IS UPSTREAM'S, NOT OURS (#531). Qwen3-VL's patch geometry — the 16-pixel
patch, the 2-frame temporal patch, the 2x2 spatial merge, the `smart_resize` that snaps a
canvas onto the patch grid inside a pixel-area budget — is implemented in `transformers`
and is the same code the official release runs. Reimplementing it here would be a second
authority on numbers that already have one, so `patchify` CALLS it. What this file states
itself are the values that configure it, which are facts about the released checkpoint in
exactly the sense `text_encoder._ROPE_SECTIONS` is: they change numerics without changing a
key, so they live in source rather than being inferred.

THE TWO CANVAS RULES ARE DIFFERENT, ON PURPOSE. A reference IMAGE is conditioned at high
detail — a short edge of its own, 2048 for the released checkpoint, upscaling included and
with NO area cap. The generated target and a reference VIDEO share the other rule, the 768
short edge under the 1,032,192-pixel budget that `_PIXELS` tabulates. An image reference
never binds the output geometry; it is read at its own resolution and the request generates
at whatever canvas it asked for.

Reference for every rule here: diffusers `modular_pipelines/minimax_h3/before_encoder.py`
(`MiniMaxH3Ref2VASetupStep`) and `encoders.py` (`MiniMaxH3Ref2VATextEncoderStep`), at the
pinned SHA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .presentation import Presentation, VisionBlock

#: The short edge a reference IMAGE is conditioned at. Upscaling included and with no area
#: cap — unlike the target canvas, which is capped. A released-checkpoint fact.
REFERENCE_IMAGE_SHORT_EDGE = 2048

#: Both axes land on this grid, which is the video VAE's spatial compression (16) times the
#: transformer's spatial patch (2). The same 32 every entry of `_PIXELS` is a multiple of.
CANVAS_MULTIPLE = 32

#: The accepted aspect band for a reference image, refused rather than letter-boxed.
MAX_ASPECT_RATIO = 4

#: Qwen3-VL's patch geometry. `patch` and `temporal_patch` set the width of one patch row
#: (3 * 2 * 16 * 16 = 1536) and `merge` sets how many patches fold into one token.
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
SPATIAL_MERGE_SIZE = 2

#: The processor's pixel-AREA budget, in patches-worth of pixels, not edge lengths — the
#: released `preprocessor_config.json` spells them `size.shortest_edge`/`size.longest_edge`
#: and they are `min_pixels`/`max_pixels`. A 2048-short-edge image at the 4:1 aspect bound
#: is 16,777,216 pixels exactly, which is why the cap is where it is.
MIN_PIXELS = 65_536
MAX_PIXELS = 16_777_216

#: Qwen3-VL's own image normalization. The tower was trained against it; a different
#: mean/std is a different model.
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)

#: The vision PAD ids, which are LOAD-BEARING rather than filler. The local
#: `text_encoder.Qwen3VLForConditionalGeneration` overwrites these positions by index and
#: would not care, but `transformers`' class of that same name (#579: a port answers to the
#: name of what it ports, and the module path is the disambiguator) FINDS the positions to
#: fill by matching these ids, so a presentation built with the wrong pad splices nothing and
#: reports nothing. Both dialects are served from one presentation, so it carries the ids
#: upstream looks for.
IMAGE_PAD_TOKEN = 151655
VIDEO_PAD_TOKEN = 151656


@dataclass(frozen=True, slots=True)
class PatchedVision:
    """What the patchifier states about one presentation's worth of vision blocks.

    `pixel_values` and `grid_thw` are batched per MODALITY, in block order within that
    modality, because that is the shape both text encoders take. `token_counts` is per BLOCK
    in presentation order, which is what the expansion below needs.
    """

    image_pixel_values: Any = None
    image_grid_thw: Any = None
    video_pixel_values: Any = None
    video_grid_thw: Any = None
    token_counts: tuple[int, ...] = ()
    video_flags: tuple[bool, ...] = ()
    """Per block in presentation order, whether it came from the VIDEO batch. Recorded
    rather than recovered: an image block and a video block can carry the SAME token count,
    so inferring the modality from the count would read the wrong batch without failing."""


def normalize_reference_image(image: Any) -> Any:
    """A decoded RGB image onto the reference canvas: short edge `REFERENCE_IMAGE_SHORT_EDGE`,
    both axes rounded onto `CANVAS_MULTIPLE`, aspect ratio refused outside 1:4 to 4:1.

    LANCZOS, and through PIL rather than through an array interpolation: the released model
    was conditioned on PIL's LANCZOS and `F.interpolate` is a different filter, so an image
    that reaches this as an array has already lost the thing being preserved. Callers hand
    it a `PIL.Image`.
    """
    from PIL import Image

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"a reference image must have a positive size, got {image.size}")
    if width > MAX_ASPECT_RATIO * height or height > MAX_ASPECT_RATIO * width:
        raise ValueError(
            f"a reference image must be within 1:{MAX_ASPECT_RATIO} and {MAX_ASPECT_RATIO}:1, "
            f"got {width}x{height}"
        )
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    target_h = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_w = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if (target_w, target_h) == image.size:
        return image
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


#: The video VAE's spatial compression. A reference image's latent extents are its pixel
#: extents over this, which is why the reference canvas is rounded onto `CANVAS_MULTIPLE`
#: (16 for the VAE, times the DiT's 2-wide spatial patch) rather than onto the VAE's ratio
#: alone: the packed rows are 2x2 PATCHES of latents, so an odd latent extent has no rows.
VAE_SPATIAL_RATIO = 16


def reference_block_geometry(image: Any) -> tuple[int, int, int]:
    """One normalized reference image -> the `(latent_t, latent_h, latent_w)` its VAE
    encoding will have.

    Derived rather than measured, because the packed LAYOUT has to be built before any
    component is leased and the encode happens inside one. The two are checked against each
    other at conditioning time — a disagreement is a defect, not a tolerance.
    """
    width, height = image.size
    return 1, height // VAE_SPATIAL_RATIO, width // VAE_SPATIAL_RATIO


def condition_pixels(image: Any) -> Any:
    """A normalized reference image -> `[1, 3, 1, H, W]` pixels in [-1, 1].

    [-1, 1] is what `video_vae.encode_condition` states it takes; it ImageNet-normalizes
    from there. Upstream hands its VAE raw `uint8` and divides by 255 inside, which is the
    same arithmetic written on the other side of the boundary.
    """
    import numpy as np
    import torch

    array = np.asarray(image, dtype=np.float32)
    pixels = torch.from_numpy(array).permute(2, 0, 1)[None, :, None]
    return pixels / 127.5 - 1.0


def _image_processor() -> Any:
    """Qwen3-VL's own image processor, configured from the constants above.

    Built explicitly rather than `from_pretrained`: given a string that is not a directory
    that spelling resolves against the Hub, which `fence.py::no-identifiers-in-code` refuses
    for exactly the reason it should — an endpoint must not be able to fetch by accident.
    The constants ARE the released `preprocessor_config.json`, stated as source.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    return Qwen2VLImageProcessor(
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        merge_size=SPATIAL_MERGE_SIZE,
        image_mean=list(IMAGE_MEAN),
        image_std=list(IMAGE_STD),
    )


def merged_token_count(grid: Any) -> int:
    """How many TEXT-ENCODER tokens one block's grid becomes: its patches, folded 2x2."""
    return int(grid.prod()) // SPATIAL_MERGE_SIZE**2


def patchify(presentation: Presentation) -> PatchedVision:
    """Every vision block of a presentation, patchified in one batched call per modality.

    A still block is presented to a temporal patch of 2 by repeating its frame, which is
    what `_VisionPatchEmbed` documents its Conv3d expects; the processor does that repeat
    itself for a single image, so a still is handed over as one image and a 2-frame video
    block as a 2-frame video.
    """
    import numpy as np
    import torch

    blocks = [row for row in presentation.rows if isinstance(row, VisionBlock)]
    if not blocks:
        return PatchedVision()

    processor = _image_processor()
    stills = [b.pixels for b in blocks if not b.video_block]
    motion = [b.pixels for b in blocks if b.video_block]

    image_pv = image_grid = video_pv = video_grid = None
    if stills:
        got = processor(images=stills, return_tensors="pt")
        image_pv, image_grid = got["pixel_values"], got["image_grid_thw"]
    if motion:
        got = processor(
            images=None,
            videos=[np.asarray(frames) for frames in motion],
            return_tensors="pt",
        )
        video_pv, video_grid = got["pixel_values_videos"], got["video_grid_thw"]

    # Per-block counts in PRESENTATION order, taken from each modality's batch in turn —
    # the two agree because the split above preserves relative order within a modality.
    still_grids = iter(image_grid) if image_grid is not None else iter(())
    motion_grids = iter(video_grid) if video_grid is not None else iter(())
    counts: list[int] = []
    for block in blocks:
        grid = next(motion_grids) if block.video_block else next(still_grids)
        counts.append(merged_token_count(grid))

    _ = torch  # the tensors above are torch's; the import states the dependency
    return PatchedVision(
        image_pv,
        image_grid,
        video_pv,
        video_grid,
        tuple(counts),
        tuple(b.video_block for b in blocks),
    )


@dataclass(frozen=True, slots=True)
class ExpandedPresentation:
    """A presentation with its vision blocks expanded into real pad runs.

    `token_ids` is what the text encoder is called on, `tags` is one modality tag per row of
    it, and `splices` gives each block's first-token position — the position AFTER its
    `<|vision_start|>`, which is what `text_encoder.VisionBlock.index` means.
    """

    token_ids: tuple[int, ...]
    tags: tuple[int, ...]
    splices: tuple[int, ...]


def expand(presentation: Presentation, token_counts: tuple[int, ...]) -> ExpandedPresentation:
    """Presentation rows -> flat token ids, tags and splice indices.

    A block row becomes `token_counts[i]` pad ids of its own modality. The count comes from
    the patchifier rather than being guessed here, for the same reason `expand_tags` takes
    it: only a resolved grid knows it, and a guess would misalign the whole text span.
    """
    from .presentation import TAG_VIDEO

    ids: list[int] = []
    tags: list[int] = []
    splices: list[int] = []
    block = 0
    for row, tag in zip(presentation.rows, presentation.tags, strict=True):
        if isinstance(row, VisionBlock):
            if block >= len(token_counts):
                raise ValueError(
                    f"the presentation has more than {len(token_counts)} vision blocks and "
                    f"only {len(token_counts)} token counts were supplied — the expansion "
                    "would silently misalign the text span"
                )
            pad = VIDEO_PAD_TOKEN if row.video_block else IMAGE_PAD_TOKEN
            splices.append(len(ids))
            ids.extend([pad] * token_counts[block])
            tags.extend([TAG_VIDEO] * token_counts[block])
            block += 1
        else:
            ids.append(row)
            tags.append(tag)
    if block != len(token_counts):
        raise ValueError(
            f"the presentation has {block} vision blocks and {len(token_counts)} token "
            "counts were supplied — the expansion would silently misalign the text span"
        )
    return ExpandedPresentation(tuple(ids), tuple(tags), tuple(splices))


def text_encoder_blocks(patched: PatchedVision, expanded: ExpandedPresentation) -> list[Any]:
    """The `text_encoder.VisionBlock(patches, grid_thw, index)` triples, in splice order.

    The patch rows of a modality's batch are one flat `[total_patches, 1536]` matrix, so
    each block's slice is found by walking the grids and consuming `t*h*w` rows per block —
    which is the same accounting `merged_token_count` divides down.
    """
    from .text_encoder import VisionBlock as EncoderVisionBlock

    out: list[Any] = []
    cursors = {"image": 0, "video": 0}
    still_grids = list(patched.image_grid_thw) if patched.image_grid_thw is not None else []
    motion_grids = list(patched.video_grid_thw) if patched.video_grid_thw is not None else []
    still_i = motion_i = 0

    for is_video, index in zip(patched.video_flags, expanded.splices, strict=True):
        if is_video:
            grid, values, key = motion_grids[motion_i], patched.video_pixel_values, "video"
            motion_i += 1
        else:
            grid, values, key = still_grids[still_i], patched.image_pixel_values, "image"
            still_i += 1
        rows = int(grid.prod())
        start = cursors[key]
        cursors[key] = start + rows
        out.append(
            EncoderVisionBlock(
                patches=values[start : start + rows],
                grid_thw=grid.reshape(1, 3),
                index=index,
            )
        )
    return out
