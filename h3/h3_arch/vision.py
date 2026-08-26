"""The VISION SEAM — keyframe pixels to the text encoder's patch/grid/index triple.

`presentation.build` emits a `VisionBlock` row standing in for a whole block of vision
tokens, because how many tokens a block becomes is a fact of its RESOLVED GRID and only a
patchifier can state it. This file is that patchifier, plus the two things that bracket it:
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

Keyframes are already resolved onto the target canvas by the endpoint before they arrive.
This module owns Qwen's patch geometry and the VAE conditioning tensor conversion, not a
second media-sizing policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .presentation import Presentation, VisionBlock

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

#: The vision PAD id is LOAD-BEARING rather than filler: it names the positions the
#: keyframe tower overwrites.
IMAGE_PAD_TOKEN = 151655


@dataclass(frozen=True, slots=True)
class PatchedVision:
    """What the patchifier states about one presentation's worth of vision blocks.

    `image_pixel_values` and `image_grid_thw` are batched in keyframe order. `token_counts`
    is per block in presentation order, which is what the expansion below needs.
    """

    image_pixel_values: Any = None
    image_grid_thw: Any = None
    token_counts: tuple[int, ...] = ()


#: The video VAE's spatial compression. The endpoint canvas is already a multiple of 32:
#: 16 for the VAE, times the DiT's two-wide spatial patch.
VAE_SPATIAL_RATIO = 16


def conditioning_geometry(image: Any) -> tuple[int, int, int]:
    """One target-canvas keyframe -> the `(latent_t, latent_h, latent_w)` its VAE
    encoding will have.

    Derived rather than measured, because the packed LAYOUT has to be built before any
    component is leased and the encode happens inside one. The two are checked against each
    other at conditioning time — a disagreement is a defect, not a tolerance.
    """
    width, height = image.size
    return 1, height // VAE_SPATIAL_RATIO, width // VAE_SPATIAL_RATIO


def condition_pixels(image: Any) -> Any:
    """A target-canvas keyframe -> `[1, 3, 1, H, W]` pixels in [-1, 1].

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
    """Every keyframe block of a presentation, patchified in one batched image call.

    The processor repeats each still for Qwen's two-frame temporal patch internally.
    """
    blocks = [row for row in presentation.rows if isinstance(row, VisionBlock)]
    if not blocks:
        return PatchedVision()

    processor = _image_processor()
    got = processor(images=[b.pixels for b in blocks], return_tensors="pt")
    image_values, image_grid = got["pixel_values"], got["image_grid_thw"]
    return PatchedVision(
        image_pixel_values=image_values,
        image_grid_thw=image_grid,
        token_counts=tuple(merged_token_count(grid) for grid in image_grid),
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
            splices.append(len(ids))
            ids.extend([IMAGE_PAD_TOKEN] * token_counts[block])
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
    cursor = 0
    grids = list(patched.image_grid_thw) if patched.image_grid_thw is not None else []
    for grid, index in zip(grids, expanded.splices, strict=True):
        rows = int(grid.prod())
        out.append(
            EncoderVisionBlock(
                patches=patched.image_pixel_values[cursor : cursor + rows],
                grid_thw=grid.reshape(1, 3),
                index=index,
            )
        )
        cursor += rows
    return out
