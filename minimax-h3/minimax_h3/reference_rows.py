"""How much of a video reference the DiT attends to.

A reference clip enters H3 as condition rows beside the generated ones: its VAE latents, one
row per 2x2 patch of every latent frame, and its Qwen vision block in the text rows. At the
official sizing (24 fps, the target's canvas) a 12 s clip next to a 15 s render is ~95k extra
rows and ~3.7x the attention of the same render without it. `ReferencePlan` thins those rows:

- `canvas_divisor` puts the clip's frames on 1/n of the canvas edge after the official
  normalization, so both its latent rows and its vision tokens shrink n^2-fold. The layout
  places a smaller grid on the same aspect-normalized rotary extent, as it does any image
  reference at its own size.
- `frame_stride` keeps every n-th latent frame of each clip, at its official rotary position:
  the kept rows are exactly the official ones and the target sits where it always does.
- `guided_steps` keeps the clip's latent rows for the first k evaluations only.

Image references, standalone audio and a continuation's carried rows are never touched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage

from .continuation import ContextReferenceLayout

#: The canvas grid: the VAE's 16-pixel cells in 2x2 patches, and Qwen's 16-pixel patches merged 2x2.
_GRID = 32
_LAYOUT_FIELDS = {"position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"}


@dataclass(frozen=True, slots=True)
class ReferencePlan:
    canvas_divisor: int = 1
    frame_stride: int = 1
    guided_steps: int | None = None

    def __post_init__(self) -> None:
        if self.canvas_divisor < 1 or self.frame_stride < 1:
            raise ValueError("canvas_divisor and frame_stride must be positive")
        if self.guided_steps is not None and self.guided_steps < 0:
            raise ValueError("guided_steps must be nonnegative")

    @classmethod
    def parse(cls, text: str) -> ReferencePlan:
        """`canvas=2,stride=2,steps=3`; empty is the official sizing."""
        names = {"canvas": "canvas_divisor", "stride": "frame_stride", "steps": "guided_steps"}
        values: dict[str, int] = {}
        for item in filter(None, (part.strip() for part in text.split(","))):
            key, _, value = item.partition("=")
            if key not in names:
                raise ValueError(f"unknown reference plan key {key!r}")
            values[names[key]] = int(value)
        return cls(**values)

    def scale_videos(self, references: Sequence[Any]) -> list[Any]:
        """The normalized references with each video's frames on 1/`canvas_divisor` of its canvas."""
        if self.canvas_divisor == 1:
            return list(references)
        return [
            replace(ref, frames=_shrink(ref.frames, self.canvas_divisor)) if _public_video(ref) else ref
            for ref in references
        ]

    def thin(self, rows: ReferenceRows, block_state: Any) -> None:
        """Before the first evaluation: keep every `frame_stride`-th latent frame of each clip."""
        if self.frame_stride > 1:
            rows.drop(block_state, lambda frame: frame % self.frame_stride != 0)

    def release(self, rows: ReferenceRows, block_state: Any, step: int) -> None:
        """Before evaluation `step`: past `guided_steps`, the clips' latent rows leave the sequence."""
        if self.guided_steps is not None and step == self.guided_steps:
            rows.drop(block_state, lambda frame: torch.ones_like(frame, dtype=torch.bool))


class ReferenceRows:
    """Each packed video row's latent frame within its public video reference (-1 for every
    other row), followed through each drop."""

    def __init__(
        self, references: Sequence[Any], condition_latents: Sequence[Any], patch: Sequence[int], block_state: Any
    ) -> None:
        _, patch_h, patch_w = patch
        self.frame = torch.full((block_state.video_indices.numel(),), -1, dtype=torch.int64)
        offset, latents = 0, iter(condition_latents)
        for reference in references:
            if reference.kind not in ("image", "video"):
                continue
            frames, height, width = (int(n) for n in next(latents).shape[2:5])
            per_frame = (height // patch_h) * (width // patch_w)
            if _public_video(reference):
                span = slice(offset, offset + frames * per_frame)
                self.frame[span] = torch.arange(frames).repeat_interleave(per_frame)
            offset += frames * per_frame
        if next(latents, None) is not None or offset != block_state.num_condition_video_rows:
            raise ValueError("video reference rows disagree with the packed layout")

    def drop(self, block_state: Any, dropped_frame: Any) -> None:
        dropped = (self.frame >= 0) & dropped_frame(self.frame)
        if not bool(dropped.any()):
            return
        video_indices = block_state.video_indices
        keep = torch.ones(block_state.token_tags.numel(), dtype=torch.bool, device=video_indices.device)
        keep[video_indices[dropped.to(video_indices.device)]] = False
        select_rows(block_state, keep)
        self.frame = self.frame[~dropped]


def _public_video(reference: Any) -> bool:
    return reference.kind == "video" and not isinstance(reference, ContextReferenceLayout)


def _shrink(frames: Any, divisor: int) -> np.ndarray:
    pixels = np.asarray(frames)
    height, width = pixels.shape[1:3]
    size = (
        max(_GRID, round(width / divisor / _GRID) * _GRID),
        max(_GRID, round(height / divisor / _GRID) * _GRID),
    )
    return np.stack(
        [np.asarray(PILImage.fromarray(f).resize(size, PILImage.Resampling.LANCZOS)) for f in pixels]
    )


def select_rows(block_state: Any, keep: torch.Tensor) -> None:
    """Keep the packed rows `keep` marks, renumbered in order, with every per-row field."""
    fields = dict(block_state.denoiser_input_fields)
    if set(fields) != _LAYOUT_FIELDS:
        raise ValueError(f"unexpected packed-layout fields {sorted(fields)}")
    position = torch.cumsum(keep, 0) - 1
    kept = {}
    for name in ("video_indices", "audio_indices", "text_indices"):
        rows = fields[name]
        kept[name] = keep[rows]
        fields[name] = position[rows[kept[name]]]
    fields["position_ids"] = fields["position_ids"][keep]
    fields["token_tags"] = fields["token_tags"][keep]
    video, audio, text = kept["video_indices"], kept["audio_indices"], kept["text_indices"]
    block_state.num_condition_video_rows = int(video[: block_state.num_condition_video_rows].sum())
    block_state.num_condition_audio_rows = int(audio[: block_state.num_condition_audio_rows].sum())
    block_state.latents = block_state.latents[video]
    block_state.audio_latents = block_state.audio_latents[audio]
    if not bool(text.all()):
        block_state.prompt_embeds = block_state.prompt_embeds[:, text]
    block_state.row_timestep_plan = [
        (timesteps, indices[keep]) for timesteps, indices in block_state.row_timestep_plan
    ]
    for name, value in fields.items():
        setattr(block_state, name, value)
    block_state.denoiser_input_fields = fields
