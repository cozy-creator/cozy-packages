"""H3's packed-sequence GEOMETRY and its exact timestep plan — pure CPU, zero weights.

This is the half of H3 that is arithmetic rather than parameters, and keeping it out of
`dit.py` is what lets `denoise` be one declared component scope over the transformer alone:
by the time a component lease is acquired, the sequence, the rotary coordinates, the
modulation segment table and the distinct timesteps are already resolved facts.

Two things are load-bearing beyond the packing itself:

  * THE SEGMENT TABLE. Every packed segment is uniform in (modality tag, timestep class),
    so a 100k-row sequence needs a handful of modulation rows and a per-segment slice — not
    a per-row timestep tensor. The one exception is the text span, whose vision-pad
    positions carry the VIDEO tag inside a text segment, so it splits into tag runs.
  * `TimestepPlan.digest()`. §1.1.1: exact baked coverage is matched by the COMPLETE plan
    digest and never by step count. `steps=30` is display metadata; two 30-step requests
    with different scheduler shifts, different condition classes or a different modality
    convention are different plans, and this digest is what says so before any byte moves.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import msgspec

#: The model's own temporal token grid. The first latent frame spans one unit and every
#: later one spans four, rescaled by 5/3 — the released checkpoint's rotary convention.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0

#: The packed order, and the modality tag each kind carries into AdaLN. Tags are the
#: checkpoint's: 0 video, 1 text, 2 audio.
SegmentKind = Literal["text", "cond", "ref_img", "ref_audio", "audio", "video"]
SEGMENT_TAG: dict[str, int] = {
    "text": 1,
    "video": 0,
    "audio": 2,
    "cond": 0,
    "ref_img": 0,
    "ref_audio": 2,
}


@dataclass(frozen=True, slots=True)
class LatentGrid:
    """One request's resolved latent geometry. `t`/`h`/`w` are LATENT extents, already
    divided by the VAE's 4x/16x ratios and by the DiT's 1x2x2 patch."""

    frames: int
    height: int
    width: int
    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_h // 2) * (self.latent_w // 2)

    @property
    def video_rows(self) -> int:
        return self.latent_t * self.rows_per_frame

    @property
    def audio_rows(self) -> int:
        return self.audio_t * 2


def latent_grid(
    frames: int, width: int, height: int, *, spatial_ratio: int = 16, temporal_ratio: int = 4
) -> LatentGrid:
    """Pixel geometry -> latent geometry. The audio row count follows the video clock: the
    audio VAE's 40 Hz latent rate against the 24 fps video grid."""
    latent_t = (frames - 1) // temporal_ratio + 1
    latent_h = height // spatial_ratio
    latent_w = width // spatial_ratio
    audio_t = round(frames / 24.0 * 40.0)
    return LatentGrid(frames, height, width, latent_t, latent_h, latent_w, audio_t)


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> list[float]:
    ratio = dim / sqrt_area
    n = dim // patch
    return [(i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0 for i in range(n)]


def _frame_grid(h: int, w: int) -> tuple[list[tuple[float, float]], list[float]]:
    """Area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows."""
    area = math.sqrt(h * w)
    hh = _axis_from_sqrt_area(h, 2, area)
    ww = _axis_from_sqrt_area(w, 2, area)
    return [(y, x) for y in hh for x in ww], ww


def _video_t_spans(n: int) -> list[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n: int, origin: float) -> list[float]:
    spans = _video_t_spans(n)
    out, acc = [], 0.0
    for i in range(n):
        out.append(origin + acc)
        acc += spans[i]
    return out


@dataclass(frozen=True, slots=True)
class RefBlock:
    """One ordered reference's packed contribution. ORDER IS SEMANTIC: it drives both the
    `<Picture i>`/`<Video i>`/`<Audio i>` labels and this shared rotary cursor."""

    kind: Literal["image", "audio", "video", "video_audio"]
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0
    ref_audio_t: int = 0


@dataclass(frozen=True, slots=True)
class Keyframe:
    """A TARGET-CLOCK anchor, never a reference: its latent rows sit at the generated
    clip's first or last temporal coordinate and stay fixed while the target denoises."""

    resolved_frame_index: int


class PackedLayout:
    """The static packed structure for one (geometry, conditioning) signature.

    Row order is `text | keyframe conds | per-reference blocks | target audio | target
    video`, and the last two are always the last two — the final layer slices them by
    position rather than by search."""

    def __init__(
        self,
        text_len: int,
        grid: LatentGrid,
        *,
        keyframes: tuple[Keyframe, ...] = (),
        refs: tuple[RefBlock, ...] = (),
    ) -> None:
        frame, w_grid = _frame_grid(grid.latent_h, grid.latent_w)
        frame_rows = len(frame)
        segments: list[tuple[str, int]] = [("text", text_len)]
        pos: list[tuple[float, float, float]] = [(float(i), 0.0, 0.0) for i in range(text_len)]
        img_update: list[bool] = []
        audio_update: list[bool] = []

        for kf in keyframes:
            if kf.resolved_frame_index == 0:
                cond_t = float(text_len)
            elif kf.resolved_frame_index == grid.frames - 1:
                cond_t = float(text_len) + sum(_video_t_spans(grid.latent_t)) - FRAME_RESCALE
            else:
                raise ValueError(
                    "only first/last keyframe anchors are supported by this release; an "
                    "interior anchor is a separate Cozy extension with its own evidence"
                )
            segments.append(("cond", frame_rows))
            pos.extend((cond_t, y, x) for y, x in frame)
            img_update.extend([False] * frame_rows)

        target_audio_w = (w_grid[0], w_grid[-1])
        cursor = float(text_len)
        for blk in refs:
            if blk.kind == "image":
                r_frame, _ = _frame_grid(blk.latent_h, blk.latent_w)
                segments.append(("ref_img", len(r_frame)))
                pos.extend((cursor, y, x) for y, x in r_frame)
                img_update.extend([False] * len(r_frame))
                cursor += 1.0
            elif blk.kind == "audio":
                if blk.ref_audio_t > 0:
                    segments.append(("ref_audio", blk.ref_audio_t * 2))
                    pos.extend(_audio_positions(cursor, blk.ref_audio_t, *target_audio_w))
                    audio_update.extend([False] * blk.ref_audio_t * 2)
                cursor += float(blk.ref_audio_t)
            else:
                # a soundtracked video's audio rows pack immediately BEFORE its video rows,
                # both sharing one cursor origin: they are the same moment, not a sequence
                r_frame, r_w = _frame_grid(blk.latent_h, blk.latent_w)
                if blk.ref_audio_t > 0:
                    segments.append(("ref_audio", blk.ref_audio_t * 2))
                    pos.extend(_audio_positions(cursor, blk.ref_audio_t, r_w[0], r_w[-1]))
                    audio_update.extend([False] * blk.ref_audio_t * 2)
                n = blk.latent_t * len(r_frame)
                segments.append(("ref_img", n))
                pos.extend(_video_positions(blk.latent_t, r_frame, cursor))
                img_update.extend([False] * n)
                cursor += max(float(blk.ref_audio_t), sum(_video_t_spans(blk.latent_t)))

        segments.append(("audio", grid.audio_rows))
        pos.extend(_audio_positions(cursor, grid.audio_t, *target_audio_w))
        audio_update.extend([True] * grid.audio_rows)

        segments.append(("video", grid.video_rows))
        pos.extend(_video_positions(grid.latent_t, frame, cursor))
        img_update.extend([True] * grid.video_rows)

        self.grid = grid
        self.text_len = text_len
        self.position_ids = pos
        self.img_update = tuple(img_update)
        self.audio_update = tuple(audio_update)
        self.signature = (text_len, grid.latent_t, grid.latent_h, grid.latent_w, grid.audio_t)
        offset = 0
        table: list[tuple[int, int, str]] = []
        for kind, n in segments:
            table.append((offset, offset + n, kind))
            offset += n
        self.segments = tuple(table)
        self.seq_len = offset

    def stream(self, kind: str) -> tuple[int, int]:
        for a, b, k in self.segments:
            if k == kind:
                return a, b
        raise KeyError(f"the packed sequence has no {kind!r} segment")


def _audio_positions(
    cursor: float, t: int, w_low: float, w_high: float
) -> list[tuple[float, float, float]]:
    """Channel-major stereo rows: t advances per latent frame, w pins to the grid extremes
    per stereo channel, h stays 0 — the two channels are two POSITIONS, not two batches."""
    return [(cursor + i, 0.0, w_low) for i in range(t)] + [
        (cursor + i, 0.0, w_high) for i in range(t)
    ]


def _video_positions(
    vt: int, frame: list[tuple[float, float]], cursor: float
) -> list[tuple[float, float, float]]:
    grid = _video_t_grid(vt, cursor)
    return [(grid[i], y, x) for i in range(vt) for y, x in frame]


class TimestepPlan(msgspec.Struct, frozen=True):
    """The EXACT plan one request presents to the modulation plane.

    Every field is part of baked-coverage identity (§1.1.1). Step count is NOT a field on
    its own: it is implied by the ordered value lists, which is the point — a plan with the
    same count and a different shift is a different plan and gets a different digest."""

    task: str
    structure: str
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    sigma_shift_video: float
    sigma_shift_audio: float
    visual_cond_timestep: float | None
    audio_cond_timestep: float | None
    modality_tags: tuple[int, ...]
    scheduler: str
    output_dtype: str
    adapters: tuple[str, ...] = ()

    def digest(self) -> str:
        return (
            "blake2b:"
            + hashlib.blake2b(msgspec.json.encode(self), digest_size=16).hexdigest()
        )

    @property
    def steps(self) -> int:
        """Display metadata. Never a cache key — that is what `digest()` is for."""
        return len(self.video_sigmas)


def flow_sigmas(steps: int, shift: float) -> tuple[float, ...]:
    """The shifted flow-match schedule: a uniform grid on [1, 0] bent by `shift`."""
    return tuple(
        shift * (1.0 - i / steps) / (1.0 + (shift - 1.0) * (1.0 - i / steps))
        for i in range(steps + 1)
    )


def shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def build_timestep_plan(
    *,
    task: str,
    structure: str,
    steps: int,
    layout: PackedLayout,
    sigma_shift_video: float,
    sigma_shift_audio: float,
    visual_cond_timestep: float | None,
    audio_cond_timestep: float | None,
    scheduler: str = "flow_match_uniform",
    output_dtype: str = "float32",
    adapters: tuple[str, ...] = (),
) -> TimestepPlan:
    video = flow_sigmas(steps, sigma_shift_video)
    audio = tuple(shift_sigma(s, sigma_shift_video, sigma_shift_audio) for s in video)
    kinds = sorted({k for _, _, k in layout.segments})
    return TimestepPlan(
        task=task,
        structure=structure,
        video_sigmas=video,
        audio_sigmas=audio,
        sigma_shift_video=sigma_shift_video,
        sigma_shift_audio=sigma_shift_audio,
        visual_cond_timestep=visual_cond_timestep,
        audio_cond_timestep=audio_cond_timestep,
        modality_tags=tuple(SEGMENT_TAG[k] for k in kinds),
        scheduler=scheduler,
        output_dtype=output_dtype,
        adapters=adapters,
    )


def modulation_segments(
    layout: PackedLayout,
    *,
    t_video: float,
    t_audio: float,
    visual_cond_t: float,
    audio_cond_t: float,
    text_token_tags: tuple[int, ...] | None = None,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """The per-segment modulation row table, plus the distinct timestep values it indexes.

    Rows are `t_row * 3 + modality_tag`, which is why one AdaLN projection over M distinct
    timesteps yields 3M modulation rows and a 100k-row sequence needs no per-row anything.
    """
    seg_t = {
        "text": t_video,
        "video": t_video,
        "audio": t_audio,
        "cond": max(t_video, visual_cond_t),
        "ref_img": max(t_video, visual_cond_t),
        "ref_audio": max(t_audio, audio_cond_t),
    }
    present = {k for _, _, k in layout.segments}
    unique = sorted({seg_t[k] for k in present})
    row_of = {value: index for index, value in enumerate(unique)}
    out: list[tuple[int, int, int]] = []
    for a, b, kind in layout.segments:
        base = row_of[seg_t[kind]] * 3
        if kind == "text" and text_token_tags is not None:
            start = 0
            for i in range(1, b - a + 1):
                if i == b - a or text_token_tags[i] != text_token_tags[start]:
                    out.append((a + start, a + i, base + int(text_token_tags[start])))
                    start = i
        else:
            out.append((a, b, base + SEGMENT_TAG[kind]))
    return out, unique


def stream_rows(
    layout: PackedLayout, unique: list[float], *, t_video: float, t_audio: float
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The final layer's two slices: (start, stop, modulation row) for video and audio."""
    row_of = {value: index for index, value in enumerate(unique)}
    va, vb = layout.stream("video")
    aa, ab = layout.stream("audio")
    return (va, vb, row_of[t_video]), (aa, ab, row_of[t_audio])
