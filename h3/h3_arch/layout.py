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
  * `H3Solver`. The sampler's SIGN and SCHEDULE, in one small object that owns nothing
    else. The model boundary returns DATA-WARD velocity and this is what turns that into
    a latent update; se-002's first render had the sign backwards for every one of its 30
    evaluations (#522a), which is not a numeric drift but a stated convention nobody
    stated. It is stated here and again at `dit.MiniMaxH3Transformer3DModel.forward`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import msgspec

#: The model's own temporal token grid. The first latent frame spans one unit and every
#: later one spans four, rescaled by 5/3 — the released checkpoint's rotary convention.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0

#: THE TEMPORAL GEOMETRY, and the single relation every downstream count reads.
#:
#: H3's video VAE is causal on a 17-frame clip grid that produces 5 latent frames per clip
#: over a 5-frame / 2-latent head: `17k + 5` pixel frames <-> `5k + 2` latent frames. It is
#: NOT a ratio. A generic `(frames - 1) // 4 + 1` agrees with it nowhere useful and was the
#: geometry half of the first garbage render (#522b): 124 frames became 31 latents instead
#: of 37, and the wrong count reached the noise SHAPE, the packed row count and the ROTARY
#: CLOCK (`_video_t_spans` is indexed by latent frame) before the VAE deterministically
#: decoded the wrong 31 into the 103 frames the artifact actually carries.
FRAMES_PER_CLIP = 17
LATENTS_PER_CLIP = 5
HEAD_FRAMES = 5
HEAD_LATENTS = 2

#: The checkpoint's native clock and its audio VAE's latent rate. The audio row count
#: follows the video CLOCK rather than the video latent count.
FPS = 24
AUDIO_LATENT_FPS = 40

#: The packed order, and the modality tag each kind carries into AdaLN. Tags are the
#: checkpoint's: 0 video, 1 text, 2 audio.
SegmentKind = Literal["text", "cond", "audio", "video"]
SEGMENT_TAG: dict[str, int] = {
    "text": 1,
    "video": 0,
    "audio": 2,
    "cond": 0,
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


#: How far a muxed soundtrack may sit from the video's own duration before the container
#: is two clips in a trenchcoat. One video FRAME is the honest tolerance for a mux; the
#: first se-002 artifact was 4.29 s of video against 5.18 s of audio — 21 frames apart.
AV_DURATION_TOLERANCE_FRAMES = 1.0


@dataclass(frozen=True, slots=True)
class MediaFacts:
    """What a request RESOLVED TO, in the vocabulary a finished file can be read in.

    It exists so "observed" has something to disagree with. Every fact the first garbage
    render carried was read off the artifact itself, so nothing could contradict it."""

    width: int
    height: int
    frames: int
    fps: int
    sample_rate: int
    mute: bool = False

    @property
    def duration_s(self) -> float:
        return self.frames / float(self.fps)

    @property
    def av_tolerance_s(self) -> float:
        return AV_DURATION_TOLERANCE_FRAMES / float(self.fps)

    def av_drift(self, audio_seconds: float) -> float:
        return abs(audio_seconds - self.duration_s)

    def av_agrees(self, audio_seconds: float) -> bool:
        """The two streams were denoised JOINTLY, so a duration disagreement is a latent
        geometry that is wrong — never a mux preference."""
        return self.av_drift(audio_seconds) <= self.av_tolerance_s


def latent_frames(frames: int) -> int:
    """`17k + 5` pixel frames -> `5k + 2` latent frames. A count OFF the grid refuses.

    Refusing rather than flooring is deliberate: flooring is what a ratio does, and a
    ratio is the thing that was wrong. Every offered duration lands on the grid by
    construction, so an off-grid count reaching here is a caller bug and says so."""
    if frames <= HEAD_FRAMES:
        return HEAD_LATENTS
    clips, remainder = divmod(frames - HEAD_FRAMES, FRAMES_PER_CLIP)
    if remainder:
        raise ValueError(
            f"{frames} frames is not on H3's clip grid: the video VAE is causal over "
            f"{FRAMES_PER_CLIP}-frame clips on a {HEAD_FRAMES}-frame head, so a frame "
            f"count is {FRAMES_PER_CLIP}k + {HEAD_FRAMES} and nothing else"
        )
    return clips * LATENTS_PER_CLIP + HEAD_LATENTS


def pixel_frames(latent_t: int) -> int:
    """The exact INVERSE of `latent_frames`, and the reason the round trip is provable
    with no card: the frame count a latent sequence of `latent_t` is entitled to."""
    if latent_t <= HEAD_LATENTS:
        return HEAD_FRAMES
    clips, remainder = divmod(latent_t - HEAD_LATENTS, LATENTS_PER_CLIP)
    if remainder:
        raise ValueError(
            f"{latent_t} latent frames is not on H3's clip grid: a latent count is "
            f"{LATENTS_PER_CLIP}k + {HEAD_LATENTS} and nothing else"
        )
    return clips * FRAMES_PER_CLIP + HEAD_FRAMES


def audio_latents(frames: int) -> int:
    """The audio latent count for a video frame count, on the video CLOCK: the audio
    VAE's 40 Hz latent rate against the checkpoint's 24 fps grid."""
    return round(frames / float(FPS) * float(AUDIO_LATENT_FPS))


def latent_grid(frames: int, width: int, height: int, *, spatial_ratio: int = 16) -> LatentGrid:
    """Pixel geometry -> latent geometry, through H3's own relations and no ratio."""
    return LatentGrid(
        frames,
        height,
        width,
        latent_frames(frames),
        height // spatial_ratio,
        width // spatial_ratio,
        audio_latents(frames),
    )


def split_windows(
    extent_px: int, window_px: int, overlap_min_px: int, ratio: int
) -> tuple[list[int], list[int], list[int]]:
    """The video VAE's SPATIAL DECODE WINDOWS, in pixels: (start, length, overlap-with-next).

    Geometry and not a memory budget, which is why it lives here. The ViT decoder's RoPE
    coordinates are normalized by the extent of the tensor it is called on, so the window is
    a term of the function: one 48x84 call and twenty-eight 16x16 calls are different
    functions of the same latents, and the first is what put a 16 px lattice in every one of
    se-002's oracle renders (#557).

    Whole `window_px` windows, at least `overlap_min_px` apart, with the slack handed back one
    `ratio`-sized latent cell at a time so every boundary lands on a cell boundary. An extent
    that already fits inside one window IS one window.

    ComfyUI `comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE.split_tiles`, whose defaults
    (`tile_size=256`, `tile_overlap_min=64`, `tiling=True`) are the reference's own — and it
    says so itself, at `decode_tiled`: "tiling is always on internally with the reference's
    SEMANTIC TILE SIZES, ignore tiling fallbacks". `tiling` is a constructor default and
    `_adaptive_decode` branches on it, never on free memory, so the reference decodes 256 px
    windows on a 24 GiB card and on a 141 GiB one alike.
    """
    if window_px >= extent_px:
        return [0], [extent_px], []

    count = math.ceil(extent_px / window_px)
    while True:
        overlaps = [overlap_min_px] * (count - 1)
        slack = window_px * count - sum(overlaps) - extent_px
        if slack < 0:
            count += 1
        else:
            break
    for i in range(slack // ratio):
        overlaps[i % (count - 1)] += ratio

    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + window_px - overlaps[i])
    return starts, [window_px] * count, overlaps


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
class Keyframe:
    """A TARGET-CLOCK anchor, never a reference: its latent rows sit at the generated
    clip's first or last temporal coordinate and stay fixed while the target denoises."""

    resolved_frame_index: int


class PackedLayout:
    """The static packed structure for one (geometry, conditioning) signature.

    Row order is `text | keyframe conds | target audio | target video`, and the last two
    are always the last two — the final layer slices them by position rather than search."""

    def __init__(
        self,
        text_len: int,
        grid: LatentGrid,
        *,
        keyframes: tuple[Keyframe, ...] = (),
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
    def grid_points(self) -> int:
        """How many sigma VALUES the schedule holds. Never a loop bound."""
        return len(self.video_sigmas)

    @property
    def evaluations(self) -> int:
        """How many times the MODEL IS EVALUATED: the number of TRANSITIONS between grid
        points, which is one fewer than the number of points.

        The two numbers are named separately because conflating them is what put an extra
        forward pass in the first render (#522c): `flow_sigmas(30)` returns 31 values, the
        loop iterated over the VALUES, and the 31st iteration stepped from sigma 0 to
        sigma 0 — a full-cost evaluation with a zero delta. A 30-step request is 30
        evaluations over 31 grid points, exactly, and `steps` as a name is gone because it
        was never clear which of the two it meant."""
        return len(self.video_sigmas) - 1


def flow_sigmas(evaluations: int, shift: float) -> tuple[float, ...]:
    """The shifted flow-match schedule: a uniform grid on [1, 0] bent by `shift`.

    Returns `evaluations + 1` GRID POINTS, from exactly 1.0 down to exactly 0.0 — the
    shift maps both ends to themselves, so the terminal zero is part of the grid rather
    than appended to it. The argument is the count of MODEL EVALUATIONS because that is
    what a caller asks for and what it is billed for; the point count is one more, and the
    two are never the same number (#522c)."""
    return tuple(
        shift * (1.0 - i / evaluations) / (1.0 + (shift - 1.0) * (1.0 - i / evaluations))
        for i in range(evaluations + 1)
    )


def shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Unbend one shifted schedule back to its uniform grid, then bend it by another.

    This is why H3 samples ONE clock: `shift_sigma(flow_sigmas(n, 12)[i], 12, 3)` is
    exactly `flow_sigmas(n, 3)[i]`, because the first half inverts the shift back to
    `1 - i/n`. Diffusers runs a SECOND `MiniMaxH3Scheduler` at shift 3 over the same
    uniform base to get the audio schedule; that is the same numbers by a different
    route, and the conformance arm checks the identity rather than assuming it."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def build_timestep_plan(
    *,
    task: str,
    structure: str,
    evaluations: int,
    layout: PackedLayout,
    sigma_shift_video: float,
    sigma_shift_audio: float,
    visual_cond_timestep: float | None,
    scheduler: str = "flow_match_uniform",
    output_dtype: str = "float32",
    adapters: tuple[str, ...] = (),
) -> TimestepPlan:
    """`evaluations` is the number of MODEL EVALUATIONS the request asked for — the
    published step preset. The plan holds `evaluations + 1` sigmas per stream."""
    video = flow_sigmas(evaluations, sigma_shift_video)
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
        modality_tags=tuple(SEGMENT_TAG[k] for k in kinds),
        scheduler=scheduler,
        output_dtype=output_dtype,
        adapters=adapters,
    )


@dataclass(frozen=True, slots=True)
class Modulation:
    """ONE evaluation's complete modulation plane, resolved before a component is leased.

    Segments, the distinct timesteps they index and the final layer's two stream slices
    are ONE object because they are one arithmetic and their row indices must agree. They
    used to be two functions that each rebuilt `{value: row}` from a list of floats, and
    the second one was handed the list AFTER a float32 tensor round trip — so the exact
    dict lookup missed on a value the first one had put there, and the pod patched it with
    a nearest-value search. Computing the rows ONCE removes the class of bug rather than
    the instance."""

    segments: tuple[tuple[int, int, int], ...]
    timesteps: tuple[float, ...]
    """The distinct timestep values, in modulation-row order. `AdalnProj` gets exactly
    these, and a row is `timestep_index * 3 + modality_tag`."""
    video_stream: tuple[int, int, int]
    audio_stream: tuple[int, int, int]
    position_ids: tuple[tuple[float, float, float], ...]


def build_modulation(
    layout: PackedLayout,
    *,
    t_video: float,
    t_audio: float,
    visual_cond_t: float,
    text_token_tags: tuple[int, ...] | None = None,
) -> Modulation:
    """The per-segment modulation row table, the distinct timesteps it indexes, and the
    two target-stream slices the final layer reads.

    Rows are `t_row * 3 + modality_tag`, which is why one AdaLN projection over M distinct
    timesteps yields 3M modulation rows and a 100k-row sequence needs no per-row anything.
    """
    seg_t = {
        "text": t_video,
        "video": t_video,
        "audio": t_audio,
        "cond": max(t_video, visual_cond_t),
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
    va, vb = layout.stream("video")
    aa, ab = layout.stream("audio")
    return Modulation(
        segments=tuple(out),
        timesteps=tuple(unique),
        video_stream=(va, vb, row_of[seg_t["video"]]),
        audio_stream=(aa, ab, row_of[seg_t["audio"]]),
        position_ids=tuple(layout.position_ids),
    )


# ------------------------------------------------------------------ the solver


@dataclass(frozen=True, slots=True)
class SolverStep:
    """One model evaluation's schedule facts, and the update that consumes its result."""

    index: int
    sigma_video: float
    sigma_video_next: float
    sigma_audio: float
    sigma_audio_next: float

    @property
    def t_video(self) -> float:
        """The modulation plane's timestep, which is `1 - sigma` and not the sigma."""
        return 1.0 - self.sigma_video

    @property
    def t_audio(self) -> float:
        return 1.0 - self.sigma_audio

    def advance_video(self, latent: Any, velocity: Any) -> Any:
        return latent + (self.sigma_video - self.sigma_video_next) * velocity

    def advance_audio(self, latent: Any, velocity: Any) -> Any:
        return latent + (self.sigma_audio - self.sigma_audio_next) * velocity


@dataclass(frozen=True, slots=True)
class H3Solver:
    """The flow-match Euler solver. It owns SIGN and SCHEDULE and nothing else.

    THE SIGN CONVENTION, stated here and again at
    `dit.MiniMaxH3Transformer3DModel.forward`, because a convention held in one place is a
    convention half the code disagrees with:

        the model returns DATA-WARD velocity — the direction from noise toward data —
        and a step toward data multiplies it by the sigma DECREASE:

            x_next = x + (sigma - sigma_next) * v

    `sigma` decreases along the schedule, so `(sigma - sigma_next)` is POSITIVE and the
    latent moves toward data. se-002's first render used `(sigma_next - sigma)` on the
    same raw heads and therefore anti-denoised for all 30 evaluations (#522a).

    The two reference implementations agree with this and with each other. ComfyUI negates
    both heads inside the model and then applies the opposite delta, so the two sign flips
    compose to the same update. Diffusers' own `MiniMaxH3Scheduler` keeps the heads raw
    and writes `x0 = x_t + sigma * v` with `x_next = ratio*x + (1-ratio)*x0`, `ratio =
    sigma_next/sigma`, which is this line rearranged. This port takes diffusers' shape
    because it is the one that states the convention rather than compensating for it.
    """

    plan: TimestepPlan

    @property
    def evaluations(self) -> int:
        return self.plan.evaluations

    def step(self, index: int) -> SolverStep:
        return SolverStep(
            index=index,
            sigma_video=self.plan.video_sigmas[index],
            sigma_video_next=self.plan.video_sigmas[index + 1],
            sigma_audio=self.plan.audio_sigmas[index],
            sigma_audio_next=self.plan.audio_sigmas[index + 1],
        )

    def steps(self) -> Iterator[SolverStep]:
        """Every model evaluation, once. Never one per GRID POINT — the terminal sigma is
        a destination, not a transition, and evaluating at it costs a full forward pass to
        add exactly zero (#522c)."""
        for index in range(self.evaluations):
            yield self.step(index)
