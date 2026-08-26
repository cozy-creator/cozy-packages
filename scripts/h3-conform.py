#!/usr/bin/env python
"""H3's DETERMINISTIC CONFORMANCE ARMS — every reference semantic that can be decided with
no card, no weights and no model library, fired for real and observed.

    nice -n 19 python scripts/h3-conform.py [arm ...]
    arms: sign, geometry, windows, schedule, pixels, facts

THESE RUN IN CI (#533). The four defects that produced se-002's first garbage render —
reversed sampler sign, generic temporal ratio, sigma points counted as evaluations, and a
double pixel rescale — were every one of them a cheap deterministic CPU check that no
manually-invoked script was running. An arm that only fires when someone remembers it is
an arm that stops firing.

THE ARMS OUTLIVE THE IMPLEMENTATION (#531). The hand port is slated for replacement by a
diffusers rebase, so every arm here checks a CONTRACT against an INDEPENDENTLY WRITTEN
REFERENCE VALUE, never an internal call shape: the reference expression is spelled out in
the arm from the upstream source it came from, and the endpoint's answer has to match it.
The same file gates the rebase.

EVERY ARM CARRIES ITS OWN RED CONTROL — the exact defective expression that shipped, kept
as a permanent negative control and asserted to DISAGREE. A green arm whose red half was
never observed is a claim; se-008 learned that with a NaN floor that was green for three
days while checking a tensor in which NaN cannot exist.

DEPENDENCIES: the standard library, msgspec (which `layout` uses for the plan struct) and
numpy (which the pixel arm uses as a real array library). No torch, no weights, no network.
The pixel conversion is written in the array-API subset numpy and torch spell identically
precisely so this arm exercises the shipped function rather than a paraphrase of it.

REFERENCES, both read from source and cited where they are used:
  * diffusers `MiniMaxH3Scheduler` / `modular_pipelines/minimax_h3` (PR #14355, the
    OFFICIAL implementation and the semantic authority per #531).
  * ComfyUI `comfy/ldm/minimax/model.py` + `comfy_extras/nodes_minimax_h3.py`, which is a
    black-box output oracle and agrees with diffusers on all four of these.
"""

from __future__ import annotations

import math
import pathlib
import sys
import traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "h3"))

from h3_arch.layout import (  # noqa: E402
    AV_DURATION_TOLERANCE_FRAMES,
    FPS,
    H3Solver,
    MediaFacts,
    PackedLayout,
    audio_latents,
    build_timestep_plan,
    flow_sigmas,
    latent_frames,
    latent_grid,
    pixel_frames,
    shift_sigma,
    split_windows,
)
from h3_arch.pixels import PIXEL_FULL_SCALE, pixel_bytes  # noqa: E402

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0

#: The released checkpoint's two flow shifts. Named here rather than imported so the arm
#: does not inherit the value it is checking against.
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

#: The video VAE's decode-window constants and its spatial ratio. Named here rather than
#: imported, for the same reason the shifts are.
TILE_SIZE = 256
TILE_OVERLAP_MIN = 64
VAE_RATIO = 16

#: Floating-point agreement between two independently written closed forms of the same
#: expression. Both are float64; the difference is reassociation, not precision loss.
TOL = 1e-12


def observe(what: str, detail: str = "") -> None:
    print(f"{PASS}{what}" + (f"\n         {detail}" if detail else ""))


def failed(what: str, detail: str = "") -> None:
    global _failures
    _failures += 1
    print(f"{FAIL}{what}" + (f"\n         {detail}" if detail else ""))


def check(what: str, got: Any, want: Any, detail: str = "") -> None:
    if got == want:
        observe(what, detail or f"{got!r}")
    else:
        failed(what, f"got {got!r}, want {want!r}")


def close(what: str, got: float, want: float, *, tol: float = TOL, detail: str = "") -> None:
    if abs(got - want) <= tol:
        observe(what, detail or f"{got!r}")
    else:
        failed(what, f"got {got!r}, want {want!r} (|d| = {abs(got - want):g})")


def red(what: str, defect: Any, correct: Any, detail: str = "") -> None:
    """THE NEGATIVE CONTROL: the defective value must DISAGREE with the correct one."""
    if defect != correct:
        observe(f"RED CONTROL — {what}", detail or f"defect {defect!r} vs {correct!r}")
    else:
        failed(f"RED CONTROL — {what}", f"the defect agrees ({defect!r}) — the arm is blind")


def expect_refusal(what: str, fn: Any) -> None:
    try:
        fn()
    except Exception as exc:
        observe(what, f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return
    failed(what, "the call SUCCEEDED — the arm did not fire")


# --------------------------------------------------------------- the references
#
# Each of these is the upstream expression written out, so the arm has something to be
# right or wrong AGAINST. None of them calls into the endpoint.


def reference_sigmas(points: int, shift: float) -> list[float]:
    """diffusers `MiniMaxH3Scheduler.set_timesteps`:

        base = torch.linspace(1.0, 0.0, num_inference_steps)
        sigmas = shift * base / (1 + (shift - 1) * base)

    `points` values, from exactly 1.0 to exactly 0.0 — the shift maps both ends to
    themselves, so the terminal zero is part of the grid rather than appended to it."""
    base = [1.0 - i / (points - 1) for i in range(points)]
    return [shift * b / (1.0 + (shift - 1.0) * b) for b in base]


def reference_timesteps(sigmas: list[float]) -> list[float]:
    """diffusers: `self.timesteps = (1.0 - sigmas[:-1])`. The terminal sigma is a
    destination and gets no timestep, which is why N points drive N-1 evaluations."""
    return [1.0 - s for s in sigmas[:-1]]


def reference_step(x: float, velocity: float, sigma: float, sigma_next: float) -> float:
    """diffusers `MiniMaxH3Scheduler.step`, verbatim in behaviour:

        denoised = sample + sigma * model_output      # data-ward: `+`, not `-`
        ratio = sigma_next / sigma
        prev  = ratio * sample + (1 - ratio) * denoised

    ComfyUI reaches the identical trajectory by negating both heads in the model and then
    applying the opposite delta; two sign flips compose to this."""
    denoised = x + sigma * velocity
    ratio = sigma_next / sigma
    return ratio * x + (1.0 - ratio) * denoised


def reference_latent_t(frames: int) -> int:
    """ComfyUI `nodes_minimax_h3.video_latent_t`, which is diffusers'
    `video_latent_num_frames` with `clip_length=17` and `tokens_chunk_size=5`."""
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def defective_latent_t(frames: int) -> int:
    """THE DEFECT (#522b): a generic 4x temporal ratio. 124 -> 31 instead of 37."""
    return (frames - 1) // 4 + 1


def defective_step(x: float, velocity: float, sigma: float, sigma_next: float) -> float:
    """THE DEFECT (#522a): ComfyUI's delta applied to un-negated (data-ward) heads."""
    return x + (sigma_next - sigma) * velocity


def reference_windows(extent_px: int) -> tuple[list[int], list[int], list[int]]:
    """ComfyUI `comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE.split_tiles`, written out at its
    own defaults (`tile_size=256`, `tile_overlap_min=64`, `vae_ratio=16`, `tiling=True`)."""
    tile_size, tile_overlap_min, vae_ratio = 256, 64, 16
    if tile_size >= extent_px:
        return [0], [extent_px], []
    n = math.ceil(extent_px / tile_size)
    while True:
        overlaps = [tile_overlap_min] * (n - 1)
        remaining = tile_size * n - sum(overlaps) - extent_px
        if remaining < 0:
            n += 1
        else:
            break
    for i in range(remaining // vae_ratio):
        overlaps[i % (n - 1)] += vae_ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * n, overlaps


def defective_windows(extent_px: int) -> tuple[list[int], list[int], list[int]]:
    """THE DEFECT (#557): ONE window over the whole frame. This is what shipped — the port
    read the reference's spatial tiling as a card-fitting trick and deleted it, having just
    argued, for the TIME axis, that an extent-normalized RoPE makes the window semantic."""
    return [0], [extent_px], []


# --------------------------------------------------------------- the arms


def arm_sign() -> None:
    print("\n== the solver's SIGN — one tensor, against diffusers' own step ==")

    # A flow-match latent is `x = (1 - sigma) * data + sigma * noise`, so the DATA-WARD
    # velocity is `data - noise`. Real numbers, chosen so the exact answer is checkable by
    # eye: at sigma 1 the latent IS the noise, and one full step must land on the data.
    data, noise = 1.0, -1.0
    velocity = data - noise
    x = noise

    exact = reference_step(x, velocity, 1.0, 0.0)
    close("one full step from pure noise lands on the data", exact, data, detail=f"{exact!r}")

    step = H3Solver(_plan(30)).step(0)
    ours = step.advance_video(x, velocity)
    theirs = reference_step(x, velocity, step.sigma_video, step.sigma_video_next)
    close("H3Solver.advance_video == diffusers' step, evaluation 0", ours, theirs)

    broken = defective_step(x, velocity, step.sigma_video, step.sigma_video_next)
    red("the shipped `x + (sigma_next - sigma) * v`", broken, theirs)
    if abs(broken - data) > abs(x - data):
        observe(
            "and it moves AWAY from the data — this is anti-denoising, not a rounding bug",
            f"|x-data| {abs(x - data):g} -> |broken-data| {abs(broken - data):g}",
        )
    else:
        failed("the defect should move away from the data", f"{broken!r}")

    # The whole 30-evaluation schedule, on one number. A flow ODE integrated with a
    # CONSTANT velocity is exact for any partition of [1, 0], so the trajectory must land
    # on `data` to the last bit, and the defective sign must not.
    solver = H3Solver(_plan(30))
    good = bad = x
    for s in solver.steps():
        good = s.advance_video(good, velocity)
        bad = defective_step(bad, velocity, s.sigma_video, s.sigma_video_next)
    close("30 evaluations of the real schedule land on the data", good, data, tol=1e-9)
    red("30 evaluations of the shipped sign", round(bad, 6), round(good, 6))

    # The audio stream runs the same sign on its own shift, and must land in the same place.
    good_audio = x
    for s in solver.steps():
        good_audio = s.advance_audio(good_audio, velocity)
    close("the audio stream's 30 evaluations land there too", good_audio, data, tol=1e-9)


def arm_geometry() -> None:
    print("\n== TEMPORAL GEOMETRY — 17k+5 frames <-> 5k+2 latents, and back ==")

    # Two k values, per #523.2. 124 is the 5 s preset (k=7); 56 is k=3.
    for frames in (124, 56, 5, 362):
        want = reference_latent_t(frames)
        got = latent_frames(frames)
        check(f"{frames} frames -> {want} latent frames", got, want)
        check(f"  and {got} latent frames -> {frames} frames (round trip)",
              pixel_frames(got), frames)

    red("the shipped generic 4x ratio at 124 frames", defective_latent_t(124), latent_frames(124),
        detail=f"31 vs {latent_frames(124)} — the wrong count reached the noise shape, the "
               "packed rows AND the rotary clock")
    red("  and at 56 frames", defective_latent_t(56), latent_frames(56))

    expect_refusal(
        "a frame count OFF the clip grid refuses instead of flooring",
        lambda: latent_frames(100),
    )
    expect_refusal(
        "and so does a latent count off it",
        lambda: pixel_frames(36),
    )

    grid = latent_grid(124, 1344, 768)
    check("the 5 s 16:9 grid", (grid.latent_t, grid.latent_h, grid.latent_w), (37, 48, 84))
    check("its audio latents follow the video CLOCK", grid.audio_t, audio_latents(124),
          detail=f"{grid.audio_t} = round(124/24*40)")
    check("audio latents at 124 frames", audio_latents(124), 207)

    # THE ROTARY CLOCK is indexed by latent frame, so the wrong count was never confined to
    # the noise tensor — this is the number that actually reaches the attention.
    layout = PackedLayout(64, grid)
    wrong = PackedLayout(64, latent_grid(124, 1344, 768))
    check("the packed sequence for 124 frames at 1344x768", layout.seq_len,
          64 + grid.audio_rows + grid.video_rows,
          detail=f"{layout.seq_len} rows = 64 text + {grid.audio_rows} audio + "
                 f"{grid.video_rows} video")
    last_t = layout.position_ids[-1][0]
    red("the rotary extent a 31-latent clip would have produced",
        round(_rotary_extent(31), 6), round(_rotary_extent(37), 6),
        detail=f"the last video row sits at t={last_t:g}; a 31-latent clip puts it at "
               f"{64 + _rotary_extent(31) - _rotary_extent(1):g}")
    del wrong


def arm_windows() -> None:
    print("\n== the VAE's SPATIAL DECODE WINDOW — 16x16 latent cells, not the whole frame ==")

    # The 5 s 16:9 preset both oracle renders and every bank render used.
    for extent, axis in ((768, "height"), (1344, "width")):
        want = reference_windows(extent)
        got = split_windows(extent, TILE_SIZE, TILE_OVERLAP_MIN, VAE_RATIO)
        check(f"{axis} {extent} px -> the reference's window plan", got, want,
              detail=f"{len(got[0])} windows, starts {got[0]}, overlaps {got[2]}")

        starts, lens, overlaps = got
        check(f"  every {axis} boundary lands on a 16 px latent cell",
              [s % VAE_RATIO for s in starts] + [o % VAE_RATIO for o in overlaps],
              [0] * (len(starts) + len(overlaps)))
        check(f"  and the windows cover {extent} px exactly",
              starts[-1] + lens[-1], extent,
              detail="the last window ends on the frame edge, so nothing is invented or lost")
        check(f"  with every overlap at least {TILE_OVERLAP_MIN} px to crossfade over",
              min(overlaps) >= TILE_OVERLAP_MIN, True, detail=f"min {min(overlaps)} px")

    # THE POINT. The decoder's RoPE divides by the extent it is handed, so this number IS the
    # function: 16 cells at 256 px, whatever the frame is.
    for extent in (768, 1344, 256, 512, 2048):
        lens = split_windows(extent, TILE_SIZE, TILE_OVERLAP_MIN, VAE_RATIO)[1]
        check(f"a {extent} px axis is decoded {len(lens)}x16 latent cells at a time",
              {length // VAE_RATIO for length in lens}, {16})

    check("an axis that already fits is ONE window and not a padded one",
          split_windows(240, TILE_SIZE, TILE_OVERLAP_MIN, VAE_RATIO), ([0], [240], []))

    red("the one-window decode that shipped, at 1344 px",
        defective_windows(1344)[1][0] // VAE_RATIO,
        split_windows(1344, TILE_SIZE, TILE_OVERLAP_MIN, VAE_RATIO)[1][0] // VAE_RATIO,
        detail="84 latent cells vs 16 — every token 5.25x closer in normalized RoPE than the "
               "decoder is ever handed, which is the 16 px lattice the owner rejected (#557): "
               "token-boundary gradient excess 150% of baseline through the one-window decode "
               "against 10% through the reference's, on the SAME video VAE file")
    red("  and at 768 px", defective_windows(768)[1][0] // VAE_RATIO,
        split_windows(768, TILE_SIZE, TILE_OVERLAP_MIN, VAE_RATIO)[1][0] // VAE_RATIO)
    red("  and the seam it was deleted to avoid, which the reference blends away",
        len(defective_windows(1344)[2]), len(reference_windows(1344)[2]),
        detail="0 overlaps vs 6 — the reference crossfades every window boundary, so the "
               "artefact the deletion was justified by does not exist")


def arm_schedule() -> None:
    print("\n== the SIGMA GRID vs the MODEL EVALUATIONS — two numbers, named apart ==")

    evaluations = 30
    ours = list(flow_sigmas(evaluations, SHIFT_VIDEO))
    theirs = reference_sigmas(evaluations + 1, SHIFT_VIDEO)

    check("30 evaluations -> 31 sigma grid points", len(ours), 31)
    check("  and the reference agrees, at set_timesteps(31)", len(theirs), len(ours))
    worst = max(abs(a - b) for a, b in zip(ours, theirs, strict=True))
    close("the whole sigma VECTOR matches diffusers', value for value", worst, 0.0,
          detail=f"max |difference| over 31 points = {worst:g}")
    close("it starts at exactly 1.0", ours[0], 1.0)
    close("and ends at exactly 0.0", ours[-1], 0.0)

    plan = _plan(evaluations)
    check("TimestepPlan.grid_points", plan.grid_points, 31)
    check("TimestepPlan.evaluations", plan.evaluations, 30)
    check("H3Solver yields one step per EVALUATION", len(list(H3Solver(plan).steps())), 30)

    t_ours = [s.t_video for s in H3Solver(plan).steps()]
    t_theirs = reference_timesteps(theirs)
    check("the timestep vector is one shorter than the sigma vector", len(t_ours), 30)
    worst_t = max(abs(a - b) for a, b in zip(t_ours, t_theirs, strict=True))
    close("and matches `1 - sigmas[:-1]` value for value", worst_t, 0.0,
          detail=f"max |difference| over 30 timesteps = {worst_t:g}")

    # THE DEFECT: the loop ran over the grid POINTS. The 31st iteration was a full-cost
    # forward pass whose delta was exactly zero.
    red("the shipped loop bound (`len(video_sigmas)`)", len(ours), plan.evaluations,
        detail="31 iterations for a 30-step request — the extra one stepped sigma 0 to "
               "sigma 0, paying a full evaluation to add nothing")
    close("the 31st transition's delta is exactly zero", ours[-1] - ours[-1], 0.0)

    # The audio schedule is the SAME uniform clock under a different shift. diffusers runs
    # a second scheduler at shift 3; the closed-form re-shift must produce those numbers.
    audio_ours = [shift_sigma(s, SHIFT_VIDEO, SHIFT_AUDIO) for s in ours]
    audio_theirs = reference_sigmas(evaluations + 1, SHIFT_AUDIO)
    worst_a = max(abs(a - b) for a, b in zip(audio_ours, audio_theirs, strict=True))
    close("re-shifting the video schedule == a second scheduler at shift 3", worst_a, 0.0,
          tol=1e-9, detail=f"max |difference| = {worst_a:g}")
    worst_p = max(
        abs(a - b) for a, b in zip(plan.audio_sigmas, audio_theirs, strict=True)
    )
    close("  and the plan carries exactly those", worst_p, 0.0, tol=1e-9)

    for preset in (20, 50):
        p = _plan(preset)
        check(f"the {preset}-step preset is {preset} evaluations", p.evaluations, preset)
        check(f"  over {preset + 1} grid points", p.grid_points, preset + 1)


def arm_pixels() -> None:
    print("\n== the PIXEL CONVERSION — [0,1] to bytes, applied exactly ONCE ==")
    import numpy as np

    # The VAE's own output convention: `_finalize_pixels` un-normalizes by the ImageNet
    # statistics and CLAMPS to [0, 1]. diffusers does the same and says so.
    ramp = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    got = pixel_bytes(ramp)
    check("a [0,1] ramp becomes whole bytes", [int(v) for v in got], [0, 64, 128, 191, 255])
    check("BLACK STAYS BLACK", int(got[0]), 0)
    check("WHITE STAYS WHITE", int(got[-1]), int(PIXEL_FULL_SCALE))

    # THE DEFECT (#522d): the second rescale, applied to an already-[0,1] tensor.
    doubled = ((ramp / 2 + 0.5).clip(0.0, 1.0) * 255).round()
    red("the shipped `decoded / 2 + 0.5` on top of the VAE's own [0,1]",
        [int(v) for v in doubled], [int(v) for v in got],
        detail="black reads 128 and the whole picture folds into the top half of the "
               "range — no correct render could ever be dark")
    check("  specifically, black would have read", int(doubled[0]), 128)

    # Out-of-range input clamps rather than wrapping, in both directions.
    edges = pixel_bytes(np.array([-0.4, 1.4], dtype=np.float32))
    check("values outside [0,1] clamp, never wrap", [int(v) for v in edges], [0, 255])

    # Real shaped clip, so the arm has exercised the conversion the endpoint calls.
    clip = np.zeros((4, 8, 8, 3), dtype=np.float32)
    clip[2:] = 1.0
    bytes_clip = pixel_bytes(clip)
    check("a (T,H,W,3) clip keeps its shape", tuple(bytes_clip.shape), (4, 8, 8, 3))
    check("  black frames stay 0 and white frames stay 255",
          (int(bytes_clip[0].max()), int(bytes_clip[-1].min())), (0, 255))


def arm_facts() -> None:
    print("\n== REQUESTED vs OBSERVED media facts — the agreement the first render lacked ==")

    frames = 124
    facts = MediaFacts(width=1344, height=768, frames=frames, fps=FPS, sample_rate=32000)
    close("the 5 s preset's video duration", facts.duration_s, frames / 24.0)
    check("the A/V tolerance is one video frame",
          round(facts.av_tolerance_s, 8), round(AV_DURATION_TOLERANCE_FRAMES / FPS, 8))

    # The two latent geometries have to AGREE by construction: the audio latent count is
    # derived from the video frame count, so its duration must land inside one frame.
    audio_seconds = audio_latents(frames) / 40.0
    if facts.av_agrees(audio_seconds):
        observe(
            "the audio geometry agrees with the video geometry, by construction",
            f"{audio_seconds:.4f} s of audio vs {facts.duration_s:.4f} s of video, "
            f"{facts.av_drift(audio_seconds) * 24:.3f} frames apart",
        )
    else:
        failed("the two geometries disagree", f"{audio_seconds} vs {facts.duration_s}")

    # THE PERMANENT RED ARM, in numbers: the measured facts of `daf50552...`, the first
    # se-002 render — 103 frames of video (4.292 s) against 5.180 s of audio. Note WHICH
    # check catches it, because the distinction is the lesson: its soundtrack was very
    # nearly the right length FOR THE REQUEST (0.013 s out), so an audio-duration check
    # alone passes it. What is unarguable is the frame count and the container's
    # disagreement with itself.
    observed = MediaFacts(width=1344, height=768, frames=103, fps=FPS, sample_rate=32000)
    red("its 103 delivered frames against the 124 requested", observed.frames, facts.frames,
        detail="the wrong 31 latents decoded deterministically into exactly 103 frames — "
               "this is the check that catches it, and the endpoint never ran one")
    if not observed.av_agrees(5.18):
        observe(
            "and the container disagrees with ITSELF — 4.292 s of video, 5.180 s of audio",
            f"{observed.av_drift(5.18):.3f} s apart, tolerance "
            f"{observed.av_tolerance_s:.4f} s",
        )
    else:
        failed("the artifact's own streams were scored as agreeing")
    if facts.av_agrees(5.18):
        observe(
            "the SAME soundtrack read against the REQUEST passes — measured, not assumed",
            f"{facts.av_drift(5.18):.3f} s from the requested 5.167 s, which is why an "
            "audio-duration check is not a substitute for counting frames",
        )
    else:
        failed("the requested-side A/V check should have passed", f"{facts.av_drift(5.18)}")

    for seconds, expect in ((5, 124), (10, 243), (15, 362)):
        got = _frames_for(seconds)
        check(f"the {seconds} s preset resolves to {expect} frames", got, expect)
        check(f"  which is on the grid ({latent_frames(got)} latents)",
              pixel_frames(latent_frames(got)), got)


# --------------------------------------------------------------- helpers


def _plan(evaluations: int) -> Any:
    grid = latent_grid(124, 1344, 768)
    return build_timestep_plan(
        task="fl2va",
        structure="h3-adaln-curve",
        evaluations=evaluations,
        layout=PackedLayout(64, grid),
        sigma_shift_video=SHIFT_VIDEO,
        sigma_shift_audio=SHIFT_AUDIO,
        visual_cond_timestep=None,
        audio_cond_timestep=None,
    )


def _rotary_extent(latent_t: int) -> float:
    """The temporal span a clip of `latent_t` latent frames occupies on the rotary clock:
    `5/3 * (1, 4, 4, 4, 4)` per latent frame, cycling. Written out here rather than
    imported, so the arm is not checking a function against itself."""
    return sum((5.0 / 3.0) * (1, 4, 4, 4, 4)[k % 5] for k in range(latent_t))


def _frames_for(seconds: int) -> int:
    """The endpoint's duration preset resolution, as the reference states it: snap UP to
    the next `17k + 5`. ComfyUI spells it `while n % 17 != 5: n += 1`."""
    n = seconds * FPS
    while n % 17 != 5:
        n += 1
    return n


ARMS = {
    "sign": arm_sign,
    "geometry": arm_geometry,
    "schedule": arm_schedule,
    "windows": arm_windows,
    "pixels": arm_pixels,
    "facts": arm_facts,
}


def main() -> int:
    names = sys.argv[1:] or list(ARMS)
    for name in names:
        if name not in ARMS:
            print(f"unknown arm {name!r}: {', '.join(ARMS)}", file=sys.stderr)
            return 2
        try:
            ARMS[name]()
        except Exception:
            failed(f"arm {name} raised")
            traceback.print_exc()
    print(f"\n{_failures} failed")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
