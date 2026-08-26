#!/usr/bin/env python
"""THE REBASE'S FALSIFIER — the graph `h3_ref/` constructs against the OFFICIAL release's
own diffusers-format headers, and the four #522 seams against upstream's own arithmetic.
Decided on the control plane, with ZERO weight bytes and zero dollars.

    nice -n 19 .venv/bin/python scripts/h3-diffusers-keys.py [keys|geometry|schedule|solver|curve]

WHAT THIS ANSWERS, AND WHAT IT DOES NOT. `scripts/h3-keys.py` asks the same question of the
hand port against the community carrier; this asks it of the upstream classes against the
official tree. Both are FILL questions: a missing key, an extra key, a transposed shape or a
wrong dtype is a failed serve on a rented card, and here it is a diff on this box. Neither
is an output-verification, and a green run here says nothing about whether a render looks
right — that is the GPU oracle's job and it has not been done.

THE SOURCE IS DECLARED, and unreadable is a REFUSAL rather than a skip: the pinned evidence
bank `~/cozy_v2/h3-evidence` carries the exact header bytes of the official release's
diffusers packaging. If the bank is not there this script must not print a verdict.

The transformer, video VAE and conditioner are SHARDED, so their key sets come from the
banked `*.index.json` weight maps — which name every key across every shard, and are the
reason a 66 GB tree can be checked from 22 KB of JSON. The audio VAE is a single file and
its header carries shapes and dtypes as well; where a header is available this checks
shapes, and where only an index is available it says so rather than claiming more.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
import sys
from typing import Any, NoReturn

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
BANK = pathlib.Path.home() / "cozy_v2" / "h3-evidence"
ROW = "minimax-diffusers-tree.json"

#: role -> (the class's config document, the file whose key set is the artifact's topology).
#: `transformer_ref` is deliberately absent: its banked config and index are BYTE-IDENTICAL
#: to `transformer`'s (they resolve to the same cache blob), so checking it twice would
#: report two facts and measure one. That identity is itself job-001's N-ary claim.
CARRIERS: dict[str, tuple[str, str]] = {
    "transformer": (
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ),
    "text_encoder": (
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
    ),
    "video_vae": (
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors.index.json",
    ),
    "audio_vae": (
        "audio_vae/config.json",
        "audio_vae/diffusion_pytorch_model.safetensors",
    ),
}

DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def refuse(message: str) -> NoReturn:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)


def banked(path: str) -> Any:
    """The pinned bytes for one banked file. Bank absent, row absent, cache absent or a
    digest disagreement all REFUSE — this proof cites pinned upstream bytes and cannot be
    run without them."""
    row_file = BANK / "rows" / ROW
    if not row_file.is_file():
        refuse(
            f"the declared evidence source {row_file} is not readable — this proof cites "
            "the official release's pinned headers and cannot be run without them"
        )
    row: dict[str, Any] = json.loads(row_file.read_bytes())
    hit = [f for f in row["banked_files"] if f["path"] == path]
    if not hit:
        refuse(f"{path} is not banked in {ROW}")
    digest = hit[0]["fetch"]["bytes_sha256"]
    cached = BANK / "cache" / digest
    if not cached.is_file():
        refuse(f"the cached bytes {digest[:16]}… for {path} are missing from the bank")
    raw = cached.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        refuse(f"the cached bytes for {path} do not match their banked digest")
    return json.loads(raw)


def build_component(role: str, config: dict[str, Any]) -> Any:
    """`h3_ref`'s builder, reached the way the release archive lays the endpoint out: the
    model library sits beside the endpoint module, not on the path this script runs from."""
    sys.path.insert(0, str(ROOT / "h3"))
    from h3_ref import build_component as build

    return build(role, config)


def pinned(role: str) -> tuple[set[str], dict[str, tuple[int, ...]], int | None]:
    """The artifact's own destination table. An index names keys and a total size; a header
    additionally names shapes."""
    document = banked(CARRIERS[role][1])
    if "weight_map" in document:
        return set(document["weight_map"]), {}, int(document["metadata"]["total_size"])
    keys = {k for k in document if k != "__metadata__"}
    shapes = {k: tuple(document[k]["shape"]) for k in keys}
    return keys, shapes, None


def constructed(role: str) -> tuple[dict[str, tuple[int, ...]], set[str]]:
    """The graph `h3_ref` builds, censused the way the runtime censuses it: one
    `state_dict()` walk of the component root, on `meta`, so no byte is allocated.

    The second return is the NON-PERSISTENT buffers — registered on the module, absent from
    the state dict, and therefore not fill destinations at all. `rope.inv_freq` is the one
    #508c named as the native/diffusers delta, and this is where that resolves: upstream
    derives it at construction, so neither side carries it and there is nothing to map.
    """
    config = {role: banked(CARRIERS[role][0])}
    with torch.device("meta"):
        module = build_component(role, config)
    table = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    return table, {n for n, _ in module.named_buffers()} - set(table)


def summarize(keys: list[str], limit: int = 8) -> list[str]:
    """Templates, not a wall of block indices — 50 keys differing the same way is ONE fact."""
    families = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in keys)
    return [f"{count}x {name}" for name, count in sorted(families.items())][:limit]


def check_keys(role: str) -> bool:
    want, want_shapes, size = pinned(role)
    got, nonpersistent = constructed(role)
    missing = sorted(want - set(got))
    extra = sorted(set(got) - want)
    shape = sorted(k for k in want_shapes if k in got and want_shapes[k] != got[k])
    ok = not (missing or extra or shape)
    gib = f"  {size / 2**30:7.1f} GiB" if size else ""
    scope = "keys+shapes" if want_shapes else "keys"
    print(
        f"{'OK  ' if ok else 'FAIL'} {role:14s} pinned {len(want):5d}  constructed "
        f"{len(got):5d}  [{scope}]{gib}"
    )
    for label, keys in (
        ("missing (the artifact has it, the graph does not)", missing),
        ("extra (the graph demands it, the artifact has none)", extra),
        ("shape", shape),
    ):
        if not keys:
            continue
        print(f"       {len(keys)} {label}")
        for line in summarize(keys):
            print(f"         {line}")
    for name in sorted(nonpersistent):
        print(f"       derived, never filled (non-persistent buffer): {name}")
    return ok


def check_geometry() -> bool:
    """#522b, against upstream's own helpers. The port used a generic divide-by-four and
    turned 124 frames into 31 latents, which the VAE then deterministically decoded into the
    103 frames se-002 measured. The real relation is 17n+5 frames to 5n+2 latents, and it is
    the released VAE's `clip_length` / chunk pair rather than a constant anyone maintains."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_FPS,
        align_num_frames,
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    vae = banked(CARRIERS["video_vae"][0])
    per_chunk, per_latent = int(vae["clip_length"]), 5
    ok = True
    print(
        f"     grid {per_chunk}n+{per_latent} frames -> {per_latent}n+2 latents, "
        f"{MINIMAX_H3_FPS} fps"
    )
    for seconds in (5, 10, 15):
        frames = align_num_frames(seconds * MINIMAX_H3_FPS, per_chunk, per_latent)
        latents = video_latent_num_frames(frames, per_chunk, per_latent)
        audio = audio_latent_num_frames(frames)
        naive = frames // 4
        good = latents != naive
        ok &= good
        print(
            f"{'OK  ' if good else 'FAIL'} {seconds:2d} s -> {frames:4d} frames -> "
            f"{latents:3d} video latents, {audio:4d} audio latents "
            f"(the generic //4 would have said {naive} — the defect)"
        )
    return ok


def check_schedule() -> bool:
    """#522a and #522c, against upstream's own scheduler. Two independent facts:

      * a request for N sigma grid points drives N-1 MODEL EVALUATIONS, and upstream names
        them separately — `sigmas` has N entries, `timesteps` has N-1, and the terminal zero
        is inside the requested count rather than appended after it;
      * the velocity is DATA-WARD. A pure-noise sample stepped with the velocity that points
        at a known target must move TOWARD that target. The port's `x + (sigma_next - sigma)*v`
        moved away from it every iteration, which is the anti-denoising #522a names.
    """
    from diffusers import MiniMaxH3Scheduler

    ok = True
    for steps, shift in ((30, 12.0), (30, 3.0), (20, 12.0), (50, 12.0)):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps)
        points, evaluations = scheduler.sigmas.numel(), scheduler.timesteps.numel()
        good = evaluations == points - 1 and float(scheduler.sigmas[-1]) == 0.0
        ok &= good
        print(
            f"{'OK  ' if good else 'FAIL'} shift {shift:5.1f}  {steps} requested -> "
            f"{points} sigma points -> {evaluations} model evaluations, terminal "
            f"sigma {float(scheduler.sigmas[-1]):.1f}"
        )

    generator = torch.Generator().manual_seed(0)
    target = torch.randn(64, generator=generator)
    sample = torch.randn(64, generator=generator)
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(30)
    before = float((sample - target).pow(2).mean())
    for timestep in scheduler.timesteps:
        sigma = 1.0 - float(timestep)
        # The data-ward velocity that reaches `target` exactly: x0 = x_t + sigma*v.
        velocity = (target - sample) / sigma
        sample = scheduler.step(velocity, timestep, sample).prev_sample
    after = float((sample - target).pow(2).mean())
    good = after < before * 1e-6
    ok &= good
    print(
        f"{'OK  ' if good else 'FAIL'} data-ward: a velocity pointing at a known target "
        f"converges to it, {before:.4f} -> {after:.3e} over 29 evaluations"
    )
    return ok


def check_solver() -> bool:
    """THE TWO-SCHEDULER SEAM (#540's banked next-lane item), END TO END.

    H3 runs TWO schedules per request — video at shift 12.0, audio at shift 3.0 — and this
    endpoint drives both from ONE `H3Solver` over one `TimestepPlan`, where upstream drives
    two `MiniMaxH3Scheduler` instances. The fix lane proved a SINGLE step agrees to 1e-12; a
    single step says nothing about the other 29, because the two sides index their grids
    differently — upstream carries a `_step_index` cursor that advances inside `step()`,
    this port indexes the plan — and an off-by-one there is invisible until the last
    evaluation.

    THREE ARMS, because the seam and the schedule's REPRESENTATION are different subjects
    and reporting one number for both would attribute a float32 rounding to the update rule:

      1. THE GRID. `flow_sigmas` computes in Python float64; upstream builds `linspace` in
         float32. They are the same expression, so they agree to float32 resolution and no
         further, and that difference is measured here rather than absorbed.
      2. THE SEAM, on a SHARED grid. Both sides driven from upstream's own sigma values, so
         the only thing left that can differ is the update expression and the order it is
         applied in. It is exact to float32 resolution and NOT bit-exact, for a reason the
         upstream scheduler states in its own `step()`: it recovers the x0 sigma from the
         TIMESTEP (`1 - t`, a float32 round trip that is lossy below sigma 0.5) while taking
         the Euler ratio from the sigma GRID, deliberately keeping the two sources apart.
         `SolverStep.advance_*` is the algebraic rearrangement of that update with ONE sigma,
         so the two agree exactly in real arithmetic and differ by that round trip in float32
         — which is why the audio schedule, whose shift 3.0 grid spends more of its length
         below 0.5, carries the larger residual of the two.
      3. THE SHIPPED PAIR. The port's own grid against upstream's own grid, whose residual
         must stay inside what arm 1 measured: `evaluations` steps each carrying at most one
         float32 epsilon of schedule error.

    The velocity is a fixed pseudo-random field that DEPENDS ON THE CURRENT SAMPLE, which is
    what makes every arm sensitive to ORDER: a schedule that is right as a set and wrong as
    a sequence passes a per-step check and fails these.

    RED CONTROL on arm 2: the defective `x + (sigma_next - sigma) * v` update (#522a) must
    DISAGREE, and it is asserted to, so a green arm is never a green nobody fired.
    """
    sys.path.insert(0, str(ROOT / "h3"))
    from diffusers import MiniMaxH3Scheduler

    from h3_arch.layout import H3Solver, TimestepPlan, flow_sigmas

    evaluations = 30
    #: float32 has 24 bits of mantissa, so one grid value carries at most this much relative
    #: error against the float64 expression — and a trajectory carries at most one per
    #: evaluation. The bound is that product, not a number chosen to make the arm pass.
    float32_eps = 2.0**-23
    bound = evaluations * float32_eps

    def velocity(sample: torch.Tensor, index: int) -> torch.Tensor:
        """Deterministic, and a function OF THE SAMPLE, so the trajectory diverges the moment
        a step is taken out of order or at the wrong sigma."""
        generator = torch.Generator().manual_seed(1000 + index)
        return torch.randn(sample.shape, generator=generator, dtype=torch.float64) + sample

    def solver_over(sigmas: tuple[float, ...], shift: float) -> Any:
        """The SHIPPED solver over a given grid. A plan is built directly rather than
        through `build_timestep_plan` so the grid under test can be upstream's own values;
        everything downstream of it is the endpoint's real code."""
        plan = TimestepPlan(
            task="fl2va",
            structure="h3-adaln-curve",
            video_sigmas=sigmas,
            audio_sigmas=sigmas,
            sigma_shift_video=shift,
            sigma_shift_audio=shift,
            visual_cond_timestep=None,
            audio_cond_timestep=None,
            modality_tags=(0, 1, 2),
            scheduler="flow_match_uniform",
            output_dtype="float32",
        )
        return H3Solver(plan)

    def ours(
        sigmas: tuple[float, ...], shift: float, modality: str, *, red: bool = False
    ) -> torch.Tensor:
        """`SolverStep.advance_video` is the shipped update and is what runs here — the red
        control is the ONE expression that is written out, because it is the one that was
        deleted (#522a) and has to be spelled to be fired."""
        generator = torch.Generator().manual_seed(7)
        sample = torch.randn(256, generator=generator, dtype=torch.float64)
        for step in solver_over(sigmas, shift).steps():
            v = velocity(sample, step.index)
            if red:
                sample = sample + (step.sigma_video_next - step.sigma_video) * v
            elif modality == "video":
                sample = step.advance_video(sample, v)
            else:
                sample = step.advance_audio(sample, v)
        return sample

    def theirs(scheduler: Any) -> torch.Tensor:
        generator = torch.Generator().manual_seed(7)
        sample = torch.randn(256, generator=generator, dtype=torch.float64)
        for index, timestep in enumerate(scheduler.timesteps):
            sample = scheduler.step(velocity(sample, index), timestep, sample).prev_sample
        return sample

    ok = True
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        native = flow_sigmas(evaluations, shift)
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(evaluations + 1)
        upstream = tuple(float(s) for s in scheduler.sigmas)

        # 1 — the grid, to float32 resolution
        points = len(upstream) == len(native)
        grid_delta = (
            max(abs(a - b) for a, b in zip(native, upstream, strict=True)) if points else 1.0
        )
        grid_ok = points and grid_delta <= float32_eps

        # 2 — the seam, both sides on upstream's own values
        reference = theirs(scheduler)
        scale = float(reference.abs().max())
        seam = float((ours(upstream, shift, name) - reference).abs().max()) / scale
        red = float((ours(upstream, shift, name, red=True) - reference).abs().max()) / scale
        seam_ok = seam <= float32_eps and red > 1e-3

        # 3 — the shipped pair, inside what arm 1 measured
        shipped = float((ours(native, shift, name) - reference).abs().max()) / scale
        shipped_ok = shipped <= bound

        ok &= grid_ok and seam_ok and shipped_ok
        verdict = "OK  " if (grid_ok and seam_ok and shipped_ok) else "FAIL"
        print(
            f"{verdict} {name:5s} shift {shift:4.1f}: {len(native)} grid points, "
            f"{evaluations} evaluations"
        )
        print(
            f"       grid   max|d| {grid_delta:.3e} (float32 resolution {float32_eps:.3e})"
        )
        print(
            f"       seam   shared grid, final rel {seam:.3e} "
            f"(bound float32 eps {float32_eps:.3e} — upstream's own x0 round trip); "
            f"the #522a sign as red control: {red:.3e}"
        )
        print(
            f"       shipped native grid vs upstream's, final rel {shipped:.3e} "
            f"(bound {evaluations} x float32 eps = {bound:.3e})"
        )
    return ok


#: The banked ATTENTION-FUSION delta between the two dialects, as arithmetic rather than as
#: a claim. The community carrier fuses q/k/v into one `attn.qkv_proj.weight` per attention
#: module where the official packaging keeps three, over 52 modules (50 blocks + 2 refiner
#: blocks), and it carries `rope.inv_freq` where upstream derives it. So any community key
#: count maps to the official one by `+ 52*2 - 1`, and that identity is what lets a curve
#: topology built on the upstream class be checked against a header written in the other
#: dialect without a rename table.
_ATTENTION_MODULES = 52
_FUSION_DELTA = _ATTENTION_MODULES * 2 - 1


def check_curve() -> bool:
    """The curve delta's topology, against the banked curve carrier's own header facts.

    The community artifact is written in the other key dialect, so this does NOT diff key
    names — it checks the two things the dialect cannot hide: the three shapes that ARE the
    curve (the collapsed projection input, the final-layer projection, and the table that
    replaced the timestep embedder), and the destination count, reconciled across the
    dialects by the fusion identity above.
    """
    row = json.loads((BANK / "rows" / "adaln-topologies.json").read_bytes())
    curve = row["h3_adaln_curve"]
    basis = int(curve["curve_only_keys"]["adaln_t_table"]["shape"][1])
    grid = int(curve["curve_only_keys"]["adaln_t_table"]["shape"][0])
    config = dict(banked(CARRIERS["transformer"][0]))
    config.update(time_embed_dim=basis, structure="h3-adaln-curve", adaln_curve_grid=grid)
    with torch.device("meta"):
        table = build_component("transformer", {"transformer": config}).state_dict()

    sample = curve["adaln_proj_weight_shape_sample"]
    want = {
        "transformer_blocks.0.adaln_proj.linear.weight": tuple(
            sample["blocks.0.adaln_proj.linear.weight"]
        ),
        "norm_out.linear.weight": tuple(sample["final_layer.adaln_proj.linear.weight"]),
        "time_embedder.adaln_t_table": (grid, basis),
    }
    ok = True
    for key, shape in want.items():
        got = tuple(table[key].shape) if key in table else None
        good = got == shape
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {key:48s} {got} (banked {shape})")
    embedder = [k for k in table if k.startswith("time_embedder.linear_")]
    gone = not embedder
    ok &= gone
    print(
        f"{'OK  ' if gone else 'FAIL'} the timestep embedder is gone: "
        f"{len(embedder)} `time_embedder.linear_*` destinations remain"
    )
    expected = int(curve["logical_tensors"]) + _FUSION_DELTA
    matched = len(table) == expected
    ok &= matched
    print(
        f"{'OK  ' if matched else 'FAIL'} {len(table)} destinations = the carrier's "
        f"{curve['logical_tensors']} + {_FUSION_DELTA} (q/k/v unfused over "
        f"{_ATTENTION_MODULES} modules, less the derived rope.inv_freq)"
    )
    return ok


def main() -> int:
    known = ("keys", "geometry", "schedule", "solver", "curve")
    sections = sys.argv[1:] or list(known)
    unknown = [s for s in sections if s not in known]
    if unknown:
        refuse(f"unknown section(s): {', '.join(unknown)}")
    results: list[bool] = []
    if "keys" in sections:
        print("KEY CENSUS — the constructed graph against the official tree's own headers")
        roles = list(CARRIERS)
        results += [check_keys(role) for role in roles]
        total = sum(len(pinned(role)[0]) for role in roles)
        print(f"     {sum(results)}/{len(roles)} roles key-exact over {total} destinations\n")
    if "geometry" in sections:
        print("TEMPORAL GEOMETRY — #522b, against the released VAE's own chunk relation")
        results.append(check_geometry())
        print()
    if "schedule" in sections:
        print("SCHEDULE — #522a and #522c, against the official H3 scheduler")
        results.append(check_schedule())
        print()
    if "solver" in sections:
        print("SOLVER SEAM — one H3Solver against upstream's TWO schedulers, all 30 steps")
        results.append(check_solver())
        print()
    if "curve" in sections:
        print("CURVE DELTA — the optimization lane's topology, against the banked carrier")
        results.append(check_curve())
        print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
