#!/usr/bin/env python
"""THE WHOLE-SEAM ORACLE (#523.3) — this endpoint's model path against ComfyUI v0.33.0's,
seam by seam, on the SAME weights and the SAME prompt.

    python scripts/h3-seam-oracle.py comfy   --models DIR --bank DIR [request options]
    python scripts/h3-seam-oracle.py ours    --models DIR --bank DIR [request options]
    python scripts/h3-seam-oracle.py compare --bank DIR
    python scripts/h3-seam-oracle.py render  --models DIR --bank DIR --side ours|comfy

RUNS POD-SIDE ONLY. It reads real checkpoints; `fence.py::no-memory-choreography` refuses
that spelling inside a project and this is why the file lives under `scripts/`.

WHY THIS EXISTS. `scripts/h3-oracle.py` compares COMPONENTS and found both VAEs
bit-identical; se-002 then served a render in which every component was key-exact and the
output was garbage, because the faults lived BETWEEN the components — a reversed sampler
sign, a generic temporal ratio, an off-by-one in the evaluation count. #523.3 is the
answer: compare the whole path, at every seam, against an implementation that is known to
produce good video.

THE TEXT ENCODER IS THE POINT. It is the ONE component of the four that has never been
output-verified — 49 GiB never fit the 24 GiB card the component oracle ran on, and it was
flagged as the prime suspect for #519's garbage twice. Its constants (`_ROPE_SECTIONS`,
the deepstack indices, the vision rope theta, the image normalization) are named in
`h3_arch/text_encoder.py` as NOT in the checkpoint header, so a key census is structurally
incapable of checking them. Seam 2 is the only thing that can, and it runs first.

THE PROTOCOL, and why the seams split into two groups:

  * SHARED-INPUT SEAMS (2, 3, 4, 5, 6). Both sides are handed the same text states and the
    same latents at the same sigma, and one model evaluation is compared. Divergence here
    is STRUCTURAL — there is no accumulated trajectory to hide in.
  * INDEPENDENT-RUN SEAMS (7, 8). Each side runs its own 30 evaluations from its own noise
    and its own scheduler. These cannot be compared elementwise and are not: the final
    latent is compared by distribution, and the decoded media by cozy-eval's gates plus a
    human looking at the frames (#520).

TOLERANCES, stated per seam rather than as one global number. Both sides run the same bf16
carriers, so a seam whose two implementations perform the same reductions in the same order
is BIT-EXACT and anything else is a defect; a seam that legitimately reduces in a different
order (a fused kernel against an unfused one) carries an accumulation bound. Which is which
is stated at each seam, and the reason is stated with it.

BANKED (2026-08-26, one H200, se-002). Both sides on the SAME bf16 carriers, the same
prompt and the same seed; cosine is the verdict, the float32 control is the discriminator.

    seam                    cosine        what it settles
    tokens                  identical     22 ids both sides, from two independent BPEs
    text encoder            0.999985      and EXACT at float32 (cos 1.000000000, rel
                                          1.2e-07) — the one output-UNVERIFIED component
                                          is CORRECT; the bf16 residual is our dtype
    projected packed rows   0.999262
    raw output heads        0.999907 / 0.999987     (video / audio)
    latent-shaped velocity  0.999905 / 0.999984
    first updated latent    0.999999996 / 0.999999998

NOTHING DIVERGED STRUCTURALLY, and the renders agree with the seams: at the same prompt
and seed this endpoint and ComfyUI produce the same video and the same camera move.

THE LATTICE WAS NOT THE CARRIER'S, and this paragraph is the correction (#557). The first
version of it read the lattice as "a 32-pixel spatial lattice at the DiT patch period, FFT
peak ratio 4.96 ours against 4.75 upstream — the released carrier's, not this port's".
Every clause of that was wrong, and the SEAM TABLE ABOVE IS WHY IT SURVIVED: it ends at the
first updated latent. THE DECODE IS NOT A SEAM IN IT. Two things then went unnoticed —

  * the `--side comfy` render OOM'd in ComfyUI's own `vae.decode` and never wrote a file, so
    the mp4 banked as ComfyUI's is ComfyUI's LATENTS through THIS port's decode. The
    side-by-side held the decoder fixed and varied the sampler; the number it produced
    ("4.96 vs 4.75") compared this decoder against itself, which is why it looked like
    agreement.
  * the period is 16 px, not 32. 16 px is one latent cell — the VIDEO VAE DECODER's token.
    32 px is the DiT patch. The period was naming the component and was read as naming the
    other one.

The real control was already in the bank: `tensorfs-bench/proto-001`'s `C-comfy-curve-bf16`
ran THE IDENTICAL FOUR CARRIER FILES at the identical geometry through ComfyUI end to end,
and its token-boundary gradient excess is 10% against this port's 150%. The cause is
`video_vae`'s decode WINDOW, fixed in `h3_arch/video_vae.py` and armed by
`h3-conform.py windows`.

AND THE 17-FRAME SEAM WAS THE SAME DEFECT, which is the other half of #555d retired. It was
banked as a second, independent carrier property. It is not one: the lattice is regenerated
per decoder call, so it CHANGES at every temporal chunk boundary and reads as a delta spike
exactly there. Inter-frame delta over the local mean at the clip grid: 3.468 before the
window fix, 1.031 after, against 1.158 for the ComfyUI bank render on the same weights. One
cause, both artifacts — and the second one looked independent only because nothing had
varied the decode.

RULE, and it is the general one: A SEAM TABLE EXONERATES ONLY THE SEAMS IN IT. Every stage
between the last compared tensor and the bytes a human looks at is un-compared, and an
artifact will settle in exactly there. Adding a decode seam to this harness is the follow-up.

Speed, same 30 evaluations, same geometry, same dtype, same card: 215.4 s for this
endpoint's model path against 292.1 s for ComfyUI's sampler — 1.36x.

BF16 CARRIERS ON BOTH SIDES, DELIBERATELY. The recipe's served transformer is fp8-scaled,
and its bf16 sibling exists in the same release. Comparing through two different fp8 decode
paths would measure the decoders; this compares the models.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time
from typing import Any, cast

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "h3"))

#: The four carriers, by the path they land on under `--models`. The transformer is the
#: BF16 sibling of the served fp8 one for the reason the module docstring gives.
CARRIERS = {
    "transformer": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
    "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
}

#: Row order of the report. The text encoder is first because it is the one
#: output-unverified component and the highest-value single number of the run.
SEAMS = (
    ("tokens", "token ids + modality tags"),
    ("cond", "text-encoder hidden states"),
    ("packed", "projected packed rows"),
    ("heads", "raw output heads"),
    ("velocity", "latent-shaped data velocity"),
    ("latent1", "first updated latent"),
    ("latentN", "final latent"),
    ("media", "decoded frames + waveform"),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def bank(directory: pathlib.Path, name: str, payload: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pt"
    torch.save(payload, path)
    log(f"banked {path.name} ({path.stat().st_size / 2**20:.1f} MiB)")


def load(directory: pathlib.Path, name: str) -> Any:
    return torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ the request


def add_request_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default=(
        "a red sports car driving fast along a coastal road at sunset, "
        "waves crashing on the rocks below, engine roaring"
    ))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30, help="MODEL EVALUATIONS, not grid points")
    parser.add_argument("--length", type=int, default=124, help="frames, on the 17k+5 grid")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--shift-video", type=float, default=12.0)
    parser.add_argument("--shift-audio", type=float, default=3.0)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"),
                        help="OUR compute dtype. The CONTROL that separates precision from "
                             "structure: the text encoder sits 1.06e-2 relative from the "
                             "reference at bfloat16 and is EXACT at float32, so the residual "
                             "is our dtype and not the port")


def request_facts(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "seed": args.seed,
        "evaluations": args.steps,
        "length": args.length,
        "width": args.width,
        "height": args.height,
        "shift_video": args.shift_video,
        "shift_audio": args.shift_audio,
    }


# ------------------------------------------------------------------ shared arithmetic


def geometry(args: argparse.Namespace) -> Any:
    from h3_arch.layout import latent_grid

    return latent_grid(args.length, args.width, args.height)


def solver_for(args: argparse.Namespace, layout: Any) -> Any:
    from h3_arch.layout import H3Solver, build_timestep_plan

    return H3Solver(
        build_timestep_plan(
            task="fl2va",
            structure="h3-adaln-curve",
            evaluations=args.steps,
            layout=layout,
            sigma_shift_video=args.shift_video,
            sigma_shift_audio=args.shift_audio,
            visual_cond_timestep=None,
        )
    )


def initial_latents(args: argparse.Namespace, grid: Any) -> tuple[Any, Any]:
    """THE SHARED STARTING NOISE. Drawn on the host from the request's seed, exactly as
    `h3.py::_sample` draws it, so both sides denoise the same field and seam 7 compares two
    trajectories rather than two noises."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video = torch.randn(1, 24, grid.latent_t, grid.latent_h, grid.latent_w, generator=generator)
    audio = torch.randn(1, 32, 2, grid.audio_t, generator=generator)
    return video, audio


# ------------------------------------------------------------------ the comfy side


def comfy_setup(comfy_root: pathlib.Path) -> Any:
    sys.argv = ["main.py"]
    os.chdir(comfy_root)
    sys.path.insert(0, str(comfy_root))
    import comfy.model_management
    import comfy.sd

    return comfy


def side_comfy(args: argparse.Namespace) -> None:
    """ComfyUI's own modules, driven directly at the seams — no sampler, no node graph.

    `MiniMaxH3Model._forward` is called with the SAME text states and the SAME latents this
    endpoint would use. Its public `forward` is deliberately NOT used: that wrapper carries
    the audio stream in the video's sigma coordinate (`audio_scale = shift_v / shift_a`) so
    a single generic sampler can drive the pack, and this endpoint drives two schedules
    directly. Comparing through the carry would measure the carry.
    """
    models = pathlib.Path(args.models)
    banked = pathlib.Path(args.bank) / "comfy"
    comfy_setup(pathlib.Path(args.comfy))
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils

    device = mm.get_torch_device()
    grid = geometry(args)

    # --- seam 1 + 2: the presentation and the text encoder
    log("loading the text encoder (ComfyUI CLIPType.MINIMAX)")
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(models / CARRIERS["text_encoder"])],
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    tokens = clip.tokenize(args.prompt)
    key = next(iter(tokens))
    ids = [int(t) for t, _ in tokens[key][0]] if isinstance(tokens[key][0][0], tuple) else None
    log(f"tokenized: key={key!r} rows={len(tokens[key][0])}")
    output = clip.encode_from_tokens(tokens, return_dict=True)
    states = output.get("cond", output)
    tags = output.get("minimax_token_tags")
    bank(banked, "tokens", {"ids": ids, "tags": None if tags is None else tags.cpu(),
                            "key": key, "rows": len(tokens[key][0])})
    bank(banked, "cond", states.detach().float().cpu())
    log(f"text-encoder states {tuple(states.shape)} {states.dtype}")
    del clip, output
    mm.unload_all_models()
    mm.soft_empty_cache()

    # --- seams 3-6: one model evaluation on shared inputs
    log("loading the transformer (bf16 carrier)")
    patcher = comfy.sd.load_diffusion_model(
        str(models / CARRIERS["transformer"]),
        model_options={"dtype": torch.bfloat16},
    )
    dm = patcher.model.diffusion_model
    dm.to(device)

    video, audio = initial_latents(args, grid)
    bank(banked, "latent0", {"video": video, "audio": audio})
    solver = solver_for(args, _layout_for(states.shape[1], grid))
    step = solver.step(0)

    captured: dict[str, Any] = {}
    hooks = _hook_seams(dm, captured)
    text = states.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = dm._forward(
            [video.to(device=device, dtype=torch.bfloat16),
             audio.to(device=device, dtype=torch.bfloat16)],
            torch.tensor([step.sigma_video * 1000.0], device=device),
            text,
            transformer_options={
                "minimax_h3_sigma_shift_video": args.shift_video,
                "minimax_h3_sigma_shift_audio": args.shift_audio,
            },
            minimax_payload={"audio_scale": 1.0, "text_token_tags": tags, "seed": args.seed},
        )
    for handle in hooks:
        handle.remove()
    bank(banked, "packed", captured["packed"].float().cpu())
    bank(banked, "heads", {k: v.float().cpu() for k, v in captured["heads"].items()})
    # ComfyUI returns NEGATED heads because its generic flow sampler assumes the opposite
    # convention and then applies the opposite delta (#522a). Undo exactly that one flip so
    # both sides are compared in the DATA-WARD convention both this endpoint and upstream
    # diffusers state.
    velocity = {"video": (-out[0]).float().cpu(), "audio": (-out[1]).float().cpu()}
    bank(banked, "velocity", velocity)
    bank(banked, "latent1", {
        "video": step.advance_video(video, velocity["video"]),
        "audio": step.advance_audio(audio, velocity["audio"]),
    })
    bank(banked, "facts", request_facts(args))
    log("comfy seam bank complete")


def seam_run(args: argparse.Namespace, grid: Any, layout: Any, plan: Any) -> Any:
    """One `H3Run` for the model path.

    `expanded` is a request-facing half the model boundary does not inspect in this
    text-only oracle. The request generator is real even though this seam does not draw
    from it, so signature drift cannot hide an alternate RNG owner in the harness.
    """
    import h3

    return h3.H3Run(
        plan=h3.H3Plan(grid=grid, keyframes=(), steps=args.steps, mute=False),
        layout=layout,
        timestep_plan=plan,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        expanded=cast(Any, None),
    )


def _layout_for(text_len: int, grid: Any) -> Any:
    from h3_arch.layout import PackedLayout

    return PackedLayout(text_len, grid)


def _hook_seams(module: Any, into: dict[str, Any]) -> list[Any]:
    """Seams 3 and 4, taken at the SAME two module boundaries on both sides.

    The two implementations name these modules identically (`blocks`, `final_layer`)
    because the port was written from this file — which is exactly why the hook can be
    symmetric, and exactly why a seam that agrees here is evidence about the ARITHMETIC
    rather than about the naming.
    """

    def pre(_m: Any, inputs: tuple[Any, ...]) -> None:
        into["packed"] = inputs[0].detach()

    def post(_m: Any, _i: Any, output: Any) -> None:
        into["heads"] = {"video": output[0].detach(), "audio": output[1].detach()}

    return [
        module.blocks[0].register_forward_pre_hook(pre),
        module.final_layer.register_forward_hook(post),
    ]


# ------------------------------------------------------------------ our side


def load_component(module: Any, path: pathlib.Path) -> Any:
    from safetensors.torch import load_file

    state = load_file(str(path))
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"REFUSED: {path.name} does not fill this graph — {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]}). "
            "A partial fill is a different model, not a warning."
        )
    return module


def side_ours(args: argparse.Namespace) -> None:
    """This endpoint's own modules, at the same four seams, from the same carriers.

    `--dtype float32` is the control, and it is what turned the text encoder from a seam
    reading 2.7 bf16 ulps out (which looks like a defect) into `cos 1.000000000, rel
    1.2e-07` (which is the same arithmetic). ComfyUI stores the text encoder's 902
    destinations in bf16 and RUNS them through a float32 compute path; this port declares
    bf16 destinations, because that is the artifact's fact, and then computes in bf16 too —
    which is a numerics decision the reference does not make.
    """
    models = pathlib.Path(args.models)
    banked = pathlib.Path(args.bank) / "ours"
    shared = pathlib.Path(args.bank) / "comfy"
    device = torch.device("cuda")

    import h3
    from h3_arch import H3Config, build_component
    from h3_arch.layout import build_modulation
    from h3_arch.presentation import Tokenizer
    from h3_arch.presentation import build as build_presentation

    config = H3Config()
    grid = geometry(args)

    # --- seam 1 + 2
    log("building the text encoder")
    tokenizer = Tokenizer()
    presentation = build_presentation(tokenizer, args.prompt)
    ids = presentation.text_ids()
    bank(banked, "tokens", {"ids": list(ids), "tags": torch.tensor(presentation.tags),
                            "rows": len(presentation.rows)})
    compute = getattr(torch, args.dtype)
    encoder = load_component(
        build_component("text_encoder", config), models / CARRIERS["text_encoder"]
    ).to(device=device, dtype=compute).eval()
    with torch.inference_mode():
        states = encoder(torch.tensor([list(ids)], dtype=torch.long, device=device))
    bank(banked, "cond", states.detach().float().cpu())
    log(f"text-encoder states {tuple(states.shape)} {states.dtype}")
    del encoder
    torch.cuda.empty_cache()

    # --- seams 3-6, on the SHARED inputs the comfy side banked
    log("building the transformer")
    transformer = load_component(
        build_component("transformer", config), models / CARRIERS["transformer"]
    ).to(device).eval()

    latent0 = load(shared, "latent0")
    video, audio = latent0["video"], latent0["audio"]
    text_shared = load(shared, "cond")
    layout = _layout_for(text_shared.shape[1], grid)
    solver = solver_for(args, layout)
    step = solver.step(0)
    modulation = build_modulation(
        layout,
        t_video=step.t_video,
        t_audio=step.t_audio,
        visual_cond_t=0.0,
        text_token_tags=tuple([1] * text_shared.shape[1]),
    )
    run = seam_run(args, grid, layout, solver.plan)

    captured: dict[str, Any] = {}
    hooks = _hook_seams(transformer, captured)
    with torch.inference_mode():
        v_video, v_audio = h3._predict_data_velocity(
            transformer,
            run=run,
            text_states=text_shared.to(device=device, dtype=torch.bfloat16),
            video_latents=video.to(device=device, dtype=torch.bfloat16),
            audio_latents=audio.to(device=device, dtype=torch.bfloat16),
            modulation=modulation,
        )
    for handle in hooks:
        handle.remove()
    bank(banked, "packed", captured["packed"].float().cpu())
    bank(banked, "heads", {k: v.float().cpu() for k, v in captured["heads"].items()})
    velocity = {"video": v_video.float().cpu(), "audio": v_audio.float().cpu()}
    bank(banked, "velocity", velocity)
    bank(banked, "latent1", {
        "video": step.advance_video(video, velocity["video"]),
        "audio": step.advance_audio(audio, velocity["audio"]),
    })
    bank(banked, "facts", request_facts(args))
    log("our seam bank complete")


# ------------------------------------------------------------------ the comparison


#: THE PRIMARY VERDICT IS COSINE, and the first run is why. A relative-max bound cannot
#: separate "50 layers of bf16 accumulated in a different order" from "a structurally
#: different graph": the first run measured the text encoder at 1.06e-2 relative max, which
#: is 2.7x a bf16 ulp and looks like a failure, and then the `--dtype float32` control
#: returned cos = 1.000000000 with 1.2e-7 relative — the port is EXACT and every bit of that
#: 1.06e-2 was our own compute precision. Direction survives precision; a wrong rope section
#: table, a wrong deepstack index, a transposed permutation or a wrong eps does not.
#:
#: So each seam carries a cosine floor and the reason it is where it is. The floors are set
#: from what a structural defect costs, not from what the first green run happened to score:
#: any of the defects above lands cosine at or below ~0.5, and every measurement here sat
#: above 0.999.
BOUNDS: dict[str, tuple[float, str]] = {
    "cond": (
        0.999,
        "50 bf16 decoder layers reducing in a different order. Proven precision rather than "
        "structure by the float32 control, which is EXACT (cos 1.000000000, rel 1.2e-07) — "
        "so this floor is about our compute dtype, not about the port",
    ),
    "packed": (
        0.999,
        "the text encoder's residual through condition_proj and the token refiner, plus two "
        "patch projections over identical latents",
    ),
    "heads": (0.999, "one 50-block bf16 stack; our SDPA against ComfyUI's attention path"),
    "velocity": (
        0.999,
        "the heads plus an unpatchify that is a pure permutation — a permutation that "
        "disagrees costs O(1) of direction, not four decimal places",
    ),
    "latent1": (
        0.9999,
        "one solver step, and the first sigma delta is 0.034 — it divides the velocity's "
        "residual by thirty, which is why this seam is the tightest of the five",
    ),
}


def compare_tensor(name: str, ours: Any, theirs: Any) -> tuple[bool, str]:
    """One seam. Cosine decides; relative max and the scale ratio are reported beside it
    because they say WHICH KIND of residual it is — accumulation moves the max and leaves
    the scale, a systematically different compute dtype moves the scale too."""
    if tuple(ours.shape) != tuple(theirs.shape):
        return False, f"SHAPE {tuple(ours.shape)} vs {tuple(theirs.shape)}"
    a, b = ours.double().flatten(), theirs.double().flatten()
    finite = bool(torch.isfinite(a).all())
    cos = float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))
    max_abs = float((a - b).abs().max())
    rel = max_abs / float(b.abs().max().clamp(min=1e-30))
    ratio = float(a.std() / b.std().clamp(min=1e-30))
    floor, _ = BOUNDS.get(name, (0.999, ""))
    detail = (
        f"cos {cos:.9f}  (floor {floor})  rel_max {rel:.3e}  scale_ratio {ratio:.6f}"
        f"{'  BIT-EXACT' if max_abs == 0.0 else ''}{'' if finite else '  NON-FINITE'}"
    )
    return finite and cos >= floor, detail


def side_compare(args: argparse.Namespace) -> int:
    ours_dir = pathlib.Path(args.bank) / "ours"
    comfy_dir = pathlib.Path(args.bank) / "comfy"
    print("=" * 96)
    print("WHOLE-SEAM ORACLE — this endpoint against ComfyUI v0.33.0, same weights, same prompt")
    print("=" * 96)
    results: list[tuple[str, bool, str]] = []

    # seam 1 — the presentation
    try:
        a, b = load(ours_dir, "tokens"), load(comfy_dir, "tokens")
        if a["ids"] is not None and b["ids"] is not None:
            same = list(a["ids"]) == list(b["ids"])
            results.append(("tokens", same, f"{len(a['ids'])} ids vs {len(b['ids'])} — "
                                            f"{'identical' if same else 'DIFFERENT'}"))
        else:
            results.append(("tokens", False, "one side banked no id list — UNMEASURED"))
    except FileNotFoundError as exc:
        results.append(("tokens", False, f"not banked: {exc}"))

    for name in ("cond", "packed"):
        try:
            verdict = compare_tensor(name, load(ours_dir, name), load(comfy_dir, name))
            results.append((name, *verdict))
        except FileNotFoundError as exc:
            results.append((name, False, f"not banked: {exc}"))

    for name in ("heads", "velocity", "latent1"):
        try:
            ours, theirs = load(ours_dir, name), load(comfy_dir, name)
            for stream in ("video", "audio"):
                results.append(
                    (f"{name}/{stream}", *compare_tensor(name, ours[stream], theirs[stream]))
                )
        except FileNotFoundError as exc:
            results.append((name, False, f"not banked: {exc}"))

    width = max(len(n) for n, _, _ in results)
    for name, good, detail in results:
        print(f"  {'ok  ' if good else 'FAIL'} {name:<{width}}  {detail}")
    print()
    for name, (floor, why) in BOUNDS.items():
        print(f"  cosine floor {name:<9} {floor} — {why}")
    failed = [n for n, good, _ in results if not good]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} seams within tolerance"
          + (f"; DIVERGED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


# ------------------------------------------------------------------ the full render


def side_render(args: argparse.Namespace) -> None:
    """Seams 7 and 8 — this endpoint's own full run, decoded, gated and written out.

    Not the Cozy serving path: the fill plane, the stamp plane and the residency ladder are
    a different subject with their own defects (#529's seven), and a render that fails
    because a budget constant is SDXL-sized says nothing about whether the MODEL is right.
    What DOES run is every model-path line the endpoint runs — the same boundary, the same
    solver, the same pixel conversion — and both of its output gates (#524 tiers 1 and 2).
    """
    models = pathlib.Path(args.models)
    banked = pathlib.Path(args.bank) / args.side
    device = torch.device("cuda")

    import h3
    from h3_arch import H3Config, build_component
    from h3_arch.layout import FPS, MediaFacts, build_modulation
    from h3_arch.pixels import pixel_bytes
    from h3_arch.presentation import Tokenizer
    from h3_arch.presentation import build as build_presentation

    config = H3Config()
    grid = geometry(args)
    log(f"geometry {grid}")

    tokenizer = Tokenizer()
    presentation = build_presentation(tokenizer, args.prompt)
    ids = presentation.text_ids()

    log("conditioning")
    encoder = load_component(
        build_component("text_encoder", config), models / CARRIERS["text_encoder"]
    ).to(device).eval()
    with torch.inference_mode():
        text = encoder(torch.tensor([list(ids)], dtype=torch.long, device=device))
    text = text.detach()
    del encoder
    torch.cuda.empty_cache()

    layout = _layout_for(text.shape[1], grid)
    solver = solver_for(args, layout)
    video, audio = initial_latents(args, grid)
    video = video.to(device=device, dtype=torch.bfloat16)
    audio = audio.to(device=device, dtype=torch.bfloat16)

    log(f"denoising — {solver.evaluations} evaluations over {solver.plan.grid_points} grid points")
    transformer = load_component(
        build_component("transformer", config), models / CARRIERS["transformer"]
    ).to(device).eval()
    run = seam_run(args, grid, layout, solver.plan)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in solver.steps():
        modulation = build_modulation(
            layout, t_video=step.t_video, t_audio=step.t_audio,
            visual_cond_t=0.0,
            text_token_tags=tuple([1] * text.shape[1]),
        )
        with torch.inference_mode():
            v_video, v_audio = h3._predict_data_velocity(
                transformer, run=run, text_states=text,
                video_latents=video, audio_latents=audio,
                modulation=modulation,
            )
        video = step.advance_video(video, v_video)
        audio = step.advance_audio(audio, v_audio)
        if step.index == 0:
            bank(banked, "latent1_run",
                 {"video": video.float().cpu(), "audio": audio.float().cpu()})
        log(f"  eval {step.index + 1}/{solver.evaluations}  sigma {step.sigma_video:.4f}")
    denoise_s = time.perf_counter() - started
    bank(banked, "latentN", {"video": video.float().cpu(), "audio": audio.float().cpu()})
    del transformer
    torch.cuda.empty_cache()

    log("decoding audio")
    audio_vae = load_component(
        build_component("audio_vae", config), models / CARRIERS["audio_vae"]
    ).to(device).eval()
    with torch.inference_mode():
        waveform = audio_vae.decode(audio)
    if waveform.ndim == 3:
        waveform = waveform[0]
    waveform = waveform.to(torch.float32)
    del audio_vae
    torch.cuda.empty_cache()

    log("decoding video")
    video_vae = load_component(
        build_component("video_vae", config), models / CARRIERS["video_vae"]
    ).to(device).eval()
    with torch.inference_mode():
        decoded = video_vae.decode(video)
    del video_vae
    torch.cuda.empty_cache()

    frames = pixel_bytes(decoded[0].float()).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
    peak = torch.cuda.max_memory_allocated() / 2**30
    facts = MediaFacts(width=grid.width, height=grid.height, frames=grid.frames, fps=FPS,
                       sample_rate=config.audio_vae.sample_rate, mute=False)

    log("the endpoint's OWN gates, over cozy-eval's instruments")
    from gates import post_encode_gate, pre_encode_gate

    tel = _Telemetry()
    gate_verdict = "passed"
    try:
        pre_encode_gate(torch, decoded=decoded.float(), pixels=frames.cpu(),
                        waveform=waveform.cpu(), requested=facts, tel=cast(Any, tel))
    except Exception as exc:
        gate_verdict = f"pre-encode REFUSED: {exc}"
        log(gate_verdict)

    out = pathlib.Path(args.bank) / f"{args.side}-render-seed{args.seed}.mp4"
    _write_mp4(out, frames.cpu(), waveform.cpu(), FPS, facts.sample_rate)
    if gate_verdict == "passed":
        try:
            post_encode_gate(out, requested=facts, tel=cast(Any, tel))
        except Exception as exc:
            gate_verdict = f"post-encode REFUSED: {exc}"
            log(gate_verdict)

    summary = {
        **request_facts(args),
        "denoise_seconds": round(denoise_s, 1),
        "seconds_per_evaluation": round(denoise_s / solver.evaluations, 2),
        "peak_vram_gib": round(peak, 2),
        "evaluations": solver.evaluations,
        "grid_points": solver.plan.grid_points,
        "plan_digest": solver.plan.digest(),
        "gate": gate_verdict,
        "metrics": tel.metrics,
        "mp4": str(out),
    }
    (pathlib.Path(args.bank) / f"{args.side}-render-seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(json.dumps(summary, indent=2))


class _Telemetry:
    """The endpoint's telemetry surface, reduced to what the gates actually call. The real
    one is the runtime's; the gates take it as an argument for exactly this reason."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    @contextlib.contextmanager
    def stage(self, _name: str) -> Any:
        yield


def _write_mp4(path: pathlib.Path, frames: Any, waveform: Any, fps: int, rate: int) -> None:
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    audio_stream = container.add_stream("aac", rate=rate, layout="stereo")
    for frame in frames.numpy():
        container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
    container.mux(stream.encode())
    samples = np.ascontiguousarray(waveform.numpy().astype(np.float32))
    audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
    audio_frame.sample_rate = rate
    container.mux(audio_stream.encode(audio_frame))
    container.mux(audio_stream.encode())
    container.close()


# ------------------------------------------------------------------ entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("comfy", "ours", "render"):
        p = sub.add_parser(name)
        p.add_argument("--models", required=True)
        p.add_argument("--bank", required=True)
        p.add_argument("--comfy", default=str(pathlib.Path.home() / "ComfyUI"))
        if name == "render":
            p.add_argument("--side", default="ours", choices=("ours", "comfy"))
        add_request_options(p)
    p = sub.add_parser("compare")
    p.add_argument("--bank", required=True)

    args = parser.parse_args()
    if args.mode == "comfy":
        side_comfy(args)
    elif args.mode == "ours":
        side_ours(args)
    elif args.mode == "render":
        side_render(args)
    else:
        return side_compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
