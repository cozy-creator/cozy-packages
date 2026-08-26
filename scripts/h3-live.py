#!/usr/bin/env python
"""se-001's CONTROL-PLANE ARMS — every H3 refusal that can be decided with no GPU, no
weights and no rented card, fired for real and observed.

    nice -n 19 .venv/bin/python scripts/h3-live.py [--installed-wheel] [group ...]
    groups: request, plan, keys, fence, gates, presentation
    nice -n 19 .venv/bin/python scripts/h3-live.py gates presentation
        (gates needs torch + cozy-eval + ffmpeg; presentation needs transformers)

DETERMINISTIC CONTRACT ARMS LIVE IN `scripts/h3-conform.py` AND RUN IN CI (#533). This
driver holds endpoint request/plan refusals, source fences, and the output gates that need
real banked media and a decoder. Runtime component-guard/restamp arms live in Runtime's own
`scripts/redarm.py`; duplicating them here with endpoint doubles would cross that boundary.

This is a driver, not a test suite (tracker README #160): it fires arms and prints what it
saw. An arm that is not observed is not banked, and a green fence nobody has fired is a
claim rather than a fact — se-008 learned that the hard way when its NaN floor turned out
to be checking a tensor where NaN cannot exist.

WHAT THIS CANNOT DECIDE, said once here rather than implied by silence: nothing about
numerics, nothing about residency, nothing about a real generation. se-002 is where a card
enters.
"""

from __future__ import annotations

import pathlib
import sys
import traceback
from typing import Any

from _h3_probe import ROOT, select

INSTALLED_WHEEL, _SUBJECTS = select(sys.argv, "h3", "h3_arch")

import msgspec  # noqa: E402
import torch  # noqa: E402

import h3  # noqa: E402
from h3_arch import H3Config  # noqa: E402
from h3_arch.layout import PackedLayout, build_timestep_plan, latent_grid  # noqa: E402

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0


def observe(what: str, detail: str = "") -> None:
    print(f"{PASS}{what}" + (f"\n         {detail}" if detail else ""))


def failed(what: str, detail: str = "") -> None:
    global _failures
    _failures += 1
    print(f"{FAIL}{what}" + (f"\n         {detail}" if detail else ""))


def expect_refusal(what: str, fn: Any, *, code: str | None = None) -> None:
    """Fire an arm. A refusal that names the right code is the observation; anything else
    — including success — is the failure."""
    try:
        fn()
    except Exception as exc:  # the arm's whole subject is which exception arrives
        got = getattr(exc, "code", None) or type(exc).__name__
        if code is not None and got != code:
            failed(what, f"refused {got!r}, expected {code!r}: {exc}")
            return
        first = str(exc).splitlines()[0]
        first = first[len(got) + 2 :] if first.startswith(f"{got}: ") else first
        observe(what, f"{got}: {first[:140]}")
        return
    failed(what, "the call SUCCEEDED — the arm did not fire")


# --------------------------------------------------------------- the arms


def group_request() -> None:
    print("\n== the request plane — layer 1, the schema the runtime validates against ==")
    from cozy_runtime.author import ImageAsset

    base = {"prompt": "a cat"}

    def hook(kind: type, value: Any) -> Any:
        """An asset arrives on the wire as a handle the SUPERVISOR resolves; plain msgspec
        has no such hook, so this driver supplies the same shape. Everything else below is
        the endpoint's own schema, unmodified."""
        if kind is ImageAsset or getattr(kind, "kind", None):
            return kind(str(value))
        raise TypeError(kind)

    def decode(kind: Any, payload: dict[str, Any]) -> Any:
        return msgspec.json.decode(msgspec.json.encode(payload), type=kind, dec_hook=hook)

    ok = decode(h3.GenerateInput, base)
    observe(
        "a bare prompt resolves every default",
        f"{ok.aspect_ratio.value} / {ok.duration_s}s / {ok.num_inference_steps} steps",
    )

    expect_refusal(
        "an unoffered aspect ratio (5:4) — no nearest-bucket rounding to fall into",
        lambda: decode(h3.GenerateInput, {**base, "aspect_ratio": "5:4"}),
    )
    expect_refusal(
        "a duration outside the preset set (30 s)",
        lambda: decode(h3.GenerateInput, {**base, "duration_s": 30}),
    )
    expect_refusal(
        "a step count that is not a published preset (25)",
        lambda: decode(h3.GenerateInput, {**base, "num_inference_steps": 25}),
    )
    expect_refusal(
        "an undeclared field (guidance) — this family HAS no guidance",
        lambda: decode(h3.GenerateInput, {**base, "guidance": 7.5}),
    )
    expect_refusal(
        "a negative_prompt — same, and the reason is the checkpoint's distillation",
        lambda: decode(h3.GenerateInput, {**base, "negative_prompt": "blurry"}),
    )

    expect_refusal(
        "references are absent from this release rather than hidden behind a flag",
        lambda: decode(h3.GenerateInput, {**base, "references": []}),
    )


def group_plan() -> None:
    print("\n== TimestepPlan identity — coverage is the DIGEST, never the step count ==")
    grid = latent_grid(124, 1344, 768)
    layout = PackedLayout(64, grid)
    observe(
        "the packed sequence for 124 frames at 1344x768",
        f"{layout.seq_len} rows = 64 text + {grid.audio_rows} audio + {grid.video_rows} video",
    )

    def plan(**over: Any) -> Any:
        args: dict[str, Any] = {
            "task": "fl2va",
            "structure": "h3-adaln-curve",
            "evaluations": 30,
            "layout": layout,
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "visual_cond_timestep": None,
        }
        args.update(over)
        return build_timestep_plan(**args)

    a, b = plan(), plan()
    if a.digest() == b.digest():
        observe("the same request twice is the same digest", a.digest())
    else:
        failed("plan digest is not stable", f"{a.digest()} vs {b.digest()}")

    cases = [
        ("the SAME 30 evaluations at a different video shift", plan(sigma_shift_video=11.0)),
        ("the same 30 evaluations at a different AUDIO shift", plan(sigma_shift_audio=4.0)),
        ("the same coverage stamped for the unserved Ref2VA task", plan(task="ref2va")),
        ("the same 30 with a visual condition present", plan(visual_cond_timestep=0.999)),
        ("the same 30 with an AdaLN-targeting adapter", plan(adapters=("turbo-4step",))),
    ]
    for what, other in cases:
        if other.digest() != a.digest():
            observe(f"{what} is a DIFFERENT plan", other.digest())
        else:
            failed(f"{what} collided with the base plan", a.digest())
    if (a.evaluations, a.grid_points) == (30, 31):
        observe(
            "the plan names EVALUATIONS and GRID POINTS apart",
            f"{a.evaluations} model evaluations over {a.grid_points} sigma values — "
            "`steps` is gone, because it never said which of the two it meant",
        )
    else:
        failed("evaluations vs grid points", f"{a.evaluations} / {a.grid_points}")


#: THE GATE FIXTURES, pinned by CONTENT DIGEST and not by path. Both live in the eval bank
#: rather than this repo — they are 2 MiB and 13 MiB of real media — so the pin is what
#: makes "the arm ran against the clip it says it ran against" checkable.
#:
#: The POSITIVE fixture is a real ComfyUI H3 render the owner accepted as good (job-003's
#: bf16 floor arm), and it is exactly the 5 s preset: 1344x768, 24 fps, 124 frames, 32 kHz
#: stereo. It REPLACES `torch.rand` — the previous positive fixture was random noise, which
#: is the statistical class of the failure the gate exists to catch, so the gate was green
#: for its own subject.
#:
#: The NEGATIVE fixture is `daf50552…` itself: se-002's first render, banked as permanent
#: evidence (#519). It is a red arm forever. If it ever passes, the gate has regressed.
_BANK = pathlib.Path.home() / "cozy_v2"
GOOD_CLIP = (
    _BANK / "tensorfs-bench/job-003-floor/outputs/B-comfy-full-bf16_r1_00001_.mp4",
    "bb83def835cc41497afff168243a059e122c44418b503727438edc8633725f7e",
)
FAILED_CLIP = (
    _BANK / "tensorfs-bench/proto-001/outputs/degraded/"
    "RED-h3-first-serve-reversed-sampler-daf50552.mp4",
    "daf50552d2b4ff79d9948e0c8167c2b1a68c20799530d49025d250fb1c0cee42",
)


class Tel:
    """The one `Telemetry` member the gates use. A structural stand-in, not a mock
    framework: their contract with telemetry is `metric(name, value)`."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def metric(self, name: str, value: float) -> None:
        self.metrics[name] = value


def tel_of(t: Tel) -> Any:
    """The stand-in, typed as the surface it stands in for. `Telemetry` is a bound runtime
    service an endpoint never constructs, so a driver cannot hand the real one over."""
    return t


def _pinned(fixture: tuple[pathlib.Path, str]) -> pathlib.Path | None:
    """The fixture, or a FAILURE naming what is missing. Never a silent skip: a gate arm
    that quietly does not run is the shape se-002 already paid for."""
    import hashlib

    path, digest = fixture
    if not path.is_file():
        failed(f"fixture missing: {path.name}", f"expected at {path}")
        return None
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        failed(f"fixture {path.name} is not the pinned bytes", f"{got[:16]}… != {digest[:16]}…")
        return None
    observe(f"fixture {path.name} matches its pin", f"sha256 {digest[:16]}…")
    return path


def group_gates() -> None:
    """THE TWO PER-REQUEST GATES, fired against digest-pinned REAL media.

    Needs `.venv` (torch, cozy-eval) and ffmpeg, not `.venv-check`."""
    print("\n== the per-request output gates — real clips, cozy-eval's own floors ==")
    import numpy as np

    from gates import post_encode_gate, pre_encode_gate
    from h3_arch.layout import MediaFacts

    good = _pinned(GOOD_CLIP)
    bad = _pinned(FAILED_CLIP)
    if good is None or bad is None:
        return

    import cozy_eval.audio as ce_audio
    import cozy_eval.frames as ce_frames
    import cozy_eval.integrity as ce_integrity

    def tensors(path: pathlib.Path) -> tuple[Any, Any, Any]:
        clip = np.stack(list(ce_frames.iter_video(path)))          # (T, H, W, 3) in [0,1]
        audio = ce_audio.read_audio(path)
        return (
            torch.from_numpy(clip),                                 # the FLOAT decode
            torch.from_numpy((clip * 255).round().astype("uint8")),  # the encoder's pixels
            torch.from_numpy(audio.samples.T.copy()),               # (channels, samples)
        )

    def facts(frames: int) -> Any:
        return MediaFacts(width=1344, height=768, frames=frames, fps=24, sample_rate=32000)

    # ---- the POSITIVE fixture: a real, owner-accepted H3 render
    decoded, pixels, waveform = tensors(good)
    tel = Tel()
    pre_encode_gate(
        torch, decoded=decoded, pixels=pixels, waveform=waveform,
        requested=facts(124), tel=tel_of(tel),
    )
    observe(
        "a REAL good H3 render passes the pre-encode gate",
        f"adjacent_frame_corr {tel.metrics['adjacent_frame_corr']}, "
        f"frame_std_min {tel.metrics['frame_std_min']}, "
        f"true peak {tel.metrics['audio_true_peak_dbtp']} dBTP",
    )
    tel = Tel()
    post_encode_gate(good, requested=facts(124), tel=tel_of(tel))
    observe(
        "and the post-encode gate, decoding the actual container",
        f"{int(tel.metrics['encoded_frames'])} frames at "
        f"{int(tel.metrics['encoded_width'])}x{int(tel.metrics['encoded_height'])} / "
        f"{tel.metrics['encoded_fps']} fps, {tel.metrics['encoded_audio_seconds']} s of audio",
    )

    # ---- THE PERMANENT RED ARM: se-002's first render, by digest
    red_decoded, red_pixels, red_waveform = tensors(bad)
    # THE HEADLINE NUMBER, read straight off cozy-eval so the arm reports what the floor
    # actually measured rather than only which refusal arrived first.
    verdict = ce_integrity.output_integrity(red_pixels.numpy())
    if not verdict.ok:
        observe(
            "daf50552… fails cozy-eval's integrity floor — the check that already existed",
            f"{verdict.summary()} — against a floor of 0.6, and the endpoint never called it",
        )
    else:
        failed("the failed render passed the integrity floor", verdict.summary())
    expect_refusal(
        "daf50552… — the first se-002 render — is REFUSED at the pre-encode gate",
        lambda: pre_encode_gate(
            torch,
            decoded=red_decoded,
            pixels=red_pixels,
            waveform=red_waveform,
            requested=facts(103),
            tel=tel_of(Tel()),
        ),
        code="output_av_duration_mismatch",
    )
    expect_refusal(
        "  and against what was actually REQUESTED, on its frame count alone",
        lambda: pre_encode_gate(
            torch,
            decoded=red_decoded,
            pixels=red_pixels,
            waveform=red_waveform,
            requested=facts(124),
            tel=tel_of(Tel()),
        ),
        code="output_shape_mismatch",
    )
    expect_refusal(
        "  and at the post-encode gate, from the container alone",
        lambda: post_encode_gate(bad, requested=facts(124), tel=tel_of(Tel())),
        code="output_container_video",
    )
    tel = Tel()
    try:
        post_encode_gate(bad, requested=facts(103), tel=tel_of(tel))
        failed("its own streams should still disagree", str(tel.metrics))
    except Exception as exc:
        observe(
            "  and even taken on its OWN terms, its two streams disagree",
            f"{getattr(exc, 'code', '?')}: {str(exc).splitlines()[0][:120]}",
        )

    # ---- the branches the floor exists for, over the REAL clip rather than noise
    nan_decoded = decoded.clone()
    nan_decoded[0, 0, 0, 0] = float("nan")
    expect_refusal(
        "NaN read off the FLOAT decode, before quantization (#411's lesson)",
        lambda: pre_encode_gate(
            torch, decoded=nan_decoded, pixels=pixels, waveform=waveform,
            requested=facts(124), tel=tel_of(Tel()),
        ),
        code="output_integrity_nan",
    )
    if not bool(torch.isnan(pixels.to(torch.float32)).any()):
        observe(
            "the same generation's uint8 pixels carry NO evidence of it",
            "quantizing to uint8 is exactly what erases a diverged decode",
        )
    black = torch.zeros_like(pixels)
    expect_refusal(
        "a black clip — what a wrong VAE scaling produces",
        lambda: pre_encode_gate(
            torch, decoded=black.to(torch.float32), pixels=black, waveform=waveform,
            requested=facts(124), tel=tel_of(Tel()),
        ),
        code="output_integrity_video",
    )
    expect_refusal(
        "A SILENT SOUNDTRACK over a perfectly good picture — H3's own third branch",
        lambda: pre_encode_gate(
            torch, decoded=decoded, pixels=pixels, waveform=torch.zeros_like(waveform),
            requested=facts(124), tel=tel_of(Tel()),
        ),
        code="output_integrity_audio",
    )
    expect_refusal(
        "a soundtrack that is 0.9 s longer than the picture",
        lambda: pre_encode_gate(
            torch, decoded=decoded, pixels=pixels,
            waveform=torch.cat([waveform, waveform[:, :29000]], dim=1),
            requested=facts(124), tel=tel_of(Tel()),
        ),
        code="output_av_duration_mismatch",
    )


def group_keys() -> None:
    print("\n== the key-exactness harness — and its own red arm ==")
    import subprocess

    command = [sys.executable, str(ROOT / "scripts" / "h3-keys.py")]
    if INSTALLED_WHEEL:
        command.append("--installed-wheel")
    result = subprocess.run(command, capture_output=True, text=True)
    tail = result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr
    expected = "4/4 components key-exact over 2913 destinations"
    if result.returncode == 0 and tail == expected:
        observe("all four components key-exact against the pinned header", tail)
    else:
        failed("key-exactness", f"{tail!r}, expected {expected!r}")

    # THE ARM: a graph that is one block short is a serve that would fail on a rented
    # card. Perturbing the config is the cheapest way to prove the harness can say so.
    import dataclasses
    import importlib.util

    import h3_arch

    curve = H3Config.from_mapping({"dit": {"structure": "h3-adaln-curve"}})
    if (curve.dit.time_embed_dim, curve.dit.adaln_curve_grid) == (8, 1025):
        observe("the artifact config admits the one bound native-curve structure")
    else:
        failed("native-curve structure config", repr(curve.dit))
    for structure in ("h3-full", "h3-adaln-baked", "not-an-h3-structure"):
        expect_refusal(
            f"the unbound structure {structure!r} at the artifact boundary",
            lambda structure=structure: H3Config.from_mapping(
                {"dit": {"structure": structure}}
            ),
        )
    expect_refusal(
        "artifact config cannot activate the deleted official graph",
        lambda: H3Config.from_mapping({"graph": "h3-diffusers"}),
    )
    expect_refusal(
        "a float cannot impersonate the curve grid's integer dimension",
        lambda: H3Config.from_mapping({"dit": {"adaln_curve_grid": 1025.0}}),
    )
    expect_refusal(
        "a boolean cannot impersonate the curve graph's integer dimension",
        lambda: H3Config.from_mapping({"dit": {"adaln_curve_grid": False}}),
    )
    expect_refusal(
        "an integer value mismatch cannot change the fixed curve graph",
        lambda: H3Config.from_mapping({"dit": {"adaln_curve_grid": 1024}}),
    )
    expect_refusal(
        "a malformed known artifact section cannot silently select release defaults",
        lambda: H3Config.from_mapping({"video_vae": "broken"}),
    )

    with torch.device("meta"):
        short = h3_arch.build_dit(dataclasses.replace(curve.dit, num_layers=49))
    keys = set(short.state_dict())

    spec = importlib.util.spec_from_file_location("h3keys", ROOT / "scripts" / "h3-keys.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    want = set(module.pinned("transformer"))

    missing = want - keys
    if len(missing) == 10 and not keys - want:
        observe(
            "a 49-layer graph against the 50-layer artifact FAILS, naming what is absent",
            f"{len(missing)} destinations missing, all on block 49: "
            f"{sorted(missing)[0]}",
        )
    else:
        failed("the short-graph arm", f"{len(missing)} missing / {len(keys - want)} extra")

def group_fence() -> None:
    """PLANT -> OBSERVE -> REMOVE, on the real file through the real fence. A fence nobody
    has fired is a claim: se-008's NaN floor was green for three days while checking a
    tensor in which NaN cannot exist."""
    print("\n== the structural fences — every rule fired against a planted violation ==")
    import subprocess

    fence = ROOT / "scripts" / "fence.py"

    def run() -> tuple[int, str]:
        result = subprocess.run(
            [sys.executable, str(fence)], capture_output=True, text=True
        )
        return result.returncode, result.stdout

    code, out = run()
    if code == 0:
        observe("the unplanted tree is green", f"{out.count('fence green')} fences")
    else:
        failed("the unplanted tree is not green", out)
        return

    plants = [
        (
            ROOT / "h3" / "h3.py",
            "import hashlib\n",
            "import hashlib\nimport torch\n",
            "light-module-scope",
            "a heavy import at the ENDPOINT module's own scope",
        ),
        (
            ROOT / "h3" / "h3_arch" / "layout.py",
            "import hashlib\n",
            "import hashlib\nfrom .dit import patchify_video\n",
            "light-module-scope",
            "an INDIRECT heavy import — a light module importing the model library",
        ),
        (
            ROOT / "h3" / "h3_arch" / "dit.py",
            "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)",
            "    q = q.pin_memory()\n"
            "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)",
            "no-memory-choreography",
            "host pinning inside the vendored architecture",
        ),
        (
            ROOT / "h3" / "h3_arch" / "dit.py",
            "        t_emb = self.timestep_embedding(t_values)",
            "        torch.cuda.empty_cache()\n"
            "        t_emb = self.timestep_embedding(t_values)",
            "no-memory-choreography",
            "an allocator command in the denoise body",
        ),
        (
            ROOT / "h3" / "h3.py",
            "        self.tokenizer = Tokenizer()",
            "        self.tokenizer = Tokenizer()\n"
            "        _ = torch.load('/tmp/x')",
            "no-memory-choreography",
            "a checkpoint read inside load()",
        ),
        (
            ROOT / "h3" / "h3.py",
            "    prompt: str\n    first_frame",
            "    prompt: str\n    _pin = 'cozy/minimax-h3@se-001'\n    first_frame",
            "no-identifiers-in-code",
            "a pinned release ref spelled in endpoint code",
        ),
    ]

    for path, needle, replacement, rule, what in plants:
        original = path.read_text()
        if needle not in original:
            failed(f"could not plant: {what}", f"{path.name} has no anchor")
            continue
        try:
            path.write_text(original.replace(needle, replacement, 1))
            code, out = run()
            red = [line for line in out.splitlines() if line.startswith("FENCE RED")]
            hit = any(rule in line for line in red)
            if code != 0 and hit:
                detail = next(
                    (line.strip() for line in out.splitlines() if line.strip().startswith("h3/")),
                    "",
                )
                observe(f"{rule}: {what}", detail[:150])
            else:
                failed(f"{rule}: {what}", f"exit {code}, red rules {red}")
        finally:
            path.write_text(original)

    code, out = run()
    if code == 0:
        observe("every plant removed; the tree is green again", f"{out.count('green')} fences")
    else:
        failed("THE TREE DID NOT RETURN TO GREEN", out)


def group_presentation() -> None:
    """THE CONDITIONING SEAM'S FIRST HALF — what the prompt actually becomes.

    NEEDS `.venv` (transformers). This group exists because of a defect that shipped and
    reported NOTHING: `Qwen2Tokenizer` renamed `vocab_file`/`merges_file` to
    `vocab`/`merges` in transformers 5, the old spelling is silently absorbed by `**kwargs`,
    and what constructs is a two-piece tokenizer that encodes every prompt to the empty
    list. `build`'s empty-rows fallback then made that ONE PAD TOKEN, so every generation
    conditioned on no prompt at all — through a green key census, green conformance arms and
    a green fence. Nothing in this repo asserted that the bundled vocabulary had loaded.
    """
    from h3_arch.presentation import Tokenizer
    from h3_arch.presentation import build as build_presentation

    print("\nPRESENTATION — the bundled vocabulary, and the refusals that keep it honest")
    tokenizer = Tokenizer()
    prompt = "a red sports car driving fast along a coastal road at sunset"
    ids = tokenizer.ids(prompt)
    if len(ids) < 8:
        failed("the bundled vocabulary encodes a real prompt",
               f"{len(ids)} ids for {len(prompt)} characters — the vocabulary did not load")
    else:
        observe("the bundled vocabulary encodes a real prompt",
                f"{len(prompt)} characters -> {len(ids)} ids, {len(tokenizer._tok)} pieces")
    back = tokenizer._tok.decode(ids)
    if back == prompt:
        observe("  and the ids ROUND-TRIP to the same string", repr(back))
    else:
        failed("  and the ids ROUND-TRIP to the same string", f"{back!r} != {prompt!r}")

    presentation = build_presentation(tokenizer, prompt)
    if presentation.text_ids() == tuple(ids) and set(presentation.tags) == {1}:
        observe("  and the presentation carries them all, every row tagged TEXT",
                f"{len(presentation.rows)} rows")
    else:
        failed("  and the presentation carries them all, every row tagged TEXT",
               f"{presentation.text_ids()[:8]} vs {ids[:8]}")

    empty = build_presentation(tokenizer, "")
    if empty.text_ids() == (151643,):
        observe("an EMPTY request still resolves to one pad token — that path is legitimate")
    else:
        failed("an EMPTY request still resolves to one pad token", str(empty.text_ids()))

    class _Mute:
        """The exact defect, as a red control: a tokenizer that encodes nothing."""

        def ids(self, _text: str) -> list[int]:
            return []

    expect_refusal(
        "  but a NON-EMPTY prompt that produces no tokens REFUSES — it is a broken "
        "tokenizer, not an empty request",
        lambda: build_presentation(_Mute(), prompt),
        code="RuntimeError",
    )


GROUPS = {
    "request": group_request,
    "presentation": group_presentation,
    "plan": group_plan,
    "gates": group_gates,
    "keys": group_keys,
    "fence": group_fence,
}


def main() -> int:
    names = sys.argv[1:] or [name for name in GROUPS if not (INSTALLED_WHEEL and name == "fence")]
    if INSTALLED_WHEEL and "fence" in names:
        print(
            "REFUSED: h3-live's fence group plants the repository source and is source-only",
            file=sys.stderr,
        )
        return 2
    for name in names:
        if name not in GROUPS:
            print(f"unknown group {name!r}: {', '.join(GROUPS)}", file=sys.stderr)
            return 2
        try:
            GROUPS[name]()
        except Exception:
            failed(f"group {name} raised")
            traceback.print_exc()
    print(f"\n{_failures} failed")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
