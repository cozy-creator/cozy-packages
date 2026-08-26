#!/usr/bin/env python
"""se-001's CONTROL-PLANE ARMS — every H3 refusal that can be decided with no GPU, no
weights and no rented card, fired for real and observed.

    nice -n 19 .venv-check/bin/python scripts/h3-live.py [group ...]
    groups: components, request, plan, integrity, keys

This is a driver, not a test suite (tracker README #160): it fires arms and prints what it
saw. An arm that is not observed is not banked, and a green fence nobody has fired is a
claim rather than a fact — se-008 learned that the hard way when its NaN floor turned out
to be checking a tensor where NaN cannot exist.

WHAT THIS CANNOT DECIDE, said once here rather than implied by silence: nothing about
numerics, nothing about residency, nothing about a real generation. The component-use arms
use the runtime's OWN `GuardedComponent` over author-built doubles, which is the real
guard and a fake pipeline. se-002 is where a card enters.
"""

from __future__ import annotations

import pathlib
import sys
import traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "h3"))

import msgspec  # noqa: E402
import torch  # noqa: E402
from cozy_runtime.author import uses_components  # noqa: E402

# The runtime's real component guard. Not on the author surface on purpose — an endpoint
# never installs one — but this driver is standing in for the runtime, so simulating it
# with a lookalike would be verifying the lookalike.
from cozy_runtime.author._model import GuardedComponent  # noqa: E402

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


# --------------------------------------------------------------- author doubles


class FakeComponent:
    """Stands in for a component root. It owns one attribute, which is all the guard
    needs: any access at all is what gets checked."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.name


class FakePipe:
    def __init__(self, transformer_role: str) -> None:
        self.transformer_role = transformer_role
        self.config = H3Config()
        self.components: dict[str, Any] = {}


def build_double(cls: Any, transformer_role: str) -> Any:
    """One role class over a fake pipeline, with the RUNTIME'S guard installed over every
    component exactly as the runtime installs it after `load`."""
    pipe = FakePipe(transformer_role)
    model = cls.for_test(pipe=pipe, tokenizer=None)
    for name in (transformer_role, "text_encoder", "video_vae", "audio_vae"):
        pipe.components[name] = GuardedComponent(name, FakeComponent(name), model)
    # the twin's transformer, present in the DUAL artifact and not in this slot's set —
    # this is the object the wrong-transformer arm reaches for
    twin = "transformer_ref" if transformer_role == "transformer" else "transformer"
    pipe.components[twin] = GuardedComponent(twin, FakeComponent(twin), model)
    return model, pipe


# --------------------------------------------------------------- the arms


def group_components() -> None:
    print("\n== component-use contract — the runtime's own guard over author doubles ==")

    # GREEN FIRST: the declared access works, or the red arms below prove nothing.
    class _Reach(h3.Fl2VAModel):
        @uses_components("transformer")
        def reach(self, which: str) -> Any:
            return self.pipe.components[which].name

    green, _ = build_double(_Reach, "transformer")
    if green.reach("transformer") == "transformer":
        observe("declared access inside its own scope", "Fl2VAModel scope -> transformer.name")
    else:
        failed("declared access inside its own scope")

    expect_refusal(
        "Fl2VAModel.denoise touching the REF transformer",
        lambda: green.reach("transformer_ref"),
        code="undeclared_component",
    )
    expect_refusal(
        "and the instance is POISONED afterwards — reuse refuses",
        lambda: green.reach("transformer"),
        code="poisoned_generation",
    )

    class _RefReach(h3.Ref2VAModel):
        @uses_components("transformer_ref")
        def reach(self, which: str) -> Any:
            return self.pipe.components[which].name

    rg, _ = build_double(_RefReach, "transformer_ref")
    expect_refusal(
        "Ref2VAModel.denoise touching the FL transformer",
        lambda: rg.reach("transformer"),
        code="undeclared_component",
    )

    class _CrossReach(h3.Fl2VAModel):
        @uses_components("text_encoder")
        def reach(self, which: str) -> Any:
            return self.pipe.components[which].name

    cross, _ = build_double(_CrossReach, "transformer")
    expect_refusal(
        "condition_text touching video_vae (a declared component, the wrong one)",
        lambda: cross.reach("video_vae"),
        code="undeclared_component",
    )

    _, outside_pipe = build_double(h3.Fl2VAModel, "transformer")
    expect_refusal(
        "any component touched OUTSIDE every scope",
        lambda: outside_pipe.components["transformer"].name,
        code="undeclared_component",
    )

    class _Nested(h3.Fl2VAModel):
        @uses_components("text_encoder")
        def outer(self) -> Any:
            return self.inner()

        @uses_components("transformer")
        def inner(self) -> Any:
            return self.pipe.components["transformer"].name

    nested, _ = build_double(_Nested, "transformer")
    expect_refusal(
        "a second scope entered while one is active",
        nested.outer,
        code="concurrent_scope",
    )

    def _empty_set() -> None:
        class _Bad(h3.Fl2VAModel):
            @uses_components()
            def nothing(self) -> None:
                return None

    expect_refusal(
        "@uses_components() empty — a component-free method is module code",
        _empty_set,
        code="empty_component_set",
    )

    # The stamps are what tell the two slots apart, since NOTHING STRUCTURAL DOES: the two
    # carriers have byte-identical headers and the dual tuple planned three ways returns
    # one topology digest (job-001).
    stamps = (h3.Fl2VAModel.__stamps__, h3.Ref2VAModel.__stamps__)
    if stamps == ({"task": "fl2va"}, {"task": "ref2va"}):
        observe("the role classes carry distinct task stamps", f"{stamps[0]} / {stamps[1]}")
    else:
        failed("task stamps", str(stamps))

    def _restamp() -> None:
        class _Bad(h3.Fl2VAModel, task="ref2va"):
            pass

    expect_refusal(
        "a role class RE-stamping its base's task",
        _restamp,
        code="restamped",
    )


def group_request() -> None:
    print("\n== the request plane — layer 1, the schema the runtime validates against ==")
    from cozy_runtime.author import AudioAsset, ImageAsset

    base = {"prompt": "a cat"}

    def hook(kind: type, value: Any) -> Any:
        """An asset arrives on the wire as a handle the SUPERVISOR resolves; plain msgspec
        has no such hook, so this driver supplies the same shape. Everything else below is
        the endpoint's own schema, unmodified."""
        if kind in (ImageAsset, AudioAsset) or getattr(kind, "kind", None):
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

    def images(n: int) -> dict[str, Any]:
        refs = [{"kind": "image", "image": f"cozy://asset/{i}"} for i in range(n)]
        return {**base, "references": refs}

    def audios(n: int) -> dict[str, Any]:
        refs = [{"kind": "audio", "audio": f"cozy://a/{i}"} for i in range(n)]
        return {**base, "references": refs}

    nine = decode(h3.RefGenerateInput, images(9))
    h3.decode_references(nine)
    observe("9 image references — the published cap, exactly", "accepted")

    expect_refusal(
        "zero references on the reference route",
        lambda: decode(h3.RefGenerateInput, {**base, "references": []}),
    )
    expect_refusal(
        "13 references, over the TOTAL cap of 12 — refused by the schema",
        lambda: decode(h3.RefGenerateInput, images(13)),
    )
    # THE PER-TYPE CAPS ARE THE ENDPOINT'S, not the schema's: 10 is under the total cap of
    # 12 and over the image cap of 9, so only `decode_references` can catch it.
    ten = decode(h3.RefGenerateInput, images(10))
    expect_refusal(
        "10 image references — under the total cap, over the per-type cap of 9",
        lambda: h3.decode_references(ten),
        code="reference_cap",
    )
    four_audio = decode(h3.RefGenerateInput, audios(4))
    expect_refusal(
        "4 audio references — over the per-type cap of 3",
        lambda: h3.decode_references(four_audio),
        code="reference_cap",
    )
    # AUDIO MAY STAND ALONE. Upstream refuses audio as the only modality; that guard was
    # measured to protect nothing (v1), and the Cozy extension carries it WITHOUT claiming
    # viseme or beat synchronization.
    solo = decode(h3.RefGenerateInput, audios(1))
    presented = h3.decode_references(solo)
    if len(presented) == 1 and presented[0].kind == "audio":
        observe("an audio reference standing alone is a legal request", "the Cozy extension")
    else:
        failed("audio-only reference", str(presented))


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
            "steps": 30,
            "layout": layout,
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "visual_cond_timestep": None,
            "audio_cond_timestep": None,
        }
        args.update(over)
        return build_timestep_plan(**args)

    a, b = plan(), plan()
    if a.digest() == b.digest():
        observe("the same request twice is the same digest", a.digest())
    else:
        failed("plan digest is not stable", f"{a.digest()} vs {b.digest()}")

    cases = [
        ("the SAME 30 steps at a different video shift", plan(sigma_shift_video=11.0)),
        ("the same 30 steps at a different AUDIO shift", plan(sigma_shift_audio=4.0)),
        ("the same 30 steps on the other task partition", plan(task="ref2va")),
        ("the same 30 steps against a FULL-AdaLN structure", plan(structure="h3-full")),
        ("the same 30 steps with a visual condition present", plan(visual_cond_timestep=0.999)),
        ("the same 30 steps with an AdaLN-targeting adapter", plan(adapters=("turbo-4step",))),
    ]
    for what, other in cases:
        if other.digest() != a.digest():
            observe(f"{what} is a DIFFERENT plan", other.digest())
        else:
            failed(f"{what} collided with the base plan", a.digest())
    if a.steps == 30 and plan(steps=30).steps == 30:
        observe("`steps` is display metadata derived from the value list", f"{a.steps}")


def group_integrity() -> None:
    print("\n== the output-integrity floor — all three branches, on real tensors ==")

    class Tel:
        """The one `Telemetry` member the floor uses. A structural stand-in, not a mock
        framework: the floor's contract with telemetry is `metric(name, value)`."""

        def __init__(self) -> None:
            self.metrics: dict[str, float] = {}

        def metric(self, name: str, value: float) -> None:
            self.metrics[name] = value

    def tel_of(t: Tel) -> Any:
        return t

    good_video = torch.rand(3, 8, 64, 64)
    good_pixels = (good_video * 255).to(torch.uint8)
    good_audio = torch.randn(2, 32000) * 0.1

    tel = Tel()
    h3._integrity(torch, good_video, good_pixels, good_audio, tel_of(tel))
    observe("a real-shaped decode passes", str(tel.metrics))

    nan_video = good_video.clone()
    nan_video[0, 0, 0, 0] = float("nan")
    expect_refusal(
        "NaN read off the FLOAT decode, before quantization (#411's lesson)",
        lambda: h3._integrity(torch, nan_video, good_pixels, good_audio, tel_of(Tel())),
        code="output_integrity_nan",
    )
    # THE POINT of reading the float tensor: the quantized pixels of that same decode are
    # clean, because a uint8 cannot be NaN.
    quantized = torch.nan_to_num(nan_video, 0.0).mul(255).to(torch.uint8)
    if not bool(torch.isnan(quantized.to(torch.float32)).any()):
        observe(
            "the same generation's uint8 pixels carry NO evidence of it",
            "clamp(0,1).to(uint8) is exactly what erases a diverged decode",
        )

    flat = torch.full((3, 8, 64, 64), 0.5)
    expect_refusal(
        "a flat field — the black clip a wrong VAE scaling produces",
        lambda: h3._integrity(torch, flat, (flat * 255).to(torch.uint8), good_audio, tel_of(Tel())),
        code="output_integrity_flat",
    )
    expect_refusal(
        "A SILENT SOUNDTRACK over a perfectly good picture — H3's own third branch",
        lambda: h3._integrity(torch, good_video, good_pixels, torch.zeros(2, 32000), tel_of(Tel())),
        code="output_integrity_silent",
    )
    expect_refusal(
        "NaN in the AUDIO alone, with the video clean",
        lambda: h3._integrity(
            torch, good_video, good_pixels, good_audio * float("nan"), tel_of(Tel())
        ),
        code="output_integrity_nan",
    )


def group_keys() -> None:
    print("\n== the key-exactness harness — and its own red arm ==")
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "h3-keys.py")],
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr
    if result.returncode == 0:
        observe("all five components key-exact against the pinned header", tail)
    else:
        failed("key-exactness", tail)

    # THE ARM: a graph that is one block short is a serve that would fail on a rented
    # card. Perturbing the config is the cheapest way to prove the harness can say so.
    import dataclasses
    import importlib.util

    import h3_arch
    from h3_arch.config import DitConfig

    with torch.device("meta"):
        short = h3_arch.build_dit(dataclasses.replace(DitConfig(), num_layers=49))
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

    with torch.device("meta"):
        full = h3_arch.build_dit(h3_arch.DIT_FULL)
    full_keys = set(full.state_dict())
    only_full = sorted(full_keys - want)
    only_curve = sorted(want - full_keys)
    if only_full and only_curve:
        observe(
            "the FULL-AdaLN structure is a DIFFERENT key set — curve is never shorthand for it",
            f"full-only: {only_full[:2]} ... curve-only: {only_curve}",
        )
    else:
        failed("full vs curve structure", f"{len(only_full)} / {len(only_curve)}")


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


GROUPS = {
    "components": group_components,
    "request": group_request,
    "plan": group_plan,
    "integrity": group_integrity,
    "keys": group_keys,
    "fence": group_fence,
}


def main() -> int:
    names = sys.argv[1:] or list(GROUPS)
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
