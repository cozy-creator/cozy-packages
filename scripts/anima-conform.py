#!/usr/bin/env python
"""Anima progress arms: the meter must MOVE, not merely be wired.

se-026 shipped a denoise hook that was installed on a DEEPCOPY of the block tree — the
`ModularPipeline.blocks` property returns one — so Diffusers never called it and
`cozy run list` read `conditioning 0%` for the whole 156-second render. An arm that
asserted the hook was installed would have passed on exactly that code.

So every arm here EXECUTES. The blocks come out of the pipeline the package itself builds,
the telemetry is Runtime's real `Telemetry` over a real attempt, and the denoise loop is
driven by Diffusers' own `AnimaDenoiseLoopWrapper.__call__` with only the model math
stubbed. No weights, GPU, network, or test framework.
"""

from __future__ import annotations

import itertools
import sys
import types
from pathlib import Path
from typing import Any

import torch
from diffusers import AnimaModularPipeline, ClassifierFreeGuidance, CosmosTransformer3DModel
from diffusers.modular_pipelines import BlockState, PipelineState
from diffusers.modular_pipelines.anima.denoise import AnimaLoopDenoiser

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anima"))

from cozy_runtime.author.fakes import fake_attempt, fake_telemetry  # noqa: E402

import anima as package  # noqa: E402

PASS = "  ok   "
FAIL = "  FAIL "
STEPS = 30
_failures = 0


def check(name: str, got: object, expected: object) -> None:
    global _failures
    if got == expected:
        print(f"{PASS}{name}")
        return
    _failures += 1
    print(f"{FAIL}{name}: got {got!r}, expected {expected!r}")


def fail(name: str, detail: str) -> None:
    global _failures
    _failures += 1
    print(f"{FAIL}{name}: {detail}")


class Recorder:
    """Runtime's real live lane, with the frames kept. `attempt.sink` is the seam the
    executor installs in production, so nothing here is a parallel telemetry."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        attempt = fake_attempt("anima-conform")
        attempt.sink = self.frames.append
        self.tel = fake_telemetry(attempt)
        self.phases = package._Phases(self.tel)

    @property
    def progress(self) -> list[Any]:
        return [frame for frame in self.frames if hasattr(frame, "advance")]

    def steps(self, stage: str) -> list[tuple[int, int]]:
        return [
            (frame.position, frame.total)
            for frame in self.progress
            if frame.stage == stage and frame.position is not None
        ]

    def line(self, frame: Any) -> str:
        """What `cozy run list` renders in its PROGRESS cell: "<stage> <percent>"."""
        return f"{frame.stage} {round((frame.overall_fraction or 0) * 100)}%"


def pipeline(recorder: Recorder, batched: bool = False) -> Any:
    """The package's own builder, on the CPU, with no components registered."""
    return package._text2image_pipeline(torch.device("cpu"), recorder.phases, batched)


def loop_state(loop: Any, steps: int) -> Any:
    """Every input the real loop block declares, so its own `get_block_state` is satisfied."""
    state = PipelineState()
    for param in loop.inputs:
        if param.name:
            state.set(param.name, torch.zeros(1))
    state.set("timesteps", torch.arange(steps))
    state.set("num_inference_steps", steps)
    return state


def drive(loop: Any, steps: int) -> None:
    """Diffusers' own denoise driver over the package's block, with the math stubbed.

    Only `loop_step` — the transformer forward — is replaced. `__call__`, its warmup
    arithmetic, its `self.progress_bar(total=...)` call and its update cadence are upstream's.
    """
    loop.loop_step = types.MethodType(
        lambda self, components, block_state, i, t: (components, block_state), loop
    )
    scheduler = types.SimpleNamespace(order=1)
    loop(types.SimpleNamespace(scheduler=scheduler), loop_state(loop, steps))


# ------------------------------------------------------------------------------- the arms


def arm_ladder() -> None:
    """The five phases are contiguous, cover the request, and are spelled for a person."""
    ladder = (
        package._ENCODE_PROMPT,
        package._CONDITION,
        package._DENOISE,
        package._DECODE,
        package._SAVE,
    )
    check(
        "phase names",
        [phase.stage for phase in ladder],
        ["encoding prompt", "conditioning", "denoise", "decoding", "saving image"],
    )
    check("ladder starts at zero", ladder[0].start, 0.0)
    check("ladder ends at one", ladder[-1].stop, 1.0)
    gaps = [
        (before.stage, after.stage)
        for before, after in itertools.pairwise(ladder)
        if before.stop != after.start
    ]
    check("no gap between phases", gaps, [])
    check("every phase advances", [p.stage for p in ladder if p.start >= p.stop], [])


def arm_phase_brackets() -> None:
    """Entering a phase emits a frame; the next entry closes the previous one at its stop."""
    recorder = Recorder()
    with recorder.phases as phases:
        for phase in (package._ENCODE_PROMPT, package._CONDITION, package._DECODE):
            phases.enter(phase)
    seen = [(frame.stage, frame.overall_fraction) for frame in recorder.progress]
    check(
        "brackets open and close in order",
        seen,
        [
            ("encoding prompt", 0.0),
            ("encoding prompt", 0.06),
            ("conditioning", 0.06),
            ("conditioning", 0.1),
            ("decoding", 0.9),
            ("decoding", 0.98),
        ],
    )
    check(
        "each phase is timed",
        sorted(recorder.tel.stages()),
        ["conditioning", "decoding", "encoding prompt"],
    )


def arm_denoise_loop_reports() -> None:
    """THE REGRESSION ARM. Diffusers' own loop driver, over the package's own pipeline.

    On the shipped code this arm is red: the hook rode a deepcopy, so the block the pipeline
    holds carried Diffusers' tqdm bar and produced no frames at all.
    """
    recorder = Recorder()
    built = pipeline(recorder)
    with recorder.phases as phases:
        phases.enter(package._ENCODE_PROMPT)
        blocks = built._blocks.sub_blocks
        check("the conditioning block is on the pipeline", "conditioning" in blocks, True)
        if "conditioning" in blocks:
            blocks["conditioning"](built, None)
        drive(blocks["denoise.denoise"], STEPS)
    counted = [(n, STEPS) for n in range(1, STEPS + 1)]
    check("every denoise step reports", recorder.steps("denoise"), counted)
    stages = sorted({frame.stage for frame in recorder.progress})
    check(
        "the loop stage is denoise",
        stages,
        ["conditioning", "decoding", "denoise", "encoding prompt"],
    )
    denoise = [frame for frame in recorder.progress if frame.stage == "denoise"]
    band = [denoise[0].overall_fraction, denoise[-1].overall_fraction] if denoise else []
    check("denoise spans its overall band", band, [0.1, 0.9])
    overall = [frame.overall_fraction for frame in recorder.progress]
    check("overall never moves backward", overall, sorted(overall))
    last = recorder.progress[-1].stage if recorder.progress else ""
    check("the loop leaves decoding open", last, "decoding")
    timed = recorder.tel.stages().get("denoise", -1.0) >= 0.0
    check("denoise step time is attributed", timed, True)
    # What the owner reads. The defect showed him one frame; this is what moved instead.
    rendered = [recorder.line(frame) for frame in recorder.progress]
    check("the meter advances", len(set(rendered)) > 10, True)
    if rendered:
        walk = f"{rendered[0]} -> {rendered[len(rendered) // 2]} -> {rendered[-1]}"
        print(f"         run list cell walks: {walk}")


def arm_deepcopy_trap() -> None:
    """THE NEGATIVE CONTROL: the shipped defect, executed.

    Installing the hook the way se-026 shipped it — through the `blocks` property — and then
    running the block the pipeline actually holds produces NO frames. This is the arm that
    distinguishes "the callback is installed" from "the callback fires", and it is why the
    denoise arm above cannot pass by accident. If a future Diffusers stops deepcopying,
    this arm goes red and `_text2image_pipeline`'s `blocks=` seam can be retired
    deliberately rather than by accident.
    """
    recorder = Recorder()
    shipped = AnimaModularPipeline(workflow="text2image")
    shipped.blocks.sub_blocks["denoise.denoise"].progress_bar = package._progress_bar(
        recorder.phases
    )
    live = shipped._blocks.sub_blocks["denoise.denoise"]
    check("`blocks` hands back a copy", shipped.blocks.sub_blocks["denoise.denoise"] is live, False)
    with recorder.phases:
        drive(live, 4)
    check("a hook installed through `blocks` reports nothing", recorder.steps("denoise"), [])
    check("...and the meter never moves", recorder.progress, [])

    fixed = pipeline(recorder)
    check(
        "`blocks=` keeps the instrumented block",
        type(fixed._blocks.sub_blocks["denoise.denoise"].progress_bar).__name__,
        "function",
    )


def arm_block_order() -> None:
    """The announce block sits between text encoding and conditioning, and nowhere else."""
    recorder = Recorder()
    order = list(pipeline(recorder)._blocks.sub_blocks)
    check(
        "text2image block order",
        order,
        [
            "text_encoder",
            "conditioning",
            "denoise.text_conditioning",
            "denoise.input",
            "denoise.prepare_latents",
            "denoise.set_timesteps",
            "denoise.denoise",
            "decode.decode",
            "decode.postprocess",
        ],
    )


def arm_batched_cfg() -> None:
    """C4: in a small pool every transformer forward streams the whole transformer, so CFG
    as one batch halves a step's copies. The package's batched denoiser, against Diffusers'
    own sequential one on a small random fp32 Cosmos transformer: one forward of batch 2
    inside the CFG interval, cond alone outside it, and the same noise prediction."""
    torch.manual_seed(3)
    transformer = CosmosTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        text_embed_dim=24,
        adaln_lora_dim=8,
        max_size=(4, 16, 16),
        rope_scale=(1.0, 4.0, 4.0),
        extra_pos_embed_type=None,
    ).eval()
    batches: list[int] = []
    transformer.register_forward_pre_hook(
        lambda module, args, kwargs: batches.append(kwargs["hidden_states"].shape[0]),
        with_kwargs=True,
    )
    built = pipeline(Recorder(), batched=True)
    denoiser = built._blocks.sub_blocks["denoise.denoise"].sub_blocks["denoiser"]
    check("the batched pipeline runs the batched denoiser", type(denoiser).__name__, "_BatchedDenoiser")
    stock = AnimaLoopDenoiser({"encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds")})
    guider = ClassifierFreeGuidance(guidance_scale=4.5, start=0.15, stop=0.7)
    components = types.SimpleNamespace(guider=guider, transformer=transformer)
    state = BlockState(
        num_inference_steps=10,
        latent_model_input=torch.randn(1, 16, 1, 12, 10),
        timestep=torch.tensor([0.4]),
        padding_mask=torch.zeros(1, 1, 96, 80),
        dtype=torch.float32,
        prompt_embeds=torch.randn(1, 7, 24),
        negative_prompt_embeds=torch.randn(1, 7, 24),
    )
    worst, shapes = 0.0, []
    for step in range(10):
        before = len(batches)
        denoiser(components, state, step, torch.tensor(400.0))
        batched = state.noise_pred
        shapes.append(batches[before:])
        stock(components, state, step, torch.tensor(400.0))
        worst = max(worst, (batched - state.noise_pred).abs().max().item())
    check("one batch-2 forward inside the interval, cond alone outside", shapes, [[1]] + [[2]] * 6 + [[1]] * 3)
    check("batched and sequential CFG agree to fp32 rounding", worst <= 1e-5, True)
    print(f"         max abs difference {worst:.2e}")


def arm_attention() -> None:
    """Diffusers' Cosmos attention repeats K and V by a ratio of head dims that is always 1,
    a whole copy of each per call. The package's processor makes none and computes the same
    bits as Diffusers' own on a small random Cosmos transformer, in both its attentions."""
    torch.manual_seed(5)

    def small() -> Any:
        return CosmosTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            text_embed_dim=24,
            adaln_lora_dim=8,
            max_size=(4, 16, 16),
            rope_scale=(1.0, 4.0, 4.0),
            extra_pos_embed_type=None,
        ).eval()

    stock = small()
    ours = small()
    ours.load_state_dict(stock.state_dict())
    for block in ours.transformer_blocks:
        for attention in (block.attn1, block.attn2):
            attention.set_processor(package._AttnProcessor())
    inputs = {
        "hidden_states": torch.randn(2, 16, 1, 12, 10),
        "timestep": torch.tensor([0.4, 0.4]),
        "encoder_hidden_states": torch.randn(2, 7, 24),
        "padding_mask": torch.zeros(1, 1, 96, 80),
        "return_dict": False,
    }

    class Count(torch.overrides.TorchFunctionMode):
        calls = 0

        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if getattr(func, "__name__", "") == "repeat_interleave":
                Count.calls += 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad():
        expected = stock(**inputs)[0]
        with Count():
            got = ours(**inputs)[0]
    check("no K/V copy in either attention", Count.calls, 0)
    check("the same bits as Diffusers' processor", torch.equal(got, expected), True)


ARMS = {
    "attention": arm_attention,
    "batched": arm_batched_cfg,
    "ladder": arm_ladder,
    "brackets": arm_phase_brackets,
    "denoise": arm_denoise_loop_reports,
    "deepcopy": arm_deepcopy_trap,
    "order": arm_block_order,
}


def main() -> int:
    selected = sys.argv[1:] or list(ARMS)
    unknown = [name for name in selected if name not in ARMS]
    if unknown:
        print(f"unknown arms: {', '.join(unknown)}", file=sys.stderr)
        return 2
    for name in selected:
        try:
            ARMS[name]()
        except Exception as exc:  # A crashed arm is red, with its exact exception.
            fail(name, f"{type(exc).__name__}: {exc}")
    print(f"\n{len(selected)} arms, {_failures} failures")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
