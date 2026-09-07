#!/usr/bin/env python
"""Deterministic MiniMax-H3 contract arms; no weights, GPU, network, or test framework.

Every arm executes the official Diffusers 0.40 implementation or a public package
boundary. Each historically dangerous invariant also carries a negative control. A green
run is a CPU semantic proof, not a generation or accelerator proof.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "minimax-h3"
sys.path.insert(0, str(H3))

from cozy_runtime.author import canonical_json, describe  # noqa: E402

import h3 as package  # noqa: E402
from conditioner import build_text_conditioner, text_conditioner_config  # noqa: E402
from gates import MediaFacts, pre_encode_gate  # noqa: E402
from h3_order import construction_order, encode_order  # noqa: E402
from official import (  # noqa: E402
    FPS,
    FRAMES,
    MAX_CONDITIONER_VISION_TOKENS,
    NumericalChecks,
    ResidentWeights,
    ScheduleFacts,
    _aligned_soundtrack,
    _apply_transformer_dtype,
    _artifact_sections,
    _as_float32,
    _dit_specs,
    _processor,
    _ScopedPipeline,
    _validate_dual_dit_topology,
    _validate_model_contract,
    _video_at_24fps,
    canonical_timestep_plan,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    supported_steps,
    timestep_plan_digest,
    validate_reference_policy,
)

STEPS = supported_steps()
DEFAULT_STEPS = min(STEPS)

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0

PLAN_DIGESTS = {
    "fl2va": "8cd647f223acb56f864773e1a86bd8bcc0bb8d7a1c83ce2de33b7209844dd049",
    "ref2va": "3ec1b8e59c8b5dc74a4656d299d25ae206249cbd2b3f90b3c2981f8123b1b4ac",
}
# Exact float32 vectors per served step count, banked against Diffusers 0.40.
VECTOR_DIGESTS = {
    30: {
        "video_sigmas": "8d29489d97d06bd9f229ff45051cfd259e59832cf61574d0f0b71d99b28b466a",
        "audio_sigmas": "0b8bf82a517ac3d9c2eef697ccddfd8ab5c8a18ea1136efc1ad94e9deb7f015b",
        "video_timesteps": "3f87bbb7cc44a4a6f87e0c77192fd3a73e936cb649497b1cce77472ce6e27b25",
        "audio_timesteps": "c8d95fe92d06fdb69c90eb9f42aba1e7c318f339960ef8950b012e32156caa85",
    },
    40: {
        "video_sigmas": "0d7ed9c26b4419707c2f495c9f447c0b74ac2576081d701891e9908c822c0fc8",
        "audio_sigmas": "9a009908dbf1f190a30469f65811401d9a6de226366466a792301db5bf104b4d",
        "video_timesteps": "e5895cca43994ec051ca9fe2bc5ad74ab230f3b3d54c3be169ef2e2f7af87bd8",
        "audio_timesteps": "eb389eb4785d0a202fca946c1b92320b5cbd29298f9bdc9717e7b4666dd32df5",
    },
    50: {
        "video_sigmas": "9f5aed90908ca1571d5d00365722d83dbf5b37c02c7e9782ec6bfb9c32361f53",
        "audio_sigmas": "acef7cee872f06e6f2a8f88c092e9d5b827ea91f33ed2f9c37636b61aa5b5643",
        "video_timesteps": "aa35beaa105f51a70a66f66dc8ad63641c9c517e4a66e0de63fb67ee18c87667",
        "audio_timesteps": "1d5b28a96440166682182782de94ba8e538a8970be3e5a67c0891c504346daec",
    },
}
BLOCK_ROWS = 315
FINAL_ROWS = 207
ASSET_DIGESTS = {
    "tokenizer/merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "tokenizer/tokenizer_config.json": (
        "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"
    ),
    "tokenizer/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "processor/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "processor/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}
TOKEN_CORPUS_DIGEST = "47759c8d2e1a24944edb8f712c7ffe66b650aa3481352487970bcdb3cd55fa58"


def dit_config(task: str, modulation: str = "full") -> dict[str, object]:
    plan = canonical_timestep_plan(cast(Any, task))
    extension = {"task": task, "modulation": modulation}
    if modulation == "adaln-pruned":
        extension["timestep_plan_digest"] = f"sha256:{plan.digest}"
    return {"cozy_h3": extension}


def arm_producer_configs() -> None:
    """Pass actual producer bytes into the serving parser before any weight transfer."""
    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
    from h3_tables.model_config import (
        dual_adaln_pruned_config,
        dual_full_config,
        parse_production_config,
    )
    from h3_tables.plans import parse_plan

    assets = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets"
    sections = parse_production_config((assets / "model-config.json").read_bytes())
    plans = {
        task: parse_plan((assets / f"timestep-plan.{task}.json").read_bytes(), task=task)
        for task in ("fl2va", "ref2va")
    }
    for structure, raw in (
        ("full", dual_full_config(sections)),
        ("adaln-pruned", dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])),
    ):
        document = canonical_json.decode(raw)
        specs = _dit_specs(_artifact_sections(document))
        check(f"actual producer {structure} config serves both tasks", set(specs), set(plans))
        check(
            f"actual producer {structure} modulation",
            {row[1] for row in specs.values()},
            {structure},
        )
        for component, task in (("fl2va_dit", "fl2va"), ("ref2va_dit", "ref2va")):
            for field, value in (
                ("task", "ref2va" if task == "fl2va" else "fl2va"),
                ("timestep_plan_digest", "sha256:" + "0" * 64),
            ):
                changed = canonical_json.decode(raw)
                changed[component]["cozy_h3"][field] = value
                refusal(
                    f"{structure} {component} wrong {field} refuses",
                    partial(_dit_specs, changed),
                    "artifact_config",
                )
            if structure == "adaln-pruned":
                changed = canonical_json.decode(raw)
                del changed[component]["cozy_h3"]["timestep_plan_digest"]
                refusal(
                    f"{component} pruned tables need their plan digest",
                    partial(_dit_specs, changed),
                    "artifact_config",
                )


def arm_producer_construction_order() -> None:
    """The producer's one ordered spec resource follows the real serving factory."""
    import torch
    from cozy_runtime.author import Config

    from official import OfficialH3Pipeline

    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
    from h3_tables.job import _asset, _full_order
    from h3_tables.model_config import (
        dual_adaln_pruned_config,
        dual_full_config,
        parse_production_config,
    )
    from h3_tables.order import current_order
    from h3_tables.plans import parse_plan
    from h3_tables.source import official_full_specs

    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    plans = {
        task: parse_plan(_asset(f"timestep-plan.{task}.json"), task=task)
        for task in ("fl2va", "ref2va")
    }
    for mode, raw, expected in (
        ("full", dual_full_config(sections), _full_order(sections, current.rows)),
        (
            "adaln-pruned",
            dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"]),
            current.rows,
        ),
    ):
        with torch.device("meta"):
            pipeline = OfficialH3Pipeline(Config(canonical_json.decode(raw)))
        actual = tuple(
            (component, key)
            for component, module in pipeline.components.items()
            for key in module.state_dict()
        )
        check(
            f"{mode} producer order equals actual serving constructor ({len(actual)} rows)",
            expected == actual,
            True,
        )
        if mode == "full":
            for component, section in (
                ("fl2va_dit", "transformer"),
                ("ref2va_dit", "transformer_ref"),
            ):
                state = pipeline.components[component].state_dict()
                specs = official_full_specs(sections[section])
                check(f"{component} ordered spec names", tuple(specs) == tuple(state), True)
                check(
                    f"{component} lexical ordering is a rejected control",
                    tuple(sorted(specs)) == tuple(state),
                    False,
                )
                for key, (dtype, shape) in specs.items():
                    value = state[key]
                    if (
                        tuple(value.shape) != shape
                        or value.dtype != {"bf16": torch.bfloat16, "f32": torch.float32}[dtype]
                    ):
                        raise AssertionError(
                            f"ordered full spec geometry changed: {component}/{key}"
                        )
                observe(f"{component} all 638 ordered spec shapes and dtypes match")


def tiny_text_config() -> dict[str, object]:
    """Production key topology at tiny dimensions: 27 vision and 64 source text layers."""
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
            "attention_dropout": 0.0,
            "use_cache": False,
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 27,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [8, 16, 24],
        },
        "image_token_id": 60,
        "video_token_id": 61,
        "vision_start_token_id": 62,
        "vision_end_token_id": 63,
        "tie_word_embeddings": False,
        "cozy_h3": text_conditioner_config(),
    }


def observe(name: str, detail: str = "") -> None:
    print(f"{PASS}{name}" + (f"\n         {detail}" if detail else ""))


def fail(name: str, detail: str) -> None:
    global _failures
    _failures += 1
    print(f"{FAIL}{name}\n         {detail}")


def check(name: str, got: Any, want: Any) -> None:
    if got == want:
        observe(name, repr(got))
    else:
        fail(name, f"got {got!r}, want {want!r}")


def red(name: str, defective: Any, correct: Any) -> None:
    if defective != correct:
        observe(f"RED CONTROL - {name}", f"defect {defective!r} != {correct!r}")
    else:
        fail(f"RED CONTROL - {name}", "the defective expression agreed; the arm is blind")


def refusal(name: str, fn: Callable[[], object], code: str | None = None) -> None:
    try:
        fn()
    except Exception as exc:  # The observed exception is the subject of this driver arm.
        got = getattr(exc, "code", None) or type(exc).__name__
        if code is None or got == code:
            observe(name, f"{got}: {str(exc).splitlines()[0][:140]}")
        else:
            fail(name, f"refused with {got!r}, expected {code!r}: {exc}")
        return
    fail(name, "the call succeeded")


def float_digest(values: Any) -> str:
    raw = b"".join(struct.pack("<f", float(value)) for value in values)
    return hashlib.sha256(raw).hexdigest()


def arm_schedule() -> None:
    import torch
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )

    print("\n== exact schedule and AdaLN-pruned handoff ==")
    plans = {task: canonical_timestep_plan(task) for task in ("fl2va", "ref2va")}
    for task, plan in plans.items():
        check(f"{task} canonical plan digest", plan.digest, PLAN_DIGESTS[task])
        committed = (H3 / "timestep-plans" / f"{task}.json").read_bytes()
        check(
            f"{task} committed semantic document",
            canonical_json.encode(canonical_json.decode(committed)),
            plan.canonical_bytes(),
        )
        check(f"{task} committed semantic identity", timestep_plan_digest(committed), plan.digest)
        parsed = cast(dict[str, Any], canonical_json.decode(committed))
        canonical = canonical_json.encode(parsed)
        reordered = json.dumps(
            dict(reversed(list(parsed.items()))), indent=2, ensure_ascii=False
        ).encode()
        check(f"{task} formatting-invariant identity", timestep_plan_digest(reordered), plan.digest)
        changed = dict(parsed, frames=parsed["frames"] + 1)
        red(
            f"{task} meaning changes identity",
            timestep_plan_digest(json.dumps(changed).encode()),
            plan.digest,
        )
        refusal(
            f"{task} duplicate JSON key refuses",
            partial(timestep_plan_digest, canonical[:-1] + b',"task":"other"}'),
            "ValueError",
        )
        refusal(
            f"{task} non-finite JSON number refuses",
            partial(timestep_plan_digest, canonical[:-1] + b',"bad":NaN}'),
            "ValueError",
        )
        document = json.loads(plan.canonical_bytes())
        check(f"{task} served step counts", plan.steps, STEPS)
        check(
            f"{task} schedule evaluation counts",
            [len(row["evaluations"]) for row in document["schedules"]],
            list(STEPS),
        )
        check(
            f"{task} complete class names",
            [
                entry["name"]
                for entry in document["schedules"][0]["evaluations"][0]["modulation_classes"]
            ],
            ["target_video", "text", "target_audio", "condition_video", "condition_audio"],
        )
        check(
            f"{task} union block-table rows",
            len(document["table_keys"]["block_modulation"]),
            BLOCK_ROWS,
        )
        check(
            f"{task} union final-normalization rows",
            len(document["table_keys"]["final_normalization"]),
            FINAL_ROWS,
        )
        check(
            f"{task} no terminal forward",
            [row["terminal"]["transformer_evaluation"] for row in document["schedules"]],
            [False] * len(STEPS),
        )
    red("task cannot collide in the plan identity", plans["fl2va"].digest, plans["ref2va"].digest)
    first = plans["fl2va"].schedules[0]
    refusal(
        "an extra penultimate sigma cannot hide outside canonical bytes",
        lambda: replace(first, video_sigmas=(*first.video_sigmas[:-1], 0.125, 0.0)),
    )
    refusal(
        "schedules cannot repeat a step count",
        lambda: replace(plans["fl2va"], schedules=(first, first)),
    )
    refusal(
        "an unserved step count refuses typed before any work",
        lambda: plans["fl2va"].schedule(DEFAULT_STEPS - 1),
        "steps",
    )
    red("the retired 30-point/29-forward grid is not a served schedule", 29 in STEPS, True)

    document = json.loads(plans["fl2va"].canonical_bytes())
    for index, offsets, final_rows in (
        (0, [0, 1, 2, 3, 8], 3),
        (1, [0, 1, 5, 6, 11], 4),
    ):
        classes = document["schedules"][0]["evaluations"][index]["modulation_classes"]
        unique = sorted({float.fromhex(entry["timestep"]) for entry in classes})
        got = [
            unique.index(float.fromhex(entry["timestep"])) * 3 + entry["modality_tag"]
            for entry in classes
        ]
        check(f"evaluation {index} exact AdaLN row offsets", got, offsets)
        check(f"evaluation {index} final-normalization row count", len(unique), final_rows)

    for schedule in plans["fl2va"].schedules:
        steps = schedule.transformer_evaluations
        check(f"{steps} steps use {steps + 1} grid points", schedule.sigma_grid_points, steps + 1)
        for name, shift, sigmas, timesteps in (
            ("video", 12.0, schedule.video_sigmas, schedule.video_timesteps),
            ("audio", 3.0, schedule.audio_sigmas, schedule.audio_timesteps),
        ):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(schedule.sigma_grid_points)
            official_sigmas = tuple(float(value) for value in scheduler.sigmas.float().cpu())
            official_timesteps = tuple(
                float(value) for value in scheduler.timesteps.float().cpu()
            )
            check(f"{steps}-step {name} sigmas equal Diffusers", sigmas, official_sigmas)
            check(f"{steps}-step {name} timesteps equal Diffusers", timesteps, official_timesteps)
            check(f"{steps}-step {name} forwards", len(official_timesteps), steps)
            check(
                f"{steps}-step {name} sigma digest",
                float_digest(official_sigmas),
                VECTOR_DIGESTS[steps][f"{name}_sigmas"],
            )
            check(
                f"{steps}-step {name} timestep digest",
                float_digest(official_timesteps),
                VECTOR_DIGESTS[steps][f"{name}_timesteps"],
            )

    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices=torch.tensor([2, 3]),
        audio_indices=torch.tensor([4, 5]),
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
        num_text_tokens=2,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    check(
        "official row plan unique levels",
        unique.tolist(),
        [0.25, 0.5, _as_float32(0.999), 1.0],
    )
    check("text inherits target-video timestep", inverse[:2].tolist(), [0, 0])
    check("condition and target rows remain distinct", inverse.tolist(), [0, 0, 2, 0, 3, 1])

    # Every (timestep, modality) pair the official row builder can present under any
    # served schedule is a row of the one union table; an off-grid level is not.
    table_timesteps, block_keys = plans["fl2va"].table_layout()
    covered = {(table_timesteps[row], tag) for row, tag in block_keys}
    tags = torch.tensor([1, 0, 0, 0, 2, 2])
    presented: set[tuple[float, int]] = set()
    for schedule in plans["fl2va"].schedules:
        for video_timestep, audio_timestep in zip(
            schedule.video_timesteps, schedule.audio_timesteps, strict=True
        ):
            unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video_indices=torch.tensor([2, 3]),
                audio_indices=torch.tensor([4, 5]),
                num_condition_video_rows=1,
                num_condition_audio_rows=1,
                num_text_tokens=2,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,
                condition_video_timestep=max(video_timestep, _as_float32(0.999)),
                condition_audio_timestep=1.0,
            )
            presented |= {
                (float(unique[row]), int(tag)) for row, tag in zip(inverse, tags, strict=True)
            }
    check("official row classes of every schedule are table rows", presented <= covered, True)
    check("the union table carries no unpresented row", covered <= presented, True)
    red("an off-grid level is not a table row", (0.5, 0) in covered, True)

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(DEFAULT_STEPS + 1)
    sample = torch.tensor([-1.0])
    velocity = torch.tensor([2.0])
    actual = float(scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0])
    sigma, sigma_next = (float(value) for value in scheduler.sigmas[:2])
    correct = -1.0 + (sigma - sigma_next) * 2.0
    defective = -1.0 + (sigma_next - sigma) * 2.0
    check("official solver moves data-ward", actual, correct)
    red("reversed anti-denoising sign", defective, correct)


def meta_h3_pipeline() -> Any:
    import torch
    from cozy_runtime.author import Config, canonical_json

    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
    from h3_tables.model_config import dual_full_config, parse_production_config

    from official import OfficialH3Pipeline

    assets = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets"
    config = canonical_json.decode(
        dual_full_config(parse_production_config((assets / "model-config.json").read_bytes()))
    )
    with torch.device("meta"):
        return OfficialH3Pipeline(Config(config))


def arm_reference_resolution() -> None:
    from typing import get_type_hints

    import msgspec
    import numpy as np
    import torch
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    print("\n== request-local reference resolution uses official preprocessing ==")
    pipe = meta_h3_pipeline()
    original_config = dict(pipe._pipes["ref2va"].config)
    pixels = np.arange(1086 * 1448 * 3, dtype=np.uint8).reshape(1448, 1086, 3)
    references = [MiniMaxH3ImageReference(image=pixels)]
    normalized = {}
    for edge in (768, 769, 1024, 2048):
        state = pipe.start_ref2va(
            prompt="A person in a garden.",
            references=references,
            generator=torch.Generator().manual_seed(7),
            steps=DEFAULT_STEPS,
            reference_image_short_edge=edge,
        )
        image = state.normalized_references[0].image
        normalized[edge] = np.asarray(image)
        width, height = image.size
        check(
            f"{edge}px budget agrees with actual upstream image geometry",
            reference_image_vision_tokens(1086, 1448, edge),
            width * height // 1024,
        )
        check(
            f"{edge}px request leaves shared config unchanged",
            dict(pipe._pipes["ref2va"].config) == original_config,
            True,
        )
    check("768px reference geometry", normalized[768].shape, (1024, 768, 3))
    check("2048px reference geometry", normalized[2048].shape, (2720, 2048, 3))
    default = pipe.start_ref2va(
        prompt="A person in a garden.",
        references=references,
        generator=torch.Generator().manual_seed(7),
        steps=DEFAULT_STEPS,
    )
    check(
        "default pixels stay identical after smaller requests",
        np.array_equal(np.asarray(default.normalized_references[0].image), normalized[2048]),
        True,
    )

    before = pipe._blocks["ref2va"].sub_blocks["before_encode"]

    def broken_setup(components: Any, state: Any) -> None:
        check(
            "failed request sees its own resolution",
            components.config.reference_image_short_edge,
            768,
        )
        raise ValueError("preprocessing failed")

    pipe._blocks["ref2va"].sub_blocks["before_encode"] = broken_setup
    try:
        refusal(
            "failed preprocessing does not mutate shared configuration",
            lambda: pipe.start_ref2va(
                prompt="A person in a garden.",
                references=references,
                generator=torch.Generator().manual_seed(7),
                steps=DEFAULT_STEPS,
                reference_image_short_edge=768,
            ),
        )
    finally:
        pipe._blocks["ref2va"].sub_blocks["before_encode"] = before
    check(
        "shared config survives preprocessing failure",
        dict(pipe._pipes["ref2va"].config) == original_config,
        True,
    )
    field = get_type_hints(package.ReferenceMediaToVideoInput, include_extras=True)[
        "reference_image_short_edge"
    ]
    check("typed request accepts 768px", msgspec.convert(768, type=field), 768)
    check("typed request permits upstream rounding", msgspec.convert(769, type=field), 769)
    for invalid in (0, 255, 2080):
        refusal(
            f"typed request refuses invalid edge {invalid}",
            partial(msgspec.convert, invalid, type=field),
        )


def arm_zero_reference_preparation() -> None:
    import torch

    print("\n== text-only request reaches the official denoise loop ==")
    pipe = meta_h3_pipeline()
    # Only preparation executes: synthetic text embeddings and a CPU scope stand
    # in for the preceding encoder and GPU. No model forward or weights are read.
    pipe.components["fl2va_dit"] = SimpleNamespace(device=torch.device("cpu"))

    def start(steps: int) -> Any:
        return pipe.start_fl2va(
            prompt="Three friends walk in a garden.",
            first_frame=None,
            last_frame=None,
            generator=torch.Generator().manual_seed(7),
            steps=steps,
        )

    refusal("an unserved step count refuses before preparation", lambda: start(29), "steps")
    check(
        "every served step count states its official grid",
        [start(steps).num_inference_steps for steps in STEPS],
        [steps + 1 for steps in STEPS],
    )
    state = start(DEFAULT_STEPS)
    state.set("prompt_embeds", torch.zeros(1, 4, 5120))
    state.set("text_token_tags", torch.ones(4, dtype=torch.long))

    class ReachedDenoise(Exception):
        pass

    def stop_before_forward() -> None:
        raise ReachedDenoise()

    try:
        pipe.denoise("fl2va", state, on_step=lambda _: None, cancel=stop_before_forward)
    except ReachedDenoise:
        observe("zero-reference request reaches first forward without keyframe-only inputs")
    else:
        check("zero-reference preparation must stop before any model forward", False, True)
    check("text-only video rows", tuple(state.latents.shape), (102816, 96))
    check("text-only audio rows", tuple(state.audio_latents.shape), (1150, 32))
    check("text-only anchors", state.keyframe_anchors, ())
    check("text-only rows remain finite", bool(torch.isfinite(state.latents).all()), True)
    check("text-only default grid means the default forwards", len(state.timesteps), DEFAULT_STEPS)
    check(
        "the executed schedule is the plan's default",
        pipe._plans["fl2va"].executed(state.timesteps.tolist(), state.audio_timesteps.tolist()),
        pipe._plans["fl2va"].schedule(DEFAULT_STEPS),
    )
    refusal(
        "an off-plan executed schedule refuses",
        lambda: pipe._plans["fl2va"].executed(state.timesteps.tolist()[:-1], []),
        "artifact_config",
    )


def arm_graph_and_dtypes() -> None:
    import torch
    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Blocks,
        MiniMaxH3ModularPipeline,
        MiniMaxH3Transformer3DModel,
    )

    print("\n== official task-pruned graphs and destination dtypes ==")
    common = {
        "image_processor",
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
        "video_processor",
    }
    for task, owned, absent in (
        ("fl2va", "transformer", "transformer_ref"),
        ("ref2va", "transformer_ref", "transformer"),
    ):
        workflow = MiniMaxH3Blocks().get_workflow(task)
        components = {component.name for component in workflow.expected_components}
        check(f"{task} exact official components", components, common | {owned})
        check(f"{task} omits sibling transformer", absent in components, False)

    with torch.device("meta"):
        transformer = MiniMaxH3Transformer3DModel()
        video_vae = AutoencoderKLMiniMaxH3()
        audio_vae = AutoencoderKLMiniMaxH3Audio()
        wrong_audio_vae = AutoencoderKLMiniMaxH3Audio(sampling_rate=44100)
    # Serving constructs nonpersistent buffers on CPU while parameters stay meta.
    # The tiny forward fixture has one frequency (exactly 1), which cannot expose
    # a lossy BF16 round trip. Exercise the actual 16-frequency release here.
    transformer.rope = type(transformer.rope)()
    frequencies = transformer.rope.inv_freq.clone()
    positions = torch.tensor([[0, 1, 2], [17420, 31, 568]], dtype=torch.float64)
    expected_rotary = transformer.rope(positions)
    _apply_transformer_dtype(transformer)
    check(
        "all 16 derived RoPE frequencies preserve FP32 values",
        torch.equal(transformer.rope.inv_freq, frequencies),
        True,
    )
    for name, actual, expected in zip(
        ("cos", "sin"), transformer.rope(positions), expected_rotary, strict=True
    ):
        check(f"{name} preserves long-reference coordinates", torch.equal(actual, expected), True)
    rounded = frequencies.bfloat16().float()
    red(
        "BF16 round trip changes the actual frequency table",
        torch.equal(rounded, frequencies),
        True,
    )
    fl_pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3Blocks().get_workflow("fl2va"))
    _validate_model_contract(fl_pipe, transformer, video_vae, audio_vae)
    observe("official scalar model contract")
    refusal(
        "shape-preserving audio clock drift refuses",
        lambda: _validate_model_contract(fl_pipe, transformer, video_vae, wrong_audio_vae),
        "artifact_config",
    )
    full_sections: dict[str, dict[str, object]] = {
        "audio_vae": {},
        "fl2va_dit": dit_config("fl2va"),
        "ref2va_dit": dit_config("ref2va"),
        "text_encoder": tiny_text_config(),
        "video_vae": {},
    }
    check(
        "uniform dual FULL config sections",
        set(_artifact_sections(full_sections)),
        set(full_sections),
    )
    refusal(
        "an unweighted processor config cannot hide in the artifact",
        lambda: _artifact_sections({**full_sections, "processor": {}}),
        "artifact_config",
    )
    check(
        "uniform dual FULL structure",
        {spec[1] for spec in _dit_specs(full_sections).values()},
        {"full"},
    )
    mixed = {
        **full_sections,
        "ref2va_dit": dit_config("ref2va", "adaln-pruned"),
    }
    refusal(
        "mixed FULL and AdaLN-pruned task structures refuse",
        lambda: _dit_specs(mixed),
        "artifact_config",
    )
    refusal(
        "task DiTs with different architecture config refuse",
        lambda: _dit_specs(
            {**full_sections, "ref2va_dit": {**full_sections["ref2va_dit"], "hidden_size": 7}}
        ),
        "artifact_config",
    )
    wrong_plan = dit_config("fl2va", "adaln-pruned")
    cast(dict[str, Any], wrong_plan["cozy_h3"])["timestep_plan_digest"] = "sha256:" + "0" * 64
    refusal(
        "an AdaLN-pruned plan digest is exact artifact config",
        lambda: _dit_specs({**full_sections, "fl2va_dit": wrong_plan}),
        "artifact_config",
    )
    transformer_counts = Counter(str(value.dtype) for value in transformer.state_dict().values())
    check("transformer destination count", len(transformer.state_dict()), 638)
    check(
        "transformer exact mixed precision",
        transformer_counts,
        Counter({"torch.bfloat16": 626, "torch.float32": 12}),
    )
    check(
        "video VAE remains fp32",
        Counter(str(value.dtype) for value in video_vae.state_dict().values()),
        Counter({"torch.float32": 703}),
    )
    check("audio VAE state count", len(audio_vae.state_dict()), 1087)
    check("audio VAE parameter destinations", len(dict(audio_vae.named_parameters())), 832)
    red("uniform bf16 transformer cast", transformer_counts, Counter({"torch.bfloat16": 638}))

    class _Component:
        device = "meta"

    class _Pipe:
        def __init__(self) -> None:
            self.existing = 1

    inner = _Pipe()
    scoped = _ScopedPipeline(inner, _Component(), overrides={"over": 2})
    check(
        "scoped view forwards reads, overrides, and the admitted component device",
        (scoped.existing, scoped.over, scoped.device),
        (1, 2, "meta"),
    )
    refusal(
        "a block attribute write on the scoped view refuses instead of evaporating",
        lambda: setattr(scoped, "communicated", 3),
        "artifact_config",
    )
    check(
        "the refused write reached neither the view nor the wrapped pipe",
        (sorted(vars(inner)), "communicated" in vars(scoped)),
        (["existing"], False),
    )


def arm_text_conditioner() -> None:
    import torch
    from cozy_runtime.author import Artifact, Config
    from cozy_runtime.internal.derive import derive
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    print("\n== exact 50-layer pre-norm Qwen3-VL conditioner ==")
    source = tiny_text_config()
    upstream = {key: value for key, value in source.items() if key != "cozy_h3"}

    torch.manual_seed(7)
    initialized = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**upstream)).eval()
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        None, config=initialized.config, state_dict=initialized.state_dict(), dtype=torch.bfloat16
    )
    torch.manual_seed(7)
    truncated = build_text_conditioner(source)
    state = truncated.state_dict()
    check("truncated conditioner persistent census", len(state), 902)
    check(
        "truncated conditioner split",
        (
            sum(key.startswith("model.visual.") for key in state),
            sum(key.startswith("model.language_model.") for key in state),
        ),
        (351, 551),
    )
    check(
        "truncated conditioner keeps source config surface",
        (
            hasattr(truncated, "model"),
            truncated.config.text_config.num_hidden_layers,
            len(truncated.model.language_model.layers),
            type(truncated.model.language_model.norm).__name__,
            type(truncated.lm_head).__name__,
        ),
        (True, 64, 50, "Identity", "Identity"),
    )
    removed = [
        key
        for key in state
        if key == "lm_head.weight"
        or key == "model.language_model.norm.weight"
        or (key.startswith("model.language_model.layers.") and int(key.split(".")[3]) >= 50)
    ]
    check("tail, final norm, and language head are absent", removed, [])
    check(
        "every retained conditioner destination is BF16",
        Counter(str(value.dtype) for value in state.values()),
        Counter({"torch.bfloat16": 902}),
    )
    check(
        "truncation preserves every retained initialized weight",
        all(torch.equal(value, full.state_dict()[key]) for key, value in state.items()),
        True,
    )
    upstream_buffers = dict(full.named_buffers())
    for name, buffer in truncated.named_buffers():
        check(
            f"conditioner derived buffer matches upstream loader: {name}",
            (buffer.dtype, torch.equal(buffer, upstream_buffers[name])),
            (upstream_buffers[name].dtype, True),
        )

    # Exercise the release's actual rotary widths rather than the tiny fixture's
    # one-frequency vision table, whose only value (1) survives a BF16 round trip.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionRotaryEmbedding,
    )

    rotary_source = tiny_text_config()
    cast(dict[str, Any], rotary_source["text_config"]).update(
        head_dim=128,
        rope_theta=5000000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    )
    cast(dict[str, Any], rotary_source["vision_config"]).update(hidden_size=72, num_heads=1)
    rotary_model = build_text_conditioner(rotary_source)
    expected_text = Qwen3VLTextRotaryEmbedding(rotary_model.config.text_config)
    expected_vision = Qwen3VLVisionRotaryEmbedding(36)
    for kind, actual_rotary, expected_rotary in (
        ("text", rotary_model.model.language_model.rotary_emb, expected_text),
        ("vision", rotary_model.model.visual.rotary_pos_emb, expected_vision),
    ):
        expected_buffers = dict(expected_rotary.named_buffers())
        for name, buffer in actual_rotary.named_buffers():
            check(
                f"{kind} full-width rotary {name} retains constructor FP32 values",
                (buffer.dtype, torch.equal(buffer, expected_buffers[name])),
                (torch.float32, True),
            )
        check(
            f"{kind} release rotary width",
            actual_rotary.inv_freq.numel(),
            64 if kind == "text" else 18,
        )
        red(
            f"{kind} BF16 rotary round trip changes the frequency table",
            torch.equal(actual_rotary.inv_freq, actual_rotary.inv_freq.bfloat16().float()),
            True,
        )
    positions = torch.tensor([0, 31, 512, 2048, 16842]).reshape(1, 1, -1).expand(3, 1, -1)
    rotary_input = torch.zeros(1, 5, 128, dtype=torch.bfloat16)
    for name, actual, expected in zip(
        ("cos", "sin"),
        rotary_model.model.language_model.rotary_emb(rotary_input, positions),
        expected_text(rotary_input, positions),
        strict=True,
    ):
        check(
            f"text {name} matches upstream at long reference positions",
            torch.equal(actual, expected),
            True,
        )
    vision_positions = torch.tensor([[0, 0], [31, 31], [68, 90]])
    check(
        "vision angles match upstream at reference image positions",
        torch.equal(
            rotary_model.model.visual.rotary_pos_emb(vision_positions),
            expected_vision(vision_positions),
        ),
        True,
    )

    imported = {
        (
            "visual." + key.removeprefix("model.visual.")
            if key.startswith("model.visual.")
            else "model." + key.removeprefix("model.language_model.")
        ): key
        for key in state
    }
    check("902-row import rekey is one-to-one", len(imported), 902)
    check(
        "import rekey reproduces the construction key set",
        set(imported.values()) == set(state),
        True,
    )

    torch.manual_seed(9)
    inputs = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    positions = torch.arange(4).reshape(1, 1, 4).expand(4, 1, 4)
    visual_mask = torch.tensor([[False, True, False, True]])
    deepstack = [torch.randn(2, 16, dtype=torch.bfloat16) for _ in range(3)]
    arguments = {
        "inputs_embeds": inputs,
        "position_ids": positions,
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "visual_pos_masks": visual_mask,
        "deepstack_visual_embeds": deepstack,
        "use_cache": False,
        "output_hidden_states": True,
    }
    with torch.no_grad():
        full_output = full.model.language_model(**arguments).hidden_states[50]
        truncated_result = truncated.model.language_model(**arguments)
        truncated_output = truncated_result.hidden_states[50]
    check(
        "early exit equals full Qwen hidden_states[50] with visual injection",
        (
            torch.equal(truncated_output, full_output),
            float((truncated_output - full_output).abs().max()),
        ),
        (True, 0.0),
    )
    check(
        "parameterless final norm exposes the same pre-norm state",
        torch.equal(truncated_result.last_hidden_state, truncated_output),
        True,
    )

    refusal(
        "missing text cozy_h3 extension refuses",
        lambda: build_text_conditioner(upstream),
        "artifact_config",
    )
    extra = dict(source)
    extra["cozy_h3"] = {**text_conditioner_config(), "fallback": True}
    refusal(
        "extra text cozy_h3 field refuses",
        lambda: build_text_conditioner(extra),
        "artifact_config",
    )
    changed = dict(source)
    changed["cozy_h3"] = {**text_conditioner_config(), "retained_decoder_layers": 51}
    refusal(
        "changed retained-layer count refuses",
        lambda: build_text_conditioner(changed),
        "artifact_config",
    )
    wrong_source = dict(source)
    wrong_source["text_config"] = {
        **cast(dict[str, object], source["text_config"]),
        "num_hidden_layers": 63,
    }
    refusal(
        "changed source architecture depth refuses",
        lambda: build_text_conditioner(wrong_source),
        "artifact_config",
    )

    model_config = {
        "audio_vae": {},
        "fl2va_dit": dit_config("fl2va", "adaln-pruned"),
        "ref2va_dit": dit_config("ref2va", "adaln-pruned"),
        "text_encoder": source,
        "video_vae": {},
    }

    def contract_document() -> dict[str, object]:
        result = derive(
            package.H3Model(),
            Artifact("se-018-audit", {}, Config(model_config)),
        )
        return {
            "schema": "cozy.runtime.model_construction_contract.v1",
            "destination_sets": [
                {"component": row.component, "keys": list(row.keys)}
                for row in result.destination_sets
            ],
        }

    first_contract = contract_document()
    second_contract = contract_document()
    first_order = encode_order(construction_order(first_contract))
    second_order = encode_order(construction_order(second_contract))
    check(
        "Runtime construction order reproduces byte-identically",
        first_order == second_order,
        True,
    )
    rows = construction_order(first_contract)
    text_rows = construction_order(first_contract, ["text_encoder"])
    check("Runtime whole-model order count", len(rows), 3858)
    check("Runtime text-conditioner order count", len(text_rows), 902)
    check(
        "Runtime component order",
        list(dict.fromkeys(component for component, _ in rows)),
        ["fl2va_dit", "ref2va_dit", "text_encoder", "video_vae", "audio_vae"],
    )
    observe(
        "Runtime order handoff",
        f"whole sha256:{hashlib.sha256(first_order).hexdigest()}, "
        f"text sha256:{hashlib.sha256(encode_order(text_rows)).hexdigest()}",
    )


def arm_adaln_pruned() -> None:
    import torch
    from diffusers import MiniMaxH3Transformer3DModel

    from adaln_pruned import AdaLNPrunedMiniMaxH3Transformer

    print("\n== AdaLN-pruned modulation over the inherited Diffusers forward ==")
    plan = canonical_timestep_plan("fl2va")
    timesteps, block_keys = plan.table_layout()
    config = {
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "hidden_size": 8,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 16,
        "in_channels": 2,
        "audio_in_channels": 2,
        "patch_size": (1, 1, 1),
        "text_dim": 8,
        "freq_dim": 4,
        "time_embed_hidden_dim": 8,
        "time_embed_dim": 4,
        "rope_freq_dim": 1,
    }
    torch.manual_seed(7)
    full = MiniMaxH3Transformer3DModel(**config).eval()
    pruned = AdaLNPrunedMiniMaxH3Transformer.from_official_config(
        config,
        table_timesteps=timesteps,
        table_block_keys=block_keys,
    ).eval()
    pruned_twin = AdaLNPrunedMiniMaxH3Transformer.from_official_config(
        config,
        table_timesteps=timesteps,
        table_block_keys=block_keys,
    ).eval()
    _validate_dual_dit_topology({"fl2va": pruned, "ref2va": pruned_twin})
    observe("two weight instances share one exact DiT class and topology")
    refusal(
        "two task component names cannot alias one DiT instance",
        lambda: _validate_dual_dit_topology({"fl2va": pruned, "ref2va": pruned}),
        "artifact_config",
    )
    check(
        "AdaLN-pruned DiT inherits the official forward",
        "forward" in AdaLNPrunedMiniMaxH3Transformer.__dict__,
        False,
    )

    pruned_state = pruned.state_dict()
    full_state = full.state_dict()
    common = {
        name: value
        for name, value in full_state.items()
        if name in pruned_state and pruned_state[name].shape == value.shape
    }
    loaded = pruned.load_state_dict(common, strict=False)
    check(
        "no official destination becomes an unexpected AdaLN-pruned key",
        loaded.unexpected_keys,
        [],
    )
    check(
        "AdaLN-pruned construction omits every dynamic modulation destination",
        any(
            name.startswith("time_embedder.")
            or ".adaln_proj.linear." in name
            or name.startswith("norm_out.linear.")
            for name in pruned_state
        ),
        False,
    )
    expected_tables = {
        *(f"transformer_blocks.{index}.adaln_proj.table" for index in range(2)),
        "norm_out.table",
    }
    check(
        "only exact tables remain to fill after shared weights",
        set(loaded.missing_keys),
        expected_tables,
    )

    with torch.no_grad():
        all_timesteps = torch.tensor(timesteps, dtype=torch.float32)
        full_temb = full.time_embedder(full.time_proj(all_timesteps))
        sparse_rows = torch.tensor(
            [timestep_row * 3 + modality_tag for timestep_row, modality_tag in block_keys]
        )
        for full_block, table_block in zip(
            full.transformer_blocks, pruned.transformer_blocks, strict=True
        ):
            dense = torch.stack(full_block.adaln_proj(full_temb), dim=1)
            table_block.adaln_proj.table.copy_(dense.index_select(0, sparse_rows))
        final = full.norm_out.linear(torch.nn.functional.silu(full_temb)).reshape(
            len(timesteps), 2, config["hidden_size"]
        )
        pruned.norm_out.table.copy_(final)

    document = json.loads(plan.canonical_bytes())
    evaluations = [row for schedule in document["schedules"] for row in schedule["evaluations"]]
    for evaluation in evaluations:
        classes = evaluation["modulation_classes"]
        local = sorted({float.fromhex(row["timestep"]) for row in classes})
        local_tensor = torch.tensor(local, dtype=torch.float32)
        full_local_temb = full.time_embedder(full.time_proj(local_tensor))
        table_rows = pruned.time_embedder(pruned.time_proj(local_tensor))
        for full_block, table_block in zip(
            full.transformer_blocks, pruned.transformer_blocks, strict=True
        ):
            full_values = full_block.adaln_proj(full_local_temb)
            table_values = table_block.adaln_proj(table_rows)
            for row in classes:
                index = local.index(float.fromhex(row["timestep"])) * 3 + row["modality_tag"]
                for full_value, table_value in zip(full_values, table_values, strict=True):
                    if not torch.allclose(
                        full_value[index], table_value[index], rtol=2e-6, atol=2e-7
                    ):
                        fail(
                            "all canonical block rows equal the dynamic fixture",
                            f"evaluation={evaluation['index']} row={row['name']}",
                        )
                        return
    observe("all canonical block rows equal the dynamic fixture", f"{len(evaluations)} evaluations")

    local = torch.tensor([0.0, _as_float32(0.999)], dtype=torch.float32)
    full_temb = full.time_embedder(full.time_proj(local))
    table_rows = pruned.time_embedder(pruned.time_proj(local))
    hidden = torch.randn(1, 4, config["hidden_size"])
    indices = torch.tensor([0, 0, 1, 0])
    check(
        "final normalization table equals the dynamic fixture",
        torch.allclose(
            full.norm_out(hidden, full_temb, indices),
            pruned.norm_out(hidden, table_rows, indices),
            rtol=2e-6,
            atol=2e-7,
        ),
        True,
    )
    torch.manual_seed(11)
    forward = {
        "hidden_states": torch.randn(1, 2, 2),
        "audio_hidden_states": torch.randn(1, 1, 2),
        "encoder_hidden_states": torch.randn(1, 1, 8),
        "timestep": local,
        "timestep_indices": indices,
        "token_tags": torch.tensor([1, 0, 0, 2]),
        "position_ids": torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        "video_indices": torch.tensor([1, 2]),
        "audio_indices": torch.tensor([3]),
        "text_indices": torch.tensor([0]),
        "return_dict": False,
    }
    full_output = full(**forward)
    numerical = _NumericalTelemetry()
    checks = NumericalChecks(cast(Any, numerical))
    checks.component("actual_h3", full)
    with checks.forwards(full, "actual_h3"):
        checked_output = full(**forward)
    check(
        "actual Diffusers H3 output PyTree is checked without changing values",
        all(torch.equal(a, b) for a, b in zip(full_output, checked_output, strict=True)),
        True,
    )
    first = next(row for row in numerical.rows if row.get("stage") == "actual_h3.prediction.0")
    check("actual H3 tuple exposes both predictions", first["tensors"], 2)
    saved = full.proj_in.bias.detach().clone()

    def poisoned_step() -> None:
        with checks.forwards(full, "actual_h3"):
            full(**forward)

    try:
        with torch.no_grad():
            full.proj_in.bias.fill_(float("nan"))
        refusal(
            "a later actual H3 prediction refuses when its loop settles",
            poisoned_step,
            "numerical_nonfinite",
        )
    finally:
        with torch.no_grad():
            full.proj_in.bias.copy_(saved)
    check(
        "actual H3 diagnostic hooks close after refusal",
        (len(full._forward_pre_hooks), len(full._forward_hooks)),
        (0, 0),
    )
    pruned_output = pruned(**forward)
    check(
        "the inherited forward changes only the AdaLN-pruned modulation source",
        all(
            torch.allclose(left, right, rtol=2e-5, atol=2e-6)
            for left, right in zip(full_output, pruned_output, strict=True)
        ),
        True,
    )
    refusal(
        "a non-plan timestep refuses rather than interpolating",
        lambda: pruned.time_proj(torch.tensor([0.123456], dtype=torch.float32)),
        "artifact_config",
    )
    refusal(
        "a plan timestep paired with an uncovered modality refuses",
        lambda: pruned(
            hidden_states=torch.empty(1, 0, 2),
            audio_hidden_states=torch.empty(1, 0, 2),
            encoder_hidden_states=torch.empty(1, 1, 8),
            timestep=torch.tensor([_as_float32(0.999)]),
            timestep_indices=torch.tensor([0]),
            token_tags=torch.tensor([1]),
            position_ids=torch.zeros(1, 3),
            video_indices=torch.empty(0, dtype=torch.int64),
            audio_indices=torch.empty(0, dtype=torch.int64),
            text_indices=torch.tensor([0]),
            return_dict=False,
        ),
        "artifact_config",
    )


def arm_processor() -> None:
    print("\n== five-file processor closure ==")
    for relative, expected in ASSET_DIGESTS.items():
        check(relative, hashlib.sha256((H3 / relative).read_bytes()).hexdigest(), expected)
    tokenizer, processor = _processor()
    check("tokenizer vocabulary size", tokenizer.vocab_size, 151643)
    check("tokenizer total size", len(tokenizer), 151676)
    check("added-token count", len(tokenizer.added_tokens_decoder), 33)
    check("image token", processor.image_token_id, 151655)
    check("video token", processor.video_token_id, 151656)
    check("vision start token", processor.vision_start_token_id, 151652)
    check("vision end token", processor.vision_end_token_id, 151653)

    corpus = [
        "",
        "a red sports car driving fast along a coastal road at sunset",
        "  leading and trailing  ",
        "line one\nline\ttwo",
        "emoji: 🐈‍⬛🚀🎬",
        "中文参考视频与音频",
        "العربية हिन्दी русский",
        "e\\u0301 vs é",
        "<|im_start|>user\nhello<|im_end|>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        "\x00\x01 control",
        "punctuation !?—… “quotes”",
        "spaces\u00a0\u2003\u3000end",
        "A" * 4096,
    ]
    rows = []
    for split_special_tokens in (False, True):
        tokenizer.split_special_tokens = split_special_tokens
        for text in corpus:
            encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=True)
            rows.append(
                [split_special_tokens, text, encoded["input_ids"], encoded["attention_mask"]]
            )
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    check("hostile tokenizer corpus", digest, TOKEN_CORPUS_DIGEST)
    red("empty-vocabulary regression", digest, hashlib.sha256(b"[]").hexdigest())


def _audio(seconds: int, *, rate: int = 4, start: Fraction = Fraction(0)) -> Any:
    from cozy_runtime.author import DecodedAudio

    samples = seconds * rate
    return DecodedAudio(
        channels=1,
        sample_count=samples,
        sample_rate=rate,
        channel_layout="mono",
        channel_names=("FC",),
        pcm_f32le=(struct.pack(f"<{samples}f", *range(samples)),),
        start_time=start,
    )


def _video(seconds: int, *, soundtrack: Any | None = None) -> Any:
    from cozy_runtime.author import DecodedVideo

    frames = tuple(bytes([index, 0, 0]) for index in range(seconds))
    return DecodedVideo(
        width=1,
        height=1,
        frame_count=seconds,
        frames_rgb=frames,
        frame_pts=tuple(range(seconds)),
        frame_durations=(1,) * seconds,
        time_base=Fraction(1),
        pixel_aspect_ratio=Fraction(1),
        soundtrack=soundtrack,
    )


def arm_media() -> None:
    import torch
    from cozy_runtime.author import AudioAsset, ImageAsset, VideoAsset
    from cozy_runtime.author.fakes import fake_attempt, fake_telemetry

    print("\n== ordered mixed references, exact clocks, and continuation identity ==")
    check(
        "maximum mixed reference policy",
        validate_reference_policy(["image"] * 9 + ["video"] * 3).total,
        12,
    )
    check(
        "three standalone audio with a visual",
        validate_reference_policy(["image", "audio", "audio", "audio"]).audios,
        3,
    )
    refusal("audio cannot stand alone", lambda: validate_reference_policy(["audio"]))
    refusal("ten images refuse", lambda: validate_reference_policy(["image"] * 10))
    refusal("four videos refuse", lambda: validate_reference_policy(["video"] * 4))
    refusal(
        "four standalone audio refuse",
        lambda: validate_reference_policy(["image"] + ["audio"] * 4),
    )
    check("square image demand", reference_image_vision_tokens(2048, 2048), 4096)
    check("4:1 image demand", reference_image_vision_tokens(4000, 1000), 16384)
    check(
        "15-second 16:9 video demand",
        reference_video_vision_tokens(1920, 1080, Fraction(15)),
        15120,
    )
    package._validate_vision_budget(32768)
    observe("vision capacity boundary")
    refusal(
        "vision demand above the release budget refuses",
        lambda: package._validate_vision_budget(MAX_CONDITIONER_VISION_TOKENS + 1),
        "reference_policy",
    )

    video = _video(2, soundtrack=_audio(1, start=Fraction(1, 2)))
    frames = _video_at_24fps(video)
    check("two exact seconds resample to 48 frames", tuple(frames.shape), (48, 3, 1, 1))
    aligned = _aligned_soundtrack(video)
    assert aligned is not None
    check("soundtrack aligns to the full video origin", tuple(aligned.shape), (1, 8))
    check(
        "half-second audio offset becomes two silent samples",
        aligned[0, :2].tolist(),
        [0.0, 0.0],
    )

    class Pipe:
        @staticmethod
        def video_reference(value: Any) -> Any:
            return value

        @staticmethod
        def audio_reference(value: Any) -> Any:
            return value

    class Decoder:
        def __init__(self, videos: list[Any], audios: list[Any]) -> None:
            self._videos = list(videos)
            self._audios = list(audios)

        def decode_video(self, asset: Any) -> Any:
            del asset
            return self._videos.pop(0)

        def decode_audio(self, asset: Any) -> Any:
            del asset
            return self._audios.pop(0)

    def decode(references: list[Any], videos: list[Any], audios: list[Any]) -> list[Any]:
        return package._decode_references(
            cast(Any, references),
            decoder=cast(Any, Decoder(videos, audios)),
            pipe=cast(Any, Pipe()),
        )

    video_ref = package.VideoReference(VideoAsset("sha256:" + "1" * 64))
    audio_ref = package.AudioReference(AudioAsset("sha256:" + "2" * 64))
    check(
        "14s of soundtracked video plus 2s standalone audio fit their separate caps",
        len(
            decode(
                [video_ref, video_ref, audio_ref],
                videos=[_video(7, soundtrack=_audio(7)), _video(7, soundtrack=_audio(7))],
                audios=[_audio(2)],
            )
        ),
        3,
    )
    check(
        "both reference modalities admit their exact independent 15-second boundary",
        len(
            decode(
                [video_ref, video_ref, audio_ref, audio_ref],
                videos=[
                    _video(10, soundtrack=_audio(10)),
                    _video(5, soundtrack=_audio(5)),
                ],
                audios=[_audio(10), _audio(5)],
            )
        ),
        4,
    )
    refusal(
        "soundtracks ride the video cap: 16 seconds of soundtracked video refuse as video",
        lambda: decode(
            [video_ref, video_ref],
            videos=[_video(8, soundtrack=_audio(8)), _video(8, soundtrack=_audio(8))],
            audios=[],
        ),
        "reference_policy",
    )
    refusal(
        "16 seconds of standalone audio refuse against the audio modality cap",
        lambda: decode(
            [video_ref, audio_ref, audio_ref, audio_ref],
            videos=[_video(2)],
            audios=[_audio(6), _audio(6), _audio(4)],
        ),
        "reference_policy",
    )

    decoded = torch.tensor(
        [
            [
                [[[-0.1, 0.49]], [[0.5, 1.1]], [[0.0, 1.0]]],
                [[[1.0, 0.0]], [[0.25, 0.75]], [[0.1, 0.9]]],
            ]
        ]
    )
    pixels = package._rgb8(torch, decoded)
    check("RGB8 conversion shape", tuple(pixels.shape), (2, 1, 2, 3))
    check("RGB8 clamp and round", pixels[0].flatten().tolist(), [0, 128, 0, 125, 255, 255])
    continuation = bytes(pixels[-1].numpy().tobytes())
    check(
        "continuation is the last pre-encode RGB frame",
        continuation,
        bytes([255, 64, 26, 0, 191, 230]),
    )

    class FinishModel:
        pipe = SimpleNamespace(sample_rate=32000)

        @staticmethod
        def decode_audio(task: Any, state: Any, *, checks: Any = None) -> tuple[Any, int]:
            del task
            return state.audio, 32000

        @staticmethod
        def decode_video(task: Any, state: Any, *, checks: Any = None) -> Any:
            del task
            return state.video

    class FinishOutputs:
        def __init__(self) -> None:
            self.continuation = b""
            self.video_pixels = b""
            self.video_audio = b""
            self.fps = 0
            self.sample_rate = 0

        def save_video(self, pixels: Any, **kwargs: Any) -> VideoAsset:
            self.video_pixels = bytes(pixels.numpy())
            self.video_audio = bytes(kwargs["audio"].numpy())
            self.fps = kwargs["fps"]
            self.sample_rate = kwargs["sample_rate"]
            digest = hashlib.sha256(self.video_pixels).hexdigest()
            return VideoAsset(f"sha256:{digest}")

        def save_image(self, frame: Any, *, format: str) -> ImageAsset:
            check("continuation encoder format", format, "png")
            self.continuation = frame.rgb
            digest = hashlib.sha256(frame.rgb).hexdigest()
            return ImageAsset(f"sha256:{digest}")

    finish_video = torch.zeros((1, FRAMES, 3, 2, 3), dtype=torch.float32)
    finish_video[0, -1] = torch.tensor(
        [
            [[1.0, 0.0, 0.25], [0.5, 0.75, 0.0]],
            [[0.5, 1.0, 0.0], [0.25, 0.75, 1.0]],
            [[0.25, 0.5, 1.0], [0.0, 0.75, 0.5]],
        ]
    )
    expected_continuation = bytes(
        [255, 128, 64, 0, 255, 128, 64, 0, 255, 128, 64, 0, 191, 191, 191, 0, 255, 128]
    )
    finish_audio = torch.tensor([[[0.0, 0.25, -0.25, 0.5]]], dtype=torch.float32)
    schedule = ScheduleFacts("a" * 64, 30, 31, *[character * 64 for character in "bcde"])
    finish_outputs = FinishOutputs()
    attempt = fake_attempt("h3-finish-receipt")
    telemetry = fake_telemetry(attempt)
    package_module = cast(Any, package)
    original_gate = package_module.pre_encode_gate
    package_module.pre_encode_gate = lambda *args, **kwargs: ["quality fixture warning"]
    try:
        finished = package._finish(
            cast(Any, FinishModel()),
            "fl2va",
            SimpleNamespace(audio=finish_audio, video=finish_video),
            schedule,
            mute=False,
            out=cast(Any, finish_outputs),
            tel=telemetry,
            cancel=lambda: None,
        )
    finally:
        package_module.pre_encode_gate = original_gate
    check(
        "finish returns exactly two typed media assets",
        (type(finished.video), type(finished.continuation_frame)),
        (VideoAsset, ImageAsset),
    )
    check(
        "finish retains quality warnings with encoded media",
        finished.warnings,
        ["quality fixture warning"],
    )
    check(
        "finish continuation preserves the final pre-encode pixel",
        finish_outputs.continuation,
        expected_continuation,
    )
    red(
        "first-frame continuation regression",
        bytes(finish_outputs.video_pixels[: len(expected_continuation)]),
        finish_outputs.continuation,
    )
    check(
        "finish passes the exact soundtrack and clocks to the video encoder",
        (finish_outputs.video_audio, finish_outputs.fps, finish_outputs.sample_rate),
        (bytes(finish_audio[0].numpy()), FPS, 32000),
    )
    log_events = [event for event in telemetry.events if event.kind == "log"]
    check(
        "finish emits exactly the three ordered proof rows",
        [event.name for event in log_events],
        ["h3 output geometry", "h3 schedule facts", "h3 source digests"],
    )
    logs = {event.name: dict(event.fields) for event in log_events}
    check(
        "Runtime-admitted output geometry receipt",
        logs.get("h3 output geometry"),
        {"width": 3, "height": 2, "frames": FRAMES, "fps": FPS, "sample_rate": 32000},
    )
    check(
        "Runtime-admitted schedule receipt",
        logs.get("h3 schedule facts"),
        {
            "timestep_plan_digest": "a" * 64,
            "video_sigma_digest": "b" * 64,
            "audio_sigma_digest": "c" * 64,
            "video_timestep_digest": "d" * 64,
            "audio_timestep_digest": "e" * 64,
            "sigma_grid_points": 31,
            "transformer_evaluations": 30,
        },
    )
    expected_pixels = finish_outputs.video_pixels
    expected_audio = finish_outputs.video_audio
    check(
        "Runtime-admitted source digest receipt",
        logs.get("h3 source digests"),
        {
            "video_pixel_digest": hashlib.sha256(expected_pixels).hexdigest(),
            "audio_sample_digest": hashlib.sha256(expected_audio).hexdigest(),
            "continuation_pixel_digest": hashlib.sha256(finish_outputs.continuation).hexdigest(),
        },
    )
    check(
        "finish receipt loses and refuses no Runtime observations",
        (attempt.ring.dropped, attempt.ring.refused),
        (0, 0),
    )

    check("single-shot frame cell", (FRAMES, FPS), (345, 24))
    check("eight-shot de-duplicated frame count", 8 * FRAMES - 7, 2753)
    check("eight-shot exact duration", Fraction(8 * FRAMES - 7, FPS), Fraction(2753, 24))


class _NumericalTelemetry:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log(self, message: str, **fields: Any) -> None:
        self.rows.append({"message": message, **fields})


def arm_numerics() -> None:
    import torch
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep
    from torch.utils._python_dispatch import TorchDispatchMode

    print("\n== first non-finite numerical boundary ==")
    telemetry = _NumericalTelemetry()
    checks = NumericalChecks(cast(Any, telemetry))
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    checks.tensors("condition_text", [("prompt_embeds", original)], required=True)
    check(
        "noncontiguous conditioning stays unchanged",
        original.tolist(),
        torch.arange(12).reshape(3, 4).T.tolist(),
    )
    check("absmax names its source", telemetry.rows[-1]["absmax_tensor"], "prompt_embeds")

    # Real stored FP8 supports conversion but not every observer reduction. Track
    # actual casts so a whole-weight float32 copy cannot pass the bounded proof.
    class CastSizes(TorchDispatchMode):  # type: ignore[misc]  # Torch is absent in static CI.
        def __init__(self) -> None:
            super().__init__()
            self.elements: list[int] = []

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if func == torch.ops.aten._to_copy.default and args[0].dtype == torch.float8_e4m3fn:
                self.elements.append(args[0].numel())
            return func(*args, **(kwargs or {}))

    fp8 = torch.ones((257, 4097), dtype=torch.float32).to(torch.float8_e4m3fn).T
    original_bytes = fp8.view(torch.uint8).clone()
    scalar = torch.tensor(-448.0).to(torch.float8_e4m3fn)
    with CastSizes() as casts:
        checks.tensors("fp8", [("encoded", fp8), ("scalar", scalar)], required=True)
    check("FP8 finite scan names the scalar maximum", telemetry.rows[-1]["absmax_tensor"], "scalar")
    check("FP8 finite scan preserves exact maximum", telemetry.rows[-1]["absmax"], 448.0)
    check("FP8 finite scan counts all elements", telemetry.rows[-1]["elements"], fp8.numel() + 1)
    check(
        "FP8 observer does not mutate stored bytes",
        torch.equal(fp8.view(torch.uint8), original_bytes),
        True,
    )
    check("FP8 casts cover exactly the observed values", sum(casts.elements), fp8.numel() + 1)
    check("FP8 float32 scratch is at most four MiB", max(casts.elements) <= 1024 * 1024, True)
    poisoned = torch.tensor([1.0, float("nan")]).to(torch.float8_e4m3fn)
    poisoned_bytes = poisoned.view(torch.uint8).clone()
    refusal(
        "FP8 NaN is refused by the numerical observer",
        lambda: checks.tensors("fp8-nan", [("encoded", poisoned)], required=True),
        "numerical_nonfinite",
    )
    check("FP8 NaN count is exact", telemetry.rows[-1]["chunk_nonfinite"], 1)
    check(
        "FP8 failed scan leaves stored bytes unchanged",
        torch.equal(poisoned.view(torch.uint8), poisoned_bytes),
        True,
    )
    large = torch.zeros(16 * 1024 * 1024 + 1, dtype=torch.float32)
    large[-1] = float("nan")
    refusal(
        "large tensors are checked in bounded chunks",
        lambda: checks.tensors("large", [("weight", large)], required=True),
        "numerical_nonfinite",
    )
    check(
        "observed scan chunk is bounded",
        telemetry.rows[-1]["chunk_elements"] <= 16 * 1024 * 1024,
        True,
    )
    refusal(
        "empty output is not called validated",
        lambda: checks.tensors("empty", [("x", None)], required=True),
        "numerical_uninspectable",
    )
    refusal(
        "opaque output cannot hide tensors",
        lambda: checks.tensors("opaque", [("x", SimpleNamespace(tensor=original))], required=True),
        "numerical_uninspectable",
    )

    module = torch.nn.Linear(4, 4)
    module.register_buffer("derived_rotary", torch.tensor([float("nan")]), persistent=False)
    refusal(
        "nonpersistent resident buffer is inspected",
        lambda: checks.component("rotary", module),
        "numerical_nonfinite",
    )
    check("bad resident buffer is named", telemetry.rows[-1]["tensor"], "derived_rotary")
    module.derived_rotary.fill_(1)
    checks.component("rotary", module)

    value = torch.ones(1, 4)
    baseline = module(value)
    with checks.forwards(module, "linear"):
        check("first checked forward is exact", torch.equal(module(value), baseline), True)
        check("second checked forward is exact", torch.equal(module(value), baseline), True)
    check(
        "each forward is checked",
        [
            r["stage"]
            for r in telemetry.rows
            if str(r.get("stage", "")).startswith("linear.prediction")
        ],
        ["linear.prediction.0", "linear.prediction.1"],
    )

    def bad_input() -> None:
        with checks.forwards(module, "bad_input"):
            module(torch.full((1, 4), float("inf")))

    refusal("invalid input refuses when its loop settles", bad_input, "numerical_nonfinite")
    check(
        "input refusal removes both hooks",
        (len(module._forward_pre_hooks), len(module._forward_hooks)),
        (0, 0),
    )

    def fail_forward(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("existing forward refusal")

    foreign = module.register_forward_pre_hook(fail_forward)
    try:
        refusal("original forward exception survives", bad_input, "RuntimeError")
        check(
            "only pre-existing hook remains",
            (len(module._forward_pre_hooks), len(module._forward_hooks)),
            (1, 0),
        )
    finally:
        foreign.remove()

    original_register = module.register_forward_hook
    module.register_forward_hook = fail_forward
    try:
        refusal("second hook registration failure cleans first", bad_input, "RuntimeError")
        check(
            "registration failure leaves no diagnostic hooks",
            (len(module._forward_pre_hooks), len(module._forward_hooks)),
            (0, 0),
        )
    finally:
        module.register_forward_hook = original_register

    # Execute the real official scheduler with finite extremes: its update overflows.
    video_scheduler, audio_scheduler = MiniMaxH3Scheduler(), MiniMaxH3Scheduler()
    for scheduler in (video_scheduler, audio_scheduler):
        scheduler.set_timesteps(30)
    maximum = torch.finfo(torch.float32).max
    state = SimpleNamespace(
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        noise_pred=torch.full((1, 1, 1), maximum),
        audio_noise_pred=torch.zeros((1, 1, 1)),
        latents=torch.full((1, 1), maximum),
        audio_latents=torch.zeros((1, 1)),
        audio_timesteps=audio_scheduler.timesteps,
    )
    checks.tensors(
        "scheduler.predictions",
        [("video", state.noise_pred), ("audio", state.audio_noise_pred)],
        required=True,
    )
    MiniMaxH3LoopSchedulerStep()(
        SimpleNamespace(scheduler=video_scheduler, audio_scheduler=audio_scheduler),
        state,
        i=0,
        t=video_scheduler.timesteps[0],
    )
    refusal(
        "actual scheduler overflow stops at updated latents",
        lambda: checks.tensors(
            "ref2va_dit.updated.0",
            [("latents", state.latents), ("audio_latents", state.audio_latents)],
            required=True,
        ),
        "numerical_nonfinite",
    )


def arm_resident_fill() -> None:
    import torch
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep
    from torch.utils._python_dispatch import TorchDispatchMode

    print("\n== resident weights are scanned once per fill ==")

    class Kernels(TorchDispatchMode):  # type: ignore[misc]  # Torch is absent in static CI.
        """Every dispatched operator; a reused verdict must launch none at all."""

        def __init__(self) -> None:
            super().__init__()
            self.dispatched = 0

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            self.dispatched += 1
            return func(*args, **(kwargs or {}))

    def request(resident: ResidentWeights, module: Any) -> tuple[dict[str, Any], int]:
        """One request: a fresh NumericalChecks over the verdicts the pipeline keeps."""
        telemetry = _NumericalTelemetry()
        with Kernels() as kernels:
            NumericalChecks(cast(Any, telemetry), resident).component("dit", module)
        return telemetry.rows[-1], kernels.dispatched

    def tensors(module: Any) -> list[tuple[str, Any]]:
        return [*module.named_parameters(), *module.named_buffers()]

    def restage(module: Any, values: dict[str, Any]) -> None:
        """Exactly Runtime's shape: park on meta, reserve fresh storage, fill in place."""
        module.to_empty(device="meta")
        module.to_empty(device="cpu")
        with torch.no_grad():
            for name, tensor in tensors(module):
                tensor.view(-1).view(torch.uint8).copy_(values[name].view(-1).view(torch.uint8))

    module = torch.nn.Linear(4, 4)
    module.register_buffer("rotary", torch.ones(2), persistent=False)
    weights = {name: tensor.detach().clone() for name, tensor in tensors(module)}
    resident = ResidentWeights()
    row, scanned = request(resident, module)
    check(
        "the first request scans the fill", (row["status"], scanned > 0), ("finite_resident", True)
    )
    check("the first request counts every resident value", row["elements"], 16 + 4 + 2)
    row, scanned = request(resident, module)
    check("the second request reuses the verdict", (row["status"], scanned), ("verified_fill", 0))
    check("the reused verdict names the verified tensors", row["tensors"], 3)
    red("a request-local registry rescans", request(ResidentWeights(), module)[1], 0)

    restage(module, weights)
    row, scanned = request(resident, module)
    check(
        "a re-staged fill is scanned again", (row["status"], scanned > 0), ("finite_resident", True)
    )
    row, scanned = request(resident, module)
    check("the re-staged verdict is reused", (row["status"], scanned), ("verified_fill", 0))

    # A paged block parks by resizing its storage to zero and refills the same storage.
    module.weight.untyped_storage().resize_(0)
    row, scanned = request(resident, module)
    check("a parked weight is neither claimed nor scanned", (row["tensors"], scanned), (2, 0))
    module.weight.untyped_storage().resize_(weights["weight"].numel() * 4)
    with torch.no_grad():
        module.weight.view(-1).view(torch.uint8).copy_(weights["weight"].view(-1).view(torch.uint8))
    row, scanned = request(resident, module)
    check("a refilled storage is rescanned alone", (row["elements"], scanned > 0), (16, True))

    poisoned = dict(weights, weight=weights["weight"].clone())
    poisoned["weight"][0, 0] = float("nan")
    restage(module, poisoned)
    refusal(
        "non-finite weights are refused after a re-stage",
        lambda: request(resident, module),
        "numerical_nonfinite",
    )
    refusal(
        "a refused fill leaves no verdict to reuse",
        lambda: request(resident, module),
        "numerical_nonfinite",
    )

    print("\n== step checks settle once, after the loop ==")

    class Sampler(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.poison_at = -1

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return value * float("nan") if self.calls - 1 == self.poison_at else value + 1

    telemetry = _NumericalTelemetry()
    checks = NumericalChecks(cast(Any, telemetry))
    sampler = Sampler()

    def loop() -> None:
        latents = torch.ones(1, 4)
        with checks.forwards(sampler, "dit"):
            for index in range(3):
                latents = sampler(latents)
                checks.tensors(
                    f"dit.updated.{index}", [("latents", latents)], required=True, defer=True
                )
            check("nothing is read inside the loop", len(telemetry.rows), 0)

    loop()
    stages = [
        f"dit.{stage}.{index}" for index in range(3) for stage in ("input", "prediction", "updated")
    ]
    check("every step settles in step order", [row["stage"] for row in telemetry.rows], stages)
    check("the settled loop names its maximum", telemetry.rows[-1]["absmax"], 4.0)

    telemetry.rows.clear()
    sampler.calls, sampler.poison_at = 0, 1
    refusal("a planted NaN in a step output fails the attempt", loop, "numerical_nonfinite")
    check("the first non-finite stage is named", telemetry.rows[-1]["stage"], "dit.prediction.1")
    check(
        "the stages before it settled first",
        [row["stage"] for row in telemetry.rows[:-1]],
        stages[:4],
    )
    check("a refusal leaves nothing pending", checks._pending, [])
    check(
        "the loop's hooks close after refusal",
        (len(sampler._forward_pre_hooks), len(sampler._forward_hooks)),
        (0, 0),
    )

    # The official update is an affine blend with no clamp: a step-0 NaN reaches the end.
    video, audio = MiniMaxH3Scheduler(), MiniMaxH3Scheduler()
    for scheduler in (video, audio):
        scheduler.set_timesteps(DEFAULT_STEPS)
    state = SimpleNamespace(
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        audio_noise_pred=torch.zeros((1, 1, 1)),
        latents=torch.ones((2, 1)),
        audio_latents=torch.ones((1, 1)),
        audio_timesteps=audio.timesteps,
    )
    components = SimpleNamespace(scheduler=video, audio_scheduler=audio)
    for index, timestep in enumerate(video.timesteps):
        state.noise_pred = torch.zeros((1, 2, 1))
        if index == 0:
            state.noise_pred[0, 1, 0] = float("nan")
        MiniMaxH3LoopSchedulerStep()(components, state, i=index, t=timestep)
    check(
        "the official update carries a step-0 NaN into the final latents",
        torch.isfinite(state.latents).flatten().tolist(),
        [True, False],
    )


def arm_output_gates() -> None:
    import torch
    from cozy_runtime.author.fakes import fake_telemetry

    print("\n== structural refusals and quality observations ==")
    requested = MediaFacts(width=1, height=1, frames=2, fps=24, sample_rate=24, mute=False)
    decoded = torch.zeros((1, 2, 3, 1, 1), dtype=torch.float32)
    pixels = torch.zeros((2, 1, 1, 3), dtype=torch.uint8)
    waveform = torch.zeros((1, 2), dtype=torch.float32)
    poisoned = decoded.clone()
    poisoned[0, 0, 0, 0, 0] = float("nan")
    refusal(
        "NaN is observed before RGB8 erases it",
        lambda: pre_encode_gate(
            torch,
            pixels=pixels,
            waveform=waveform,
            video_nonfinite_fraction=package._nonfinite_fraction(torch, poisoned),
            audio_nonfinite_fraction=0.0,
            requested=requested,
            tel=fake_telemetry(),
        ),
        "output_integrity",
    )
    check(
        "the uint8 control contains no NaN evidence",
        bool(torch.isnan(pixels.float()).any()),
        False,
    )
    refusal(
        "audio outside the one-frame A/V tolerance refuses",
        lambda: pre_encode_gate(
            torch,
            pixels=pixels,
            waveform=torch.zeros((1, 5)),
            video_nonfinite_fraction=0.0,
            audio_nonfinite_fraction=0.0,
            requested=requested,
            tel=fake_telemetry(),
        ),
        "output_integrity",
    )
    # A real quality rejection must remain visible without suppressing an
    # otherwise encodable inference result. Checkpoints are qualified separately.
    requested = MediaFacts(width=512, height=512, frames=5, fps=24, sample_rate=240, mute=True)
    pixels = torch.full((5, 512, 512, 3), 100, dtype=torch.uint8)
    pixels[:, ::16] = 220
    warnings = pre_encode_gate(
        torch,
        pixels=pixels,
        waveform=torch.zeros((2, 50)),
        video_nonfinite_fraction=0.0,
        audio_nonfinite_fraction=0.0,
        requested=requested,
        tel=fake_telemetry(),
    )
    check(
        "periodic output is returned with its actual quality rejection",
        len(warnings) == 1 and "GRID:" in warnings[0] and "REJECT" in warnings[0],
        True,
    )


def arm_interface() -> None:
    print("\n== committed public surface ==")
    interface_path = H3 / "metadata" / "package-interface.json"
    interface = json.loads(interface_path.read_text())
    entries = {entry["name"]: entry for entry in interface["entrypoints"]}
    surfaces = {surface.name: surface for surface in describe(package.app)}
    check(
        "exact action names",
        set(entries),
        {"first_last_frame_to_video", "reference_media_to_video"},
    )
    check(
        "both official actions are visible",
        set(entries),
        {"first_last_frame_to_video", "reference_media_to_video"},
    )
    expected = {
        "first_last_frame_to_video": (
            ["prompt", "first_frame", "last_frame", "mute", "seed", "steps"],
            "fl2va_dit",
        ),
        "reference_media_to_video": (
            ["prompt", "references", "mute", "seed", "reference_image_short_edge", "steps"],
            "ref2va_dit",
        ),
    }
    for name, (fields, dit) in expected.items():
        entry = entries[name]
        check(
            f"{name} request fields",
            [field["name"] for field in entry["request"]["fields"]],
            fields,
        )
        check(
            f"{name} steps wire enum is the plan's step set",
            entry["request"]["fields"][-1]["type"],
            {"literal": list(STEPS)},
        )
        check(f"{name} shared model", entry["models"][0]["class"], "H3Model")
        check(f"{name} carries no retired stamps member", "stamps" in entry["models"][0], False)
        check(f"{name} slot admits encoded leaves", entry["models"][0]["encoded_leaves"], "accept")
        component_use = entry["models"][0]["component_use"]
        check(
            f"{name} task DiT lease exists",
            component_use[f"sample_{dit.removesuffix('_dit')}"],
            [dit],
        )
        check(f"{name} media capability", "media_decode" in surfaces[name].capabilities, True)
        check(
            f"{name} exact customer result fields",
            [field["name"] for field in entry["result"]["fields"]],
            ["video", "continuation_frame", "warnings"],
        )
    check("H3 permits Runtime encoded linear leaves", package.H3Model.__encoded_leaves__, "accept")


ARMS = {
    "producer-configs": arm_producer_configs,
    "producer-construction-order": arm_producer_construction_order,
    "schedule": arm_schedule,
    "zero-reference": arm_zero_reference_preparation,
    "reference-resolution": arm_reference_resolution,
    "graph": arm_graph_and_dtypes,
    "conditioner": arm_text_conditioner,
    "adaln-pruned": arm_adaln_pruned,
    "processor": arm_processor,
    "media": arm_media,
    "gates": arm_output_gates,
    "numerics": arm_numerics,
    "resident-fill": arm_resident_fill,
    "interface": arm_interface,
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
