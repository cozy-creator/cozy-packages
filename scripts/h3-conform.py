#!/usr/bin/env python
"""Deterministic MiniMax-H3 contract arms; no weights, GPU, network, or test framework.

Every arm executes the official Diffusers 0.40 implementation or a public package
boundary. Each historically dangerous invariant also carries a negative control. A green
run is a CPU semantic proof, not a generation or accelerator proof.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "h3"
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
    SIGMA_GRID_POINTS,
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
    _validate_row_timestep_plan,
    _video_at_24fps,
    canonical_timestep_plan,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    timestep_plan_digest,
    validate_reference_policy,
)

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0

PLAN_DIGESTS = {
    "fl2va": "8da103b9b09629f9f4bcc7c3311929a83c4bc76d5ac2a49fa8ad6c08a140d99b",
    "ref2va": "f99dec0b673105a6b7cabdc57a62df9653afd943a9092eef6018aa48095a9487",
}
PACKAGE_DESCRIPTOR_DIGEST = (
    "sha256:c3e5db6f1cc08c2caa88b9bc5774a975999a3e0ad4e918f7f49015845043b114"
)
VECTOR_DIGESTS = {
    "video_sigmas": "9908cdf87605da6006148af6e7ef8be63e806d40bf750efcaae24386c4eb4e86",
    "audio_sigmas": "120d46f5ce12fcbeb24f50707cc9f045a0fe283a5b35ad16dbd86f4810fb58e9",
    "video_timesteps": "0aa2dd3bd6d296de7b0ede70932f8cd89a9f21fb10b3ac0144d18738fd0db3c9",
    "audio_timesteps": "10ca44b527cfe2fa7f711d42772654f830037b92c8276e6e4e3bed9df00d4ea7",
}
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
    return {
        "cozy_h3": {
            "task": task,
            "modulation": modulation,
            "timestep_plan_digest": f"sha256:{plan.digest}",
        }
    }


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
        check(f"{task} evaluation count", len(document["evaluations"]), 29)
        check(
            f"{task} complete class names",
            [entry["name"] for entry in document["evaluations"][0]["modulation_classes"]],
            ["target_video", "text", "target_audio", "condition_video", "condition_audio"],
        )
        check(
            f"{task} canonical block-table rows",
            len(document["table_keys"]["block_modulation"]),
            89,
        )
        check(
            f"{task} canonical final-normalization rows",
            len(document["table_keys"]["final_normalization"]),
            59,
        )
        check(
            f"{task} terminal has no forward",
            document["terminal"]["transformer_evaluation"],
            False,
        )
    red("task cannot collide in the plan identity", plans["fl2va"].digest, plans["ref2va"].digest)
    refusal(
        "an extra penultimate sigma cannot hide outside canonical bytes",
        lambda: replace(
            plans["fl2va"],
            video_sigmas=(*plans["fl2va"].video_sigmas[:-1], 0.125, 0.0),
        ),
    )

    document = json.loads(plans["fl2va"].canonical_bytes())
    for index, offsets, final_rows in (
        (0, [0, 1, 2, 3, 8], 3),
        (1, [0, 1, 5, 6, 11], 4),
    ):
        classes = document["evaluations"][index]["modulation_classes"]
        unique = sorted({float.fromhex(entry["timestep"]) for entry in classes})
        got = [
            unique.index(float.fromhex(entry["timestep"])) * 3 + entry["modality_tag"]
            for entry in classes
        ]
        check(f"evaluation {index} exact AdaLN row offsets", got, offsets)
        check(f"evaluation {index} final-normalization row count", len(unique), final_rows)

    for name, shift, ours in (
        ("video", 12.0, plans["fl2va"].video_sigmas),
        ("audio", 3.0, plans["fl2va"].audio_sigmas),
    ):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(SIGMA_GRID_POINTS)
        official_sigmas = tuple(float(value) for value in scheduler.sigmas.float().cpu())
        official_timesteps = tuple(float(value) for value in scheduler.timesteps.float().cpu())
        check(f"{name} sigmas equal Diffusers", ours, official_sigmas)
        check(f"{name} has 30 sigma points", len(official_sigmas), 30)
        check(f"{name} has 29 forwards", len(official_timesteps), 29)
        check(
            f"{name} sigma digest",
            float_digest(official_sigmas),
            VECTOR_DIGESTS[f"{name}_sigmas"],
        )
        check(
            f"{name} timestep digest",
            float_digest(official_timesteps),
            VECTOR_DIGESTS[f"{name}_timesteps"],
        )
        red(f"{name} 31-point interpretation", len(official_sigmas), 31)
        red(f"{name} 30-forward interpretation", len(official_timesteps), 30)

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

    video_indices = torch.tensor([2, 3])
    audio_indices = torch.tensor([4, 5])
    row_plans = [
        MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=video_indices,
            audio_indices=audio_indices,
            num_condition_video_rows=1,
            num_condition_audio_rows=1,
            num_text_tokens=2,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
            condition_video_timestep=max(video_timestep, _as_float32(0.999)),
            condition_audio_timestep=1.0,
        )
        for video_timestep, audio_timestep in zip(
            plans["fl2va"].video_timesteps,
            plans["fl2va"].audio_timesteps,
            strict=True,
        )
    ]
    state = SimpleNamespace(
        row_timestep_plan=row_plans,
        token_tags=torch.tensor([1, 0, 0, 0, 2, 2]),
        text_token_tags=torch.tensor([1, 0]),
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=torch.tensor([0, 1]),
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
    )
    _validate_row_timestep_plan(state, plans["fl2va"])
    observe("mixed ordinary-text and vision-text tags validate")
    wrong_tags = state.token_tags.clone()
    wrong_tags[1] = 1
    refusal(
        "a vision token mislabeled as ordinary text refuses",
        lambda: _validate_row_timestep_plan(
            SimpleNamespace(**{**vars(state), "token_tags": wrong_tags}), plans["fl2va"]
        ),
        "canonical_schedule",
    )

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(30)
    sample = torch.tensor([-1.0])
    velocity = torch.tensor([2.0])
    actual = float(scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0])
    sigma, sigma_next = (float(value) for value in scheduler.sigmas[:2])
    correct = -1.0 + (sigma - sigma_next) * 2.0
    defective = -1.0 + (sigma_next - sigma) * 2.0
    check("official solver moves data-ward", actual, correct)
    red("reversed anti-denoising sign", defective, correct)


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
        transformer = _apply_transformer_dtype(MiniMaxH3Transformer3DModel())
        video_vae = AutoencoderKLMiniMaxH3()
        audio_vae = AutoencoderKLMiniMaxH3Audio()
        wrong_audio_vae = AutoencoderKLMiniMaxH3Audio(sampling_rate=44100)
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
    wrong_plan = dit_config("fl2va")
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
        "scoped_pipeline_write",
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
    full = (
        Qwen3VLForConditionalGeneration(Qwen3VLConfig(**upstream)).to(dtype=torch.bfloat16).eval()
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
            release="se-018-audit",
            application="h3:h3.app",
            hardware_variant="sm90",
            declared_variants=("sm90",),
        )
        return cast(dict[str, object], result.contract.render())

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
    for evaluation in document["evaluations"]:
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
    observe("all canonical block rows equal the dynamic fixture", "29 evaluations")

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
        "adaln_pruned_timestep",
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
        "adaln_pruned_timestep",
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
        "reference_capacity",
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
        "reference_video_duration_total",
    )
    refusal(
        "16 seconds of standalone audio refuse against the audio modality cap",
        lambda: decode(
            [video_ref, audio_ref, audio_ref, audio_ref],
            videos=[_video(2)],
            audios=[_audio(6), _audio(6), _audio(4)],
        ),
        "reference_audio_duration_total",
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
        def decode_audio(task: Any, state: Any) -> tuple[Any, int]:
            del task
            return state.audio, 32000

        @staticmethod
        def decode_video(task: Any, state: Any) -> Any:
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
    schedule = ScheduleFacts(*[character * 64 for character in "abcde"])
    finish_outputs = FinishOutputs()
    attempt = fake_attempt("h3-finish-receipt")
    telemetry = fake_telemetry(attempt)
    package_module = cast(Any, package)
    original_gate = package_module.pre_encode_gate
    package_module.pre_encode_gate = lambda *args, **kwargs: None
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
            "sigma_grid_points": SIGMA_GRID_POINTS,
            "transformer_evaluations": 29,
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


class _Telemetry:
    def metric(self, name: str, value: int | float) -> None:
        del name, value


def arm_output_gates() -> None:
    import torch

    print("\n== pre-encode output refusals ==")
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
            tel=cast(Any, _Telemetry()),
        ),
        "output_integrity_nan",
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
            tel=cast(Any, _Telemetry()),
        ),
        "output_av_duration_mismatch",
    )


def arm_descriptor() -> None:
    print("\n== committed public surface ==")
    descriptor_path = H3 / "package.descriptor.json"
    descriptor = json.loads(descriptor_path.read_text())
    check(
        "descriptor semantic identity",
        canonical_json.digest_bytes(descriptor_path.read_bytes()),
        PACKAGE_DESCRIPTOR_DIGEST,
    )
    check(
        "README launch contract names the current descriptor identity",
        PACKAGE_DESCRIPTOR_DIGEST in (ROOT / "README.md").read_text(),
        True,
    )
    entries = {entry["name"]: entry for entry in descriptor["entrypoints"]}
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
            ["prompt", "first_frame", "last_frame", "mute", "seed"],
            "fl2va_dit",
        ),
        "reference_media_to_video": (
            ["prompt", "references", "mute", "seed"],
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
        check(f"{name} shared model", entry["models"][0]["class"], "H3Model")
        check(f"{name} carries no task-selection stamp", entry["models"][0]["stamps"], {})
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
            ["video", "continuation_frame"],
        )
    check("H3 permits Runtime encoded linear leaves", package.H3Model.__encoded_leaves__, "accept")


def arm_live_probe() -> None:
    print("\n== installed-artifact production probe boundary ==")
    probe = runpy.run_path(str(ROOT / "scripts" / "h3-live.py"))
    command_json = cast(Callable[[list[str]], dict[str, Any]], probe["command_json"])
    load_request = cast(Callable[..., Any], probe["load_request"])
    load_plan_facts = cast(Callable[..., dict[str, dict[str, str]]], probe["load_plan_facts"])
    reserve_output = cast(Callable[[Path], Path], probe["reserve_output"])
    make_requests_read_only = cast(
        Callable[[Path, list[Path]], None], probe["make_requests_read_only"]
    )
    stage_request = cast(Callable[..., Path], probe["stage_request"])
    verify_bindings = cast(Callable[..., list[dict[str, Any]]], probe["verify_bindings"])
    verify_staged_request = cast(Callable[..., None], probe["verify_staged_request"])
    verify_media_contract = cast(Callable[..., None], probe["verify_media_contract"])
    probe_media = cast(Callable[..., dict[str, Any]], probe["probe_media"])
    verify_result = cast(
        Callable[..., tuple[dict[str, Any], dict[str, str]]], probe["verify_result"]
    )
    verify_package_descriptor_digest = cast(
        Callable[..., str], probe["verify_package_descriptor_digest"]
    )
    require_visible = cast(Callable[..., set[str]], probe["require_visible"])
    select_actions = cast(Callable[[list[str] | None], set[str]], probe["select_actions"])
    probe_kind = cast(str, probe["PROBE_KIND"])

    check(
        "Runtime multiline JSON is one document",
        command_json(
            [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'multiline': True}, indent=2))",
            ]
        ),
        {"multiline": True},
    )

    binding_ref = "cozy/minimax-h3@1.0.0"
    binding_lane = "profile=fp8-adaln-pruned"
    snapshot = "sha256:" + "1" * 64
    installed_descriptor = json.loads((H3 / "package.descriptor.json").read_text())
    package_descriptor_digest = canonical_json.digest(installed_descriptor)
    runtime_plan = "sha256:" + "5" * 64
    construction = "sha256:" + "6" * 64
    components = ["audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"]

    def binding(path: str, *, installed: bool = True) -> dict[str, Any]:
        return {
            "model_binding_path": path,
            "ref": binding_ref,
            "lane": binding_lane,
            "installed": installed,
            "components": components,
            "snapshots": dict.fromkeys(components, snapshot),
            "stamps": {"task": ["fl2va", "ref2va"]},
        }

    document = {
        "bindings": [
            binding("first_last_frame_to_video.models.model"),
            binding("reference_media_to_video.models.model"),
        ]
    }
    ref_action = {"reference_media_to_video"}
    check(
        "one installed uniform dual binding arms both actions",
        len(
            verify_bindings(
                document,
                expected_ref=binding_ref,
                expected_checkpoint=snapshot,
                expected_lane=binding_lane,
            )
        ),
        2,
    )
    refusal(
        "a partial binding set refuses even when probing one action",
        lambda: verify_bindings(
            {"bindings": [binding("reference_media_to_video.models.model")]},
            expected_ref=binding_ref,
            expected_checkpoint=snapshot,
            expected_lane=binding_lane,
        ),
        "RuntimeError",
    )
    refusal(
        "an uninstalled action refuses before inference",
        lambda: verify_bindings(
            {
                "bindings": [
                    binding("first_last_frame_to_video.models.model", installed=False),
                    binding("reference_media_to_video.models.model"),
                ]
            },
            expected_ref=binding_ref,
            expected_checkpoint=snapshot,
            expected_lane=binding_lane,
        ),
        "RuntimeError",
    )
    refusal(
        "a different quantization lane refuses before inference",
        lambda: verify_bindings(
            {
                "bindings": [
                    {
                        **binding("first_last_frame_to_video.models.model"),
                        "lane": "profile=mxfp8-adaln-pruned",
                    },
                    binding("reference_media_to_video.models.model"),
                ]
            },
            expected_ref=binding_ref,
            expected_checkpoint=snapshot,
            expected_lane=binding_lane,
        ),
        "RuntimeError",
    )
    refusal(
        "a narrow artifact refuses before inference",
        lambda: verify_bindings(
            {
                "bindings": [
                    {
                        **binding("first_last_frame_to_video.models.model"),
                        "components": components[:-1],
                    },
                    binding("reference_media_to_video.models.model"),
                ]
            },
            expected_ref=binding_ref,
            expected_checkpoint=snapshot,
            expected_lane=binding_lane,
        ),
        "RuntimeError",
    )
    refusal(
        "a different component snapshot refuses before inference",
        lambda: verify_bindings(
            document,
            expected_ref=binding_ref,
            expected_checkpoint="sha256:" + "2" * 64,
            expected_lane=binding_lane,
        ),
        "RuntimeError",
    )
    refusal(
        "a single-task artifact refuses before inference",
        lambda: verify_bindings(
            {
                "bindings": [
                    {
                        **binding("first_last_frame_to_video.models.model"),
                        "stamps": {"task": ["fl2va"]},
                    },
                    binding("reference_media_to_video.models.model"),
                ]
            },
            expected_ref=binding_ref,
            expected_checkpoint=snapshot,
            expected_lane=binding_lane,
        ),
        "RuntimeError",
    )

    check(
        "the described package digest must equal the launch contract",
        verify_package_descriptor_digest(installed_descriptor, expected=package_descriptor_digest),
        package_descriptor_digest,
    )
    check(
        "default proof selection remains both public product actions",
        select_actions(None),
        {"first_last_frame_to_video", "reference_media_to_video"},
    )
    check(
        "explicit Ref2VA proof selection is singular",
        select_actions(["reference_media_to_video"]),
        ref_action,
    )
    check(
        "selecting Ref2VA preserves the complete visible surface",
        require_visible(ref_action, installed_descriptor),
        {"first_last_frame_to_video", "reference_media_to_video"},
    )
    refusal(
        "a different package descriptor refuses before inference",
        lambda: verify_package_descriptor_digest(
            installed_descriptor, expected="sha256:" + "3" * 64
        ),
        "RuntimeError",
    )

    expected_plans = {
        "first_last_frame_to_video": "sha256:" + PLAN_DIGESTS["fl2va"],
        "reference_media_to_video": "sha256:" + PLAN_DIGESTS["ref2va"],
    }
    selected_plan_facts = load_plan_facts(H3, expected_plans)
    check(
        "selected package plan semantics match both launch identities",
        {action: facts["document_digest"] for action, facts in selected_plan_facts.items()},
        expected_plans,
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        selected = root / "selected-package"
        plans = selected / "timestep-plans"
        plans.mkdir(parents=True)
        for task in ("fl2va", "ref2va"):
            (plans / f"{task}.json").write_bytes(
                (H3 / "timestep-plans" / f"{task}.json").read_bytes()
            )
        fl_plan = plans / "fl2va.json"
        fl_document = cast(dict[str, Any], canonical_json.decode(fl_plan.read_bytes()))
        fl_plan.write_text(
            json.dumps(dict(reversed(list(fl_document.items()))), indent=2, ensure_ascii=False)
        )
        check(
            "selected package formatting twin preserves semantic plan identity",
            load_plan_facts(selected, expected_plans)["first_last_frame_to_video"][
                "document_digest"
            ],
            expected_plans["first_last_frame_to_video"],
        )

        fl_plan.write_text(json.dumps(dict(fl_document, frames=fl_document["frames"] + 1)))
        refusal(
            "selected package semantic plan drift cannot fall back to checkout-global plans",
            lambda: load_plan_facts(selected, expected_plans),
            "RuntimeError",
        )
        original = canonical_json.encode(fl_document)
        fl_plan.write_bytes(original[:-1] + b',"task":"other"}')
        refusal(
            "selected package duplicate plan key refuses",
            lambda: load_plan_facts(selected, expected_plans),
            "RuntimeError",
        )
        fl_plan.write_bytes(original[:-1] + b',"bad":NaN}')
        refusal(
            "selected package non-finite plan number refuses",
            lambda: load_plan_facts(selected, expected_plans),
            "RuntimeError",
        )

        def request(name: str, raw: bytes) -> tuple[Path, str]:
            path = root / name
            path.write_bytes(raw)
            return path, f"sha256:{hashlib.sha256(raw).hexdigest()}"

        good_path, good_digest = request(
            "good.json", b'{ "prompt": "proof", "mute": false, "seed": 17 }\n'
        )
        request_snapshot = load_request(good_path, expected_sha256=good_digest)
        check("proof request preserves its exact integer seed", request_snapshot.seed, 17)
        refusal(
            "a wrong raw request digest refuses before inference",
            lambda: load_request(good_path, expected_sha256="sha256:" + "4" * 64),
            "RuntimeError",
        )
        for name, raw in (
            ("missing-seed.json", b'{"prompt":"proof"}'),
            ("null-seed.json", b'{"seed":null}'),
            ("bool-seed.json", b'{"seed":true}'),
        ):
            path, digest = request(name, raw)
            refusal(
                f"{name.removesuffix('.json')} refuses before inference",
                partial(load_request, path, expected_sha256=digest),
                "RuntimeError",
            )
        muted, muted_digest = request("muted.json", b'{"prompt":"proof","mute":true,"seed":17}')
        refusal(
            "muted proof refuses before inference",
            partial(load_request, muted, expected_sha256=muted_digest),
            "RuntimeError",
        )

        existing = root / "existing-proof"
        existing.mkdir()
        refusal(
            "a pre-existing proof root refuses",
            lambda: reserve_output(existing),
            "RuntimeError",
        )
        output = reserve_output(root / "fresh-proof")
        refusal(
            "a proof root is single-use",
            lambda: reserve_output(output),
            "RuntimeError",
        )
        requests = output / "requests"
        requests.mkdir()
        staged = stage_request(requests / "fl2va.json", request_snapshot)
        check("staged request bytes are exact", staged.read_bytes(), request_snapshot.raw)
        check("staged request has no writable mode", staged.stat().st_mode & 0o222, 0)
        refusal(
            "a read-only request cannot be rewritten without changing its mode",
            lambda: staged.write_bytes(b"different"),
            "PermissionError",
        )
        staged.chmod(0o644)
        staged.write_bytes(staged.read_bytes() + b" ")
        staged.chmod(0o444)
        refusal(
            "a chmod-and-mutate staged request refuses",
            lambda: verify_staged_request(staged, request_snapshot),
            "RuntimeError",
        )
        staged.chmod(0o644)
        staged.write_bytes(request_snapshot.raw)
        staged.chmod(0o444)
        make_requests_read_only(requests, [staged])
        verify_staged_request(staged, request_snapshot)
        observe("read-only request identity validates after directory mode change")

        descriptor_bytes = (H3 / "package.descriptor.json").read_bytes()
        descriptor_identity = canonical_json.digest_bytes(descriptor_bytes)
        fake_runtime = root / "cozy-runtime-fake"
        responses = {
            ("describe",): installed_descriptor,
            ("doctor",): {"device": {"type": "cuda", "name": "contract-fake"}},
            ("bindings",): document,
        }
        fake_runtime.write_text(
            f"#!{sys.executable}\n"
            "import json\n"
            "import sys\n"
            f"responses = {responses!r}\n"
            "command = tuple(sys.argv[sys.argv.index('--json') + 1:])\n"
            "if command not in responses:\n"
            "    print(f'unsupported fake Runtime command: {command}', file=sys.stderr)\n"
            "    raise SystemExit(2)\n"
            "print(json.dumps(responses[command]))\n"
        )
        fake_runtime.chmod(0o755)
        live_base = [
            sys.executable,
            str(ROOT / "scripts" / "h3-live.py"),
            "--runtime",
            str(fake_runtime),
            "--package",
            str(H3),
            "--expected-binding-ref",
            binding_ref,
            "--expected-lane",
            binding_lane,
            "--expected-checkpoint",
            snapshot,
            "--expected-package-descriptor-digest",
            descriptor_identity,
            "--expected-fl-plan-digest",
            expected_plans["first_last_frame_to_video"],
            "--expected-ref-plan-digest",
            expected_plans["reference_media_to_video"],
            "--inspect-only",
        ]
        ref_main = subprocess.run(
            [
                *live_base,
                "--action",
                "reference_media_to_video",
                "--out",
                str(root / "main-ref-proof"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        check("selected Ref2VA main path exits green", ref_main.returncode, 0)
        if ref_main.returncode == 0:
            receipt_path = Path(ref_main.stdout.strip().splitlines()[-1])
            receipt = json.loads(receipt_path.read_text())
            check("selected-action receipt kind", receipt["kind"], probe_kind)
            check(
                "selected-action receipt identity",
                receipt["selected_actions"],
                ["reference_media_to_video"],
            )
            check(
                "inspect-only receipt status",
                (
                    receipt["automated_status"],
                    receipt["package_observations_status"],
                ),
                (
                    "runtime-device-observed-and-binding-inspected",
                    "pending-runtime-triage-join",
                ),
            )
        else:
            fail("selected Ref2VA main path receipt", ref_main.stderr[:500])

        fl_main = subprocess.run(
            [
                *live_base,
                "--action",
                "first_last_frame_to_video",
                "--out",
                str(root / "main-fl-proof"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        check("selected FL2VA main path exits green", fl_main.returncode, 0)
        if fl_main.returncode == 0:
            receipt_path = Path(fl_main.stdout.strip().splitlines()[-1])
            receipt = json.loads(receipt_path.read_text())
            check(
                "selected FL2VA receipt identity",
                receipt["selected_actions"],
                ["first_last_frame_to_video"],
            )
        else:
            fail("selected FL2VA main path receipt", fl_main.stderr[:500])

    result = {
        "video": {"digest": "sha256:" + "7" * 64},
        "continuation_frame": {"digest": "sha256:" + "9" * 64},
    }
    accepted_plan = {
        "plan_digest": runtime_plan,
        "model_construction_digest": construction,
        "invocation_spec_digest": "sha256:" + "e" * 64,
        "delivery": "resident",
        "materialization": "installed",
        "placement": "all_resident",
        "compute_dtype": "bfloat16",
        "reserved_device_memory_bytes": 1,
    }
    outcome = {
        "status": "OUTCOME_STATUS_SUCCEEDED",
        "result": result,
        "outputs": {"video": "/proof/video.mp4", "continuation_frame": "/proof/frame.png"},
        "invocation": {
            "digest": "sha256:" + "e" * 64,
            "document": {"format": "cozy.worker.v1.InvocationSpec/1"},
        },
        "plan": accepted_plan,
        "ledger": {"format": "cozy.runtime.Ledger/0", "classes": [{"class": "vram"}]},
        "timings": {"wall_ms": 1.0},
        "warnings": [],
    }
    check(
        "full Runtime outcome carries the exact two-field catalog result",
        sorted(verify_result("first_last_frame_to_video", outcome)[0]),
        ["continuation_frame", "video"],
    )
    refusal(
        "an identity or digest rider in the customer result refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "result": {**result, "checkpoint": snapshot}},
        ),
        "RuntimeError",
    )
    refusal(
        "an empty accepted Runtime plan cannot masquerade as an execution outcome",
        lambda: verify_result("first_last_frame_to_video", {**outcome, "plan": {}}),
        "RuntimeError",
    )
    refusal(
        "malformed Runtime accepted-plan identity refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "plan": {**accepted_plan, "plan_digest": "b" * 64}},
        ),
        "RuntimeError",
    )
    refusal(
        "malformed Runtime model-construction identity refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "plan": {**accepted_plan, "model_construction_digest": "b" * 64}},
        ),
        "RuntimeError",
    )
    refusal(
        "an empty canonical invocation receipt refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "invocation": {}},
        ),
        "RuntimeError",
    )
    refusal(
        "a canonical invocation receipt without its document refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {
                **outcome,
                "invocation": {"digest": "sha256:" + "e" * 64},
            },
        ),
        "RuntimeError",
    )
    refusal(
        "a malformed canonical invocation digest refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {
                **outcome,
                "invocation": {
                    **cast(dict[str, Any], outcome["invocation"]),
                    "digest": "e" * 64,
                },
            },
        ),
        "RuntimeError",
    )
    refusal(
        "accepted plan and canonical invocation disagreement refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {
                **outcome,
                "plan": {
                    **accepted_plan,
                    "invocation_spec_digest": "sha256:" + "f" * 64,
                },
            },
        ),
        "RuntimeError",
    )
    refusal(
        "incomplete Runtime output grants refuse",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "outputs": {"video": "/proof/video.mp4"}},
        ),
        "RuntimeError",
    )
    refusal(
        "an extra Runtime outcome field refuses",
        lambda: verify_result("first_last_frame_to_video", {**outcome, "unbound": True}),
        "RuntimeError",
    )
    refusal(
        "a Runtime warning refuses the proof",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "warnings": ["ignored extra key"]},
        ),
        "RuntimeError",
    )

    media_contract = {
        "video_format": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_major_brand": "isom",
        "video_codec": "h264",
        "audio_codec": "aac",
        "video_width": 768,
        "video_height": 512,
        "image_format": "png_pipe",
        "image_codec": "png",
        "continuation_width": 768,
        "continuation_height": 512,
    }
    verify_media_contract(**media_contract)
    observe("stored H264/AAC MP4 and PNG contract validates")
    for name, changes in (
        ("wrong stored video codec refuses", {"video_codec": "hevc"}),
        ("wrong stored audio codec refuses", {"audio_codec": "pcm_s16le"}),
        ("QuickTime major brand refuses", {"video_major_brand": "qt  "}),
        ("wrong stored video geometry refuses", {"video_width": 640}),
        ("wrong continuation container refuses", {"image_format": "image2"}),
    ):
        refusal(
            name,
            partial(verify_media_contract, **{**media_contract, **changes}),
            "RuntimeError",
        )

    def encoded_media_fixture(
        root: Path,
        *,
        video_codec: str = "libx264",
        extension: str = "mp4",
        frame_count: int = 345,
        audio_rate: int = 32000,
        audio_sample_count: int = 460000,
        continuation_width: int = 32,
        continuation_height: int = 32,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        import av
        import numpy as np
        from av.audio.stream import AudioStream
        from av.video.stream import VideoStream
        from PIL import Image

        root.mkdir()
        video = root / f"video.{extension}"
        continuation = root / "continuation.png"
        container = av.open(str(video), "w")
        video_stream = cast(VideoStream, container.add_stream(video_codec, rate=24))
        video_stream.width = 32
        video_stream.height = 32
        video_stream.pix_fmt = "yuv420p"
        audio_stream: AudioStream = container.add_stream("aac", rate=audio_rate)
        audio_stream.layout = "mono"
        last = np.zeros((32, 32, 3), dtype=np.uint8)
        for index in range(frame_count):
            last.fill(index % 256)
            frame = av.VideoFrame.from_ndarray(last, format="rgb24")
            frame.pts = index
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        audio_pts = 0
        while audio_pts < audio_sample_count:
            samples = min(1024, audio_sample_count - audio_pts)
            audio_frame = av.AudioFrame.from_ndarray(
                np.zeros((1, samples), dtype=np.float32), format="fltp", layout="mono"
            )
            audio_frame.sample_rate = audio_rate
            audio_frame.pts = audio_pts
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            audio_pts += samples
        for packet in audio_stream.encode():
            container.mux(packet)
        container.close()
        continuation_pixels_array = np.zeros(
            (continuation_height, continuation_width, 3), dtype=np.uint8
        )
        continuation_pixels_array.fill(last[0, 0, 0])
        Image.fromarray(continuation_pixels_array).save(continuation)

        video_digest = hashlib.sha256(video.read_bytes()).hexdigest()
        continuation_digest = hashlib.sha256(continuation.read_bytes()).hexdigest()
        result = {
            "video": {"digest": f"sha256:{video_digest}"},
            "continuation_frame": {"digest": f"sha256:{continuation_digest}"},
        }
        outputs = {"video": str(video), "continuation_frame": str(continuation)}
        return result, outputs

    with tempfile.TemporaryDirectory() as temporary:
        media_root = Path(temporary)
        good_root = media_root / "good"
        good_result, good_outputs = encoded_media_fixture(good_root)
        stored = probe_media(good_root, good_result, good_outputs)
        check(
            "real stored media is H264/AAC MP4 plus PNG",
            (stored["video_codec"], stored["audio_codec"], stored["continuation_codec"]),
            ("h264", "aac", "png"),
        )
        refusal(
            "a real wrong-codec MP4 refuses",
            lambda: probe_media(
                media_root / "mpeg4",
                *encoded_media_fixture(media_root / "mpeg4", video_codec="mpeg4"),
            ),
            "RuntimeError",
        )
        refusal(
            "a real wrong-container file refuses",
            lambda: probe_media(
                media_root / "matroska",
                *encoded_media_fixture(media_root / "matroska", extension="mkv"),
            ),
            "RuntimeError",
        )
        refusal(
            "a real QuickTime MOV refuses",
            lambda: probe_media(
                media_root / "quicktime",
                *encoded_media_fixture(media_root / "quicktime", extension="mov"),
            ),
            "RuntimeError",
        )
        refusal(
            "a real wrong frame clock refuses",
            lambda: probe_media(
                media_root / "short-video",
                *encoded_media_fixture(media_root / "short-video", frame_count=344),
            ),
            "RuntimeError",
        )
        refusal(
            "a real wrong audio sample rate refuses",
            lambda: probe_media(
                media_root / "wrong-audio-rate",
                *encoded_media_fixture(
                    media_root / "wrong-audio-rate",
                    audio_rate=44100,
                    audio_sample_count=633938,
                ),
            ),
            "RuntimeError",
        )
        refusal(
            "a real A/V duration mismatch refuses",
            lambda: probe_media(
                media_root / "short-audio",
                *encoded_media_fixture(media_root / "short-audio", audio_sample_count=450000),
            ),
            "RuntimeError",
        )
        refusal(
            "a Runtime output path outside the action root refuses",
            lambda: probe_media(
                good_root,
                good_result,
                {**good_outputs, "video": str(media_root / "outside.mp4")},
            ),
            "RuntimeError",
        )
        refusal(
            "a real stored asset digest mismatch refuses",
            lambda: probe_media(
                good_root,
                {**good_result, "video": {"digest": "sha256:" + "0" * 64}},
                good_outputs,
            ),
            "RuntimeError",
        )
        refusal(
            "a real continuation asset digest mismatch refuses",
            lambda: probe_media(
                good_root,
                {
                    **good_result,
                    "continuation_frame": {"digest": "sha256:" + "0" * 64},
                },
                good_outputs,
            ),
            "RuntimeError",
        )
        refusal(
            "a real continuation geometry mismatch refuses",
            lambda: probe_media(
                media_root / "wrong-continuation-geometry",
                *encoded_media_fixture(
                    media_root / "wrong-continuation-geometry", continuation_width=31
                ),
            ),
            "RuntimeError",
        )


ARMS = {
    "schedule": arm_schedule,
    "graph": arm_graph_and_dtypes,
    "conditioner": arm_text_conditioner,
    "adaln-pruned": arm_adaln_pruned,
    "processor": arm_processor,
    "media": arm_media,
    "gates": arm_output_gates,
    "descriptor": arm_descriptor,
    "live-probe": arm_live_probe,
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
