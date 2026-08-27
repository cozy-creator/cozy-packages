#!/usr/bin/env python
"""Deterministic MiniMax-H3 contract arms; no weights, GPU, network, or test framework.

Every arm executes the official Diffusers 0.40 implementation or a public endpoint
boundary. Each historically dangerous invariant also carries a negative control. A green
run is a CPU semantic proof, not a generation or accelerator proof.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import struct
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

import h3 as endpoint  # noqa: E402
from gates import MediaFacts, pre_encode_gate  # noqa: E402
from official import (  # noqa: E402
    FPS,
    FRAMES,
    MAX_CONDITIONER_VISION_TOKENS,
    SIGMA_GRID_POINTS,
    _aligned_soundtrack,
    _apply_transformer_dtype,
    _artifact_sections,
    _as_float32,
    _processor,
    _validate_model_contract,
    _validate_row_timestep_plan,
    _video_at_24fps,
    canonical_timestep_plan,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    validate_reference_policy,
)

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0

PLAN_DIGESTS = {
    "fl2va": "b72b46a6d753b4db3be175cd2c14ea012bd3524327e170d9bf064dd2fd1f2075",
    "ref2va": "86143d5ad14f3936b01cc1732241c0bb9b943138d585d8f7b887c7533e4a8732",
}
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

    print("\n== exact schedule and baked-AdaLN handoff ==")
    plans = {task: canonical_timestep_plan(task) for task in ("fl2va", "ref2va")}
    for task, plan in plans.items():
        check(f"{task} canonical plan digest", plan.digest, PLAN_DIGESTS[task])
        check(
            f"{task} committed canonical bytes",
            hashlib.sha256((H3 / "timestep-plans" / f"{task}.json").read_bytes()).hexdigest(),
            plan.digest,
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
            len(document["baked_table_keys"]["block_modulation"]),
            89,
        )
        check(
            f"{task} canonical final-normalization rows",
            len(document["baked_table_keys"]["final_normalization"]),
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
        "text_encoder": {},
        "transformer": {},
        "transformer_ref": {},
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
    from cozy_runtime.author import AudioAsset, VideoAsset

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
    endpoint._validate_vision_budget(32768)
    observe("vision capacity boundary")
    refusal(
        "vision demand above the release budget refuses",
        lambda: endpoint._validate_vision_budget(MAX_CONDITIONER_VISION_TOKENS + 1),
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
        @staticmethod
        def decode_video(asset: Any) -> Any:
            del asset
            return _video(10, soundtrack=_audio(10))

        @staticmethod
        def decode_audio(asset: Any) -> Any:
            del asset
            return _audio(6)

    references = [
        endpoint.VideoReference(VideoAsset("sha256:" + "1" * 64)),
        endpoint.AudioReference(AudioAsset("sha256:" + "2" * 64)),
    ]
    refusal(
        "embedded soundtrack plus standalone audio share the 15-second audio budget",
        lambda: endpoint._decode_references(
            cast(Any, references), decoder=cast(Any, Decoder()), pipe=cast(Any, Pipe())
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
    pixels = endpoint._rgb8(torch, decoded)
    check("RGB8 conversion shape", tuple(pixels.shape), (2, 1, 2, 3))
    check("RGB8 clamp and round", pixels[0].flatten().tolist(), [0, 128, 0, 125, 255, 255])
    continuation = bytes(pixels[-1].numpy().tobytes())
    check(
        "continuation is the last pre-encode RGB frame",
        continuation,
        bytes([255, 64, 26, 0, 191, 230]),
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
            video_nonfinite_fraction=endpoint._nonfinite_fraction(torch, poisoned),
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
    descriptor = json.loads((H3 / "endpoint.descriptor.json").read_text())
    entries = {entry["name"]: entry for entry in descriptor["entrypoints"]}
    check(
        "exact action names",
        set(entries),
        {"first_last_frame_to_video", "reference_media_to_video"},
    )
    expected = {
        "first_last_frame_to_video": (
            ["prompt", "first_frame", "last_frame", "mute", "seed"],
            "Fl2VAModel",
            "fl2va",
        ),
        "reference_media_to_video": (
            ["prompt", "references", "mute", "seed"],
            "Ref2VAModel",
            "ref2va",
        ),
    }
    for name, (fields, model, task) in expected.items():
        entry = entries[name]
        check(
            f"{name} request fields",
            [field["name"] for field in entry["request"]["fields"]],
            fields,
        )
        check(f"{name} model", entry["models"][0]["class"], model)
        check(f"{name} task stamp", entry["models"][0]["stamps"]["task"], task)
        check(f"{name} media capability", "media_decode" in entry["capabilities"], True)


def arm_live_probe() -> None:
    print("\n== installed-artifact production probe boundary ==")
    probe = runpy.run_path(str(ROOT / "scripts" / "h3-live.py"))
    command_json = cast(Callable[[list[str]], dict[str, Any]], probe["command_json"])
    load_request = cast(Callable[..., Any], probe["load_request"])
    load_plan_facts = cast(Callable[..., dict[str, dict[str, str]]], probe["load_plan_facts"])
    reserve_output = cast(Callable[[Path], Path], probe["reserve_output"])
    seal_requests = cast(Callable[[Path, list[Path]], None], probe["seal_requests"])
    stage_request = cast(Callable[..., Path], probe["stage_request"])
    verify_bindings = cast(Callable[..., list[dict[str, Any]]], probe["verify_bindings"])
    verify_staged_request = cast(Callable[..., None], probe["verify_staged_request"])
    verify_media_contract = cast(Callable[..., None], probe["verify_media_contract"])
    probe_media = cast(Callable[..., dict[str, Any]], probe["probe_media"])
    verify_result = cast(
        Callable[..., tuple[dict[str, Any], dict[str, str]]], probe["verify_result"]
    )
    verify_surface = cast(Callable[..., str], probe["verify_surface"])

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

    binding_ref = "cozy/minimax-h3@se-012"
    snapshot = "sha256:" + "1" * 64
    surface = "sha256:" + "2" * 64
    runtime_plan = "sha256:" + "5" * 64
    construction = "sha256:" + "6" * 64
    components = ["audio_vae", "text_encoder", "transformer", "transformer_ref", "video_vae"]

    def binding(path: str, *, installed: bool = True) -> dict[str, Any]:
        return {
            "model_binding_path": path,
            "ref": binding_ref,
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
    check(
        "one installed uniform dual binding arms both actions",
        len(
            verify_bindings(
                document,
                expected_ref=binding_ref,
                expected_checkpoint=snapshot,
            )
        ),
        2,
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
        ),
        "RuntimeError",
    )
    refusal(
        "a different component snapshot refuses before inference",
        lambda: verify_bindings(
            document,
            expected_ref=binding_ref,
            expected_checkpoint="sha256:" + "2" * 64,
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
        ),
        "RuntimeError",
    )

    check(
        "the described endpoint surface must equal the launch contract",
        verify_surface({"surface_digest": surface}, expected=surface),
        surface,
    )
    refusal(
        "a different endpoint surface refuses before inference",
        lambda: verify_surface({"surface_digest": surface}, expected="sha256:" + "3" * 64),
        "RuntimeError",
    )

    expected_plans = {
        "first_last_frame_to_video": "sha256:" + PLAN_DIGESTS["fl2va"],
        "reference_media_to_video": "sha256:" + PLAN_DIGESTS["ref2va"],
    }
    selected_plan_facts = load_plan_facts(H3, expected_plans)
    check(
        "selected endpoint plan bytes match both launch identities",
        {action: facts["document_digest"] for action, facts in selected_plan_facts.items()},
        expected_plans,
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        selected = root / "selected-endpoint"
        plans = selected / "timestep-plans"
        plans.mkdir(parents=True)
        for task in ("fl2va", "ref2va"):
            (plans / f"{task}.json").write_bytes(
                (H3 / "timestep-plans" / f"{task}.json").read_bytes()
            )
        (plans / "fl2va.json").write_bytes((plans / "fl2va.json").read_bytes() + b" ")
        refusal(
            "selected endpoint plan drift cannot fall back to checkout-global plans",
            lambda: load_plan_facts(selected, expected_plans),
            "RuntimeError",
        )

        def request(name: str, raw: bytes) -> tuple[Path, str]:
            path = root / name
            path.write_bytes(raw)
            return path, f"sha256:{hashlib.sha256(raw).hexdigest()}"

        good_path, good_digest = request("good.json", b'{ "prompt": "proof", "seed": 17 }\n')
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
            "a sealed request cannot be rewritten",
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
        seal_requests(requests, [staged])
        verify_staged_request(staged, request_snapshot)
        observe("sealed request identity validates after directory seal")

    result = {
        "frames": 345,
        "fps": 24,
        "sample_rate": 32000,
        "sigma_grid_points": 30,
        "transformer_evaluations": 29,
        "timestep_plan_digest": PLAN_DIGESTS["fl2va"],
        "checkpoint": snapshot,
        **{
            key: value
            for key, value in selected_plan_facts["first_last_frame_to_video"].items()
            if key != "document_digest"
        },
        "video_pixel_digest": "7" * 64,
        "audio_sample_digest": "8" * 64,
        "continuation_frame_digest": "9" * 64,
        "continuation_pixel_digest": "a" * 64,
    }
    outcome = {
        "status": "OUTCOME_STATUS_SUCCEEDED",
        "result": result,
        "outputs": {"video": "/proof/video.mp4", "continuation_frame": "/proof/frame.png"},
        "plan": {
            "plan_digest": runtime_plan,
            "model_construction_digest": construction,
            "delivery": "resident",
            "materialization": "installed",
            "placement": "all_resident",
            "compute_dtype": "bfloat16",
            "reserved_device_memory_bytes": 1,
        },
        "ledger": {"format": "cozy.runtime.Ledger/0", "classes": [{"class": "vram"}]},
        "timings": {"wall_ms": 1.0},
        "warnings": [],
    }
    check(
        "full Runtime outcome carries the expected action plan",
        verify_result(
            "first_last_frame_to_video",
            outcome,
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        )[0]["timestep_plan_digest"],
        PLAN_DIGESTS["fl2va"],
    )
    refusal(
        "a sibling action plan in the Runtime result refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            outcome,
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["reference_media_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )
    refusal(
        "an empty accepted Runtime plan cannot masquerade as an execution outcome",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "plan": {}},
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )
    refusal(
        "wrong Runtime accepted-plan identity refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            outcome,
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest="sha256:" + "b" * 64,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )
    refusal(
        "incomplete Runtime output grants refuse",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "outputs": {"video": "/proof/video.mp4"}},
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )
    refusal(
        "an extra Runtime outcome field refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "unbound": True},
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )
    refusal(
        "a Runtime warning refuses the proof",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "warnings": ["ignored extra key"]},
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )
    refusal(
        "a drifting schedule observation refuses",
        lambda: verify_result(
            "first_last_frame_to_video",
            {**outcome, "result": {**result, "video_sigma_digest": "c" * 64}},
            expected_checkpoint=snapshot,
            expected_plan=selected_plan_facts["first_last_frame_to_video"],
            expected_runtime_plan_digest=runtime_plan,
            expected_construction_digest=construction,
        ),
        "RuntimeError",
    )

    media_contract = {
        "video_format": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_codec": "h264",
        "audio_codec": "aac",
        "video_width": 768,
        "video_height": 512,
        "image_format": "png_pipe",
        "image_codec": "png",
        "expected_width": 768,
        "expected_height": 512,
    }
    verify_media_contract(**media_contract)
    observe("stored H264/AAC MP4 and PNG contract validates")
    for name, changes in (
        ("wrong stored video codec refuses", {"video_codec": "hevc"}),
        ("wrong stored audio codec refuses", {"audio_codec": "pcm_s16le"}),
        ("wrong stored video geometry refuses", {"video_width": 640}),
        ("wrong continuation container refuses", {"image_format": "image2"}),
    ):
        refusal(
            name,
            partial(verify_media_contract, **{**media_contract, **changes}),
            "RuntimeError",
        )

    def encoded_media_fixture(
        root: Path, *, video_codec: str = "libx264", extension: str = "mp4"
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
        audio_stream: AudioStream = container.add_stream("aac", rate=32000)
        audio_stream.layout = "mono"
        last = np.zeros((32, 32, 3), dtype=np.uint8)
        for index in range(345):
            last.fill(index % 256)
            frame = av.VideoFrame.from_ndarray(last, format="rgb24")
            frame.pts = index
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        audio_pts = 0
        while audio_pts < 460000:
            samples = min(1024, 460000 - audio_pts)
            audio_frame = av.AudioFrame.from_ndarray(
                np.zeros((1, samples), dtype=np.float32), format="fltp", layout="mono"
            )
            audio_frame.sample_rate = 32000
            audio_frame.pts = audio_pts
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            audio_pts += samples
        for packet in audio_stream.encode():
            container.mux(packet)
        container.close()
        Image.fromarray(last).save(continuation)

        video_digest = hashlib.sha256(video.read_bytes()).hexdigest()
        continuation_digest = hashlib.sha256(continuation.read_bytes()).hexdigest()
        continuation_pixels = hashlib.sha256(last.tobytes()).hexdigest()
        result = {
            "video": {"digest": f"sha256:{video_digest}"},
            "continuation_frame": {"digest": f"sha256:{continuation_digest}"},
            "continuation_frame_digest": continuation_digest,
            "continuation_pixel_digest": continuation_pixels,
            "width": 32,
            "height": 32,
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
            "a real stored asset digest mismatch refuses",
            lambda: probe_media(
                good_root,
                {**good_result, "video": {"digest": "sha256:" + "0" * 64}},
                good_outputs,
            ),
            "RuntimeError",
        )


ARMS = {
    "schedule": arm_schedule,
    "graph": arm_graph_and_dtypes,
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
