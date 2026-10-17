#!/usr/bin/env python
"""Deterministic MiniMax-H3 contract arms; no weights, GPU, network, or test framework.

Every arm executes the official Diffusers 0.40 implementation or a public package
boundary. Each historically dangerous invariant also carries a negative control. A green
run is a CPU semantic proof, not a generation or accelerator proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import struct
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from functools import partial
from importlib.resources import files
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from typing import Any, Literal, cast, get_type_hints

import av
import msgspec
import numpy as np
import torch
from cozy_runtime.author import (
    App,
    Artifact,
    Assets,
    AttentionLayout,
    AudioAsset,
    Cancelled,
    Config,
    Context,
    DecodedAudio,
    DecodedVideo,
    ImageAsset,
    InvalidRequest,
    Invocation,
    VideoAsset,
    attention_scope,
    canonical_json,
    decode_request,
    describe,
)
from cozy_runtime.author import attempt as kernel_attempt
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.author._demand import normalize
from cozy_runtime.author.fakes import (
    fake_attempt,
    fake_input,
    fake_outputs,
    fake_telemetry,
    granted_input,
)
from cozy_runtime.internal.derive import derive
from diffusers import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3LoopDenoiser,
)
from PIL import Image as PILImage
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "minimax-h3"
sys.path.insert(0, str(H3))
sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))

import minimax_h3.official as official_module  # noqa: E402
from minimax_h3 import official  # noqa: E402
from minimax_h3.adaln_pruned import (  # noqa: E402
    AdaLNPrunedMiniMaxH3Transformer,
    _AdaLNPrunedBlockTable,
    _AdaLNPrunedOutputTable,
    _AdaLNPrunedTimestepLookup,
)
from minimax_h3.conditioner import (  # noqa: E402
    FinalHiddenState,
    build_text_conditioner,
    text_conditioner_config,
)
from minimax_h3.continuation import Carry, ContinuationPlan  # noqa: E402
from minimax_h3.official import (  # noqa: E402
    FPS,
    MAX_CONDITIONER_VISION_TOKENS,
    MAX_DURATION,
    MAX_FRAMES,
    MIN_DURATION,
    MIN_FRAMES,
    REFERENCE_IMAGE_SHORT_EDGE,
    OfficialH3Pipeline,
    ScheduleFacts,
    _aligned_soundtrack,
    _apply_transformer_dtype,
    _apply_video_vae_dtype,
    _artifact_sections,
    _as_float32,
    _attention_layout,
    _dit_specs,
    _processor,
    _ScopedPipeline,
    _frames,
    _validate_dual_dit_topology,
    _validate_model_contract,
    canonical_timestep_plan,
    denoise_rows,
    frames_for,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    supported_durations,
    supported_steps,
    timestep_plan_digest,
    validate_reference_policy,
)
from minimax_h3 import turbo as turbo_module  # noqa: E402
from minimax_h3.reference_rows import motion_size  # noqa: E402
from minimax_h3.turbo import (  # noqa: E402
    ATTENTION_KWARG,
    LORA_FAMILIES,
    OVERLAY_KWARG,
    TURBO_BANK,
    LoRAFactors,
    TurboHeads,
    TurboOverlay,
    TurboSchedule,
    _LoRAHook,
)
from minimax_h3.vae_tiles import TileBatchedVideoVAE  # noqa: E402
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention  # noqa: E402
from minimax_h3 import glue, producer  # noqa: E402
from cozy_runtime.author.kernels import Plan, PreQuantized  # noqa: E402
from h3_tables.lanes import NORMALISED_COMPONENTS, decode_operand  # noqa: E402
from h3_tables.model_config import (  # noqa: E402
    dual_adaln_pruned_config,
    dual_full_config,
    parse_production_config,
)
from h3_tables.order import current_order, full_order  # noqa: E402
from h3_tables.plans import TASKS, parse_plan  # noqa: E402
from h3_tables.source import official_full_specs  # noqa: E402
from h3_tables.table_layout import TableLayout  # noqa: E402
from h3_tables.turbo import collapse_head_bank, pdd_head_plan, pdd_time_grid  # noqa: E402

import h3 as package  # noqa: E402
from story import StoryReference, compile_segments  # noqa: E402
from gates import MediaFacts, refuse_before_encode  # noqa: E402
from h3_order import construction_order, encode_order  # noqa: E402

STEPS = supported_steps()
DEFAULT_STEPS = min(STEPS)
DURATIONS = supported_durations()
DEFAULT_DURATION_S = min(DURATIONS)  # this driver's fixture length; the package has no default
DEFAULT_FRAMES = frames_for(DEFAULT_DURATION_S)
#: The one canvas a generated clip resolves to without a keyframe.
CANVAS_HEIGHT, CANVAS_WIDTH = 768, 1344

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0

PLAN_DIGESTS = {
    "fl2va": "9a48803d17d7bb5499ca8c018f60c86c9890eb91496249ac7cb199bc9e45201e",
    "ref2va": "565a164cbf0cefa58d4976cb4c84887cf9807cc7c4be62263d5e0e0c5d9bc50e",
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
# PDD-8 (alibaba-pai/MiniMax-H3-Acc-LoRAs rev 335001fb): one fixed 8-evaluation schedule per
# trunk, stamped with the trunk, whose tables the turbo overlay components are keyed to.
TURBO_PLAN_DIGESTS = {
    "fl2va_turbo": "d2eb1605c1c1e01a4c5fdaaf1912ab43f33d9ce4772febf1c75b9bbfc8f7ba9d",
    "ref2va_turbo": "896a10881805e3047f6df514dd4e70fcea078837469eeb22ce67f0bd3f679cb7",
}
TURBO_BLOCK_ROWS = 26
TURBO_FINAL_ROWS = 17
#: The adapter file's header, read 2026-09-08: what the overlay's destinations must match.
PDD_HEADER: dict[str, Any] = {
    "lora_rank": 64,
    "lora_alpha": 64.0,
    "pdd_num_steps": 32,
    "pdd_block_size": 4,
    "lora_targets": "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear",
}
#: Per-family factor shapes in that file at the release widths, `[rank, in]` / `[out, rank]`.
PDD_FACTOR_SHAPES = {
    "to_q": ((64, 5376), (7168, 64)),
    "to_k": ((64, 5376), (7168, 64)),
    "to_v": ((64, 5376), (7168, 64)),
    "to_out.0": ((64, 7168), (5376, 64)),
    "ff.net.0.proj": ((64, 5376), (28672, 64)),
    "ff.net.2": ((64, 14336), (5376, 64)),
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


def dit_config(task: str, modulation: str = "full") -> dict[str, Any]:
    plan = canonical_timestep_plan(cast(Any, task))
    extension: dict[str, Any] = {"task": task, "modulation": modulation}
    if modulation == "adaln-pruned":
        extension["table_keys"] = canonical_json.decode(plan.canonical_bytes())["table_keys"]
    return {"cozy_h3": extension}


def arm_checkpoint_table_layout() -> None:
    """Row labels travel with table values; no package digest determines their meaning."""
    plan = canonical_timestep_plan("fl2va")
    document = canonical_json.decode(plan.canonical_bytes())["table_keys"]
    layout = TableLayout.parse(document)
    for schedule in plan.schedules:
        layout.require(schedule.video_timesteps, schedule.audio_timesteps)
    reordered = copy.deepcopy(document)
    for rows in reordered.values():
        rows.reverse()
        for index, row in enumerate(rows):
            row["index"] = index
    shuffled = TableLayout.parse(reordered)
    for schedule in plan.schedules:
        shuffled.require(schedule.video_timesteps, schedule.audio_timesteps)

    # Exercise the real inference lookup and projections with labelled, nontrivial
    # values. Reverse each checkpoint tensor with its labels: its output must agree.
    block_values = torch.arange(len(layout.block_keys) * 12, dtype=torch.float32).reshape(-1, 6, 2)
    final_values = torch.arange(len(layout.timesteps) * 4, dtype=torch.float32).reshape(-1, 2, 2)
    requested = torch.tensor(layout.timesteps)

    def run_rows(labels: TableLayout, block: torch.Tensor, final: torch.Tensor) -> tuple[Any, Any]:
        lookup = _AdaLNPrunedTimestepLookup(labels.timesteps)
        projection = _AdaLNPrunedBlockTable(
            hidden_size=2, timestep_count=len(labels.timesteps), keys=labels.block_keys
        )
        output = _AdaLNPrunedOutputTable(
            torch.nn.Identity(), hidden_size=2, timestep_count=len(labels.timesteps)
        )
        with torch.no_grad():
            projection.table.copy_(block)
            output.table.copy_(final)
        indexes = lookup(requested)
        projected = torch.stack(projection(indexes), dim=1).reshape(-1, 3, 6, 2)
        selected = torch.stack([projected[row, tag] for row, tag in layout.block_keys])
        normalized = output(torch.ones(len(requested), 2), indexes, torch.arange(len(requested)))
        return selected, normalized

    original = run_rows(layout, block_values, final_values)
    reordered_output = run_rows(shuffled, block_values.flip(0), final_values.flip(0))
    for name, old, new in zip(
        ("block modulation", "final normalization"), original, reordered_output, strict=True
    ):
        check(f"checkpoint row reordering preserves {name}", torch.equal(old, new), True)
    wrong = run_rows(shuffled, block_values, final_values)
    red("moving labels without table rows changes output", torch.equal(original[0], wrong[0]), True)

    config: dict[str, Any] = {"fl2va_dit": dit_config("fl2va", "adaln-pruned")}
    config["fl2va_dit"]["cozy_h3"]["table_keys"] = reordered
    check(
        "serving constructor accepts arbitrary checkpoint row order",
        official._dit_spec(config, "fl2va")[2] == shuffled,
        True,
    )
    extra = copy.deepcopy(document)
    extra["final_normalization"].append({"index": len(layout.timesteps), "timestep": (0.125).hex()})
    extra["block_modulation"].append(
        {
            "index": len(layout.block_keys),
            "timestep": (0.125).hex(),
            "modality": "video",
            "modality_tag": 0,
        }
    )
    config["fl2va_dit"]["cozy_h3"]["table_keys"] = extra
    extended = official._dit_spec(config, "fl2va")[2]
    assert extended is not None
    check(
        "checkpoint can add rows without a package change",
        len(extended.block_keys),
        len(layout.block_keys) + 1,
    )
    trunks = {task: torch.nn.Module() for task in TASKS}
    for index, model in enumerate(trunks.values()):
        model.norm_out = _AdaLNPrunedOutputTable(
            torch.nn.Identity(), hidden_size=2, timestep_count=len(layout.timesteps) + index
        )
    official._validate_dual_dit_topology(trunks)
    check("trunks may have independent table row counts", True, True)
    trunks["ref2va"].norm_out = _AdaLNPrunedOutputTable(
        torch.nn.Identity(), hidden_size=3, timestep_count=len(layout.timesteps)
    )
    refusal(
        "independent row counts do not weaken feature-axis validation",
        partial(official._validate_dual_dit_topology, trunks),
        "artifact_config",
    )

    t = (1.0).hex()
    tolerant = TableLayout.parse(
        {
            "final_normalization": [{"timestep": t, "writer": "future"}],
            "block_modulation": [
                {"index": 0, "timestep": t, "modality_tag": 0, "modality": "video"},
                {"timestep": t, "modality": "text"},
                {"timestep": t, "modality_tag": 2, "note": 1},
            ],
            "provenance": {"tool": "future"},
        }
    )
    check(
        "labels from other writers ignore extras and default optionals",
        tolerant.block_keys,
        ((0, 0), (0, 1), (0, 2)),
    )
    corruptions: tuple[tuple[str, Callable[[Any], None]], ...] = (
        ("missing layout", lambda value: value.clear()),
        ("noncontiguous index", lambda value: value["block_modulation"][0].update(index=1)),
        ("nonfinite value", lambda value: value["final_normalization"][0].update(timestep="inf")),
        (
            "inexact float32",
            lambda value: value["final_normalization"][0].update(timestep=(0.1).hex()),
        ),
        ("unknown modality", lambda value: value["block_modulation"][0].update(modality_tag=3)),
        (
            "mislabeled modality",
            lambda value: value["block_modulation"][0].update(modality="audio"),
        ),
        (
            "unbound final row",
            lambda value: value["block_modulation"][0].update(timestep=(0.125).hex()),
        ),
        (
            "duplicate final row",
            lambda value: value["final_normalization"][1].update(
                timestep=value["final_normalization"][0]["timestep"]
            ),
        ),
        (
            "duplicate block row",
            lambda value: value["block_modulation"][1].update(
                **{**value["block_modulation"][0], "index": 1}
            ),
        ),
    )
    for name, change in corruptions:
        value = copy.deepcopy(document)
        change(value)
        refusal(name, partial(TableLayout.parse, value), "artifact_config")
    refusal(
        "an uncovered requested timestep refuses",
        lambda: layout.require([0.125], [0.0]),
        "artifact_config",
    )


def arm_producer_configs() -> None:
    """Pass actual producer bytes into the serving parser before any weight transfer."""
    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))

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
                ("table_keys", {}),
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
                del changed[component]["cozy_h3"]["table_keys"]
                refusal(
                    f"{component} pruned tables need their row labels",
                    partial(_dit_specs, changed),
                    "artifact_config",
                )


def _asset(name: str) -> bytes:
    return files("h3_tables").joinpath("assets", name).read_bytes()


def arm_producer_construction_order() -> None:
    """The producer's one ordered spec resource follows the real serving factory."""
    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))

    sections = parse_production_config(_asset("model-config.json"))
    current = current_order(_asset("whole-order.json"))
    plans = {
        task: parse_plan(_asset(f"timestep-plan.{task}.json"), task=task)
        for task in ("fl2va", "ref2va")
    }
    for mode, raw, expected in (
        ("full", dual_full_config(sections), full_order(sections, current.rows)),
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
    print("\n== exact schedule and AdaLN-pruned handoff ==")
    plans = {task: canonical_timestep_plan(task) for task in ("fl2va", "ref2va")}
    for task, plan in plans.items():
        check(f"{task} canonical plan digest", plan.digest, PLAN_DIGESTS[task])
        committed = files("minimax_h3").joinpath("timestep-plans", f"{task}.json").read_bytes()
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
        changed = dict(parsed, fps=parsed["fps"] + 1)
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
            official_timesteps = tuple(float(value) for value in scheduler.timesteps.float().cpu())
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
    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))

    assets = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets"
    config = canonical_json.decode(
        dual_full_config(parse_production_config((assets / "model-config.json").read_bytes()))
    )
    with torch.device("meta"):
        return OfficialH3Pipeline(Config(config))


def arm_reference_resolution() -> None:
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
            frames=DEFAULT_FRAMES,
            reference_image_short_edges=[edge],
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
    official = pipe.start_ref2va(
        prompt="A person in a garden.",
        references=references,
        generator=torch.Generator().manual_seed(7),
        steps=DEFAULT_STEPS,
        frames=DEFAULT_FRAMES,
        reference_image_short_edges=[REFERENCE_IMAGE_SHORT_EDGE],
    )
    check(
        "official 2048px pixels stay identical after smaller requests",
        np.array_equal(np.asarray(official.normalized_references[0].image), normalized[2048]),
        True,
    )
    mixed = pipe.start_ref2va(
        prompt="A person in a garden.",
        references=references * 3,
        generator=torch.Generator().manual_seed(7),
        steps=DEFAULT_STEPS,
        frames=DEFAULT_FRAMES,
        reference_image_short_edges=[512, 2048, 1024],
    )
    shapes = [np.asarray(entry.image).shape for entry in mixed.normalized_references]
    check(
        "mixed per-image edges reach official preprocessing in packed order",
        shapes,
        [(672, 512, 3), (2720, 2048, 3), (1376, 1024, 3)],
    )
    check(
        "each image's budget agrees with its own upstream geometry",
        [height * width // 1024 for height, width, _ in shapes],
        [reference_image_vision_tokens(1086, 1448, edge) for edge in (512, 2048, 1024)],
    )
    check(
        "mixed request leaves shared config unchanged",
        dict(pipe._pipes["ref2va"].config) == original_config,
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
                frames=DEFAULT_FRAMES,
                reference_image_short_edges=[768],
            ),
        )
    finally:
        pipe._blocks["ref2va"].sub_blocks["before_encode"] = before
    check(
        "shared config survives preprocessing failure",
        dict(pipe._pipes["ref2va"].config) == original_config,
        True,
    )

    def greedy_setup(components: Any, state: Any) -> None:
        del state
        first = components.config.reference_image_short_edge
        second = components.config.reference_image_short_edge
        fail("greedy setup read a second edge for one image", f"{first}, {second}")

    def idle_setup(components: Any, state: Any) -> None:
        del components, state

    for drift, step in (("more", greedy_setup), ("fewer", idle_setup)):
        pipe._blocks["ref2va"].sub_blocks["before_encode"] = step
        try:
            refusal(
                f"a setup step reading {drift} edges than images refuses typed",
                lambda: pipe.start_ref2va(
                    prompt="A person in a garden.",
                    references=references,
                    generator=torch.Generator().manual_seed(7),
                    steps=DEFAULT_STEPS,
                    frames=DEFAULT_FRAMES,
                    reference_image_short_edges=[768],
                ),
                "artifact_config",
            )
        finally:
            pipe._blocks["ref2va"].sub_blocks["before_encode"] = before

    check(
        "typed request ignores the removed reference edge argument",
        msgspec.convert(
            {"prompt": "A person in a garden.", "duration_s": 5, "reference_image_short_edge": 768},
            type=package.StandardClipInput,
        ),
        package.StandardClipInput(prompt="A person in a garden.", duration_s=5),
    )


def arm_zero_reference_preparation() -> None:
    print("\n== text-only request reaches the official denoise loop ==")
    pipe = meta_h3_pipeline()
    # Only preparation executes: synthetic text embeddings and a CPU scope stand
    # in for the preceding encoder and GPU. No model forward or weights are read.
    pipe.components["fl2va_dit"] = SimpleNamespace(device=torch.device("cpu"))

    def start(steps: int, frames: int = DEFAULT_FRAMES) -> Any:
        return pipe.start_fl2va(
            prompt="Three friends walk in a garden.",
            first_frame=None,
            last_frame=None,
            generator=torch.Generator().manual_seed(7),
            steps=steps,
            frames=frames,
        )

    refusal("an unserved step count refuses before preparation", lambda: start(29), "steps")
    check(
        "the declared request default has an artifact schedule",
        package.DEFAULT_STEPS in STEPS,
        True,
    )
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
    check(
        "text-only packed rows are the default length's",
        int(state.latents.shape[0]) + int(state.audio_latents.shape[0]),
        denoise_rows(DEFAULT_FRAMES, CANVAS_HEIGHT, CANVAS_WIDTH),
    )
    check(
        "text-only latent widths",
        (state.latents.shape[1], state.audio_latents.shape[1]),
        (96, 32),
    )
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


def arm_clip_length() -> None:
    print("\n== clip length is a request field on the model's own frame grid ==")
    # The grid is the video VAE's: `clip_length` pixel frames per chunk keeping
    # `tokens_chunk_size` latents, so a decodable clip is `17n + 5` frames long. The
    # envelope is the pipeline's own [min_duration, max_duration]; the served set is every
    # whole second whose snapped count lands inside it.
    check("served whole seconds", DURATIONS, tuple(range(5, 16)))
    check(
        "each second's frame count",
        [frames_for(seconds) for seconds in DURATIONS],
        [124, 158, 175, 192, 226, 243, 277, 294, 328, 345, 362],
    )
    check(
        "every served count is on the 17n + 5 grid",
        {frames_for(seconds) % 17 for seconds in DURATIONS},
        {5},
    )
    check(
        "the snap is never downward",
        [frames_for(seconds) >= seconds * FPS for seconds in DURATIONS],
        [True] * len(DURATIONS),
    )
    check(
        "the frame envelope holds for every served second",
        [MIN_FRAMES <= frames_for(seconds) <= MAX_FRAMES for seconds in DURATIONS],
        [True] * len(DURATIONS),
    )
    check(
        "the two seconds just outside the served set",
        [
            (seconds, frames_for(seconds), round(frames_for(seconds) / FPS, 3))
            for seconds in (4, 16)
        ],
        [(4, 107, 4.458), (16, 396, 16.5)],
    )
    check(
        "and neither lands inside the frame envelope",
        [MIN_FRAMES <= frames_for(seconds) <= MAX_FRAMES for seconds in (4, 16)],
        [False, False],
    )
    check("the served frame envelope", (MIN_FRAMES, MAX_FRAMES), (124, 362))
    # Derived, not invented: each bound is a declared second carried onto the grid by the
    # same upward snap a request gets, so the envelope cannot drift from what the library
    # declares even though it is no longer expressed in the library's units.
    check(
        "the frame envelope is the declared seconds carried onto the grid",
        (MIN_FRAMES, MAX_FRAMES),
        (frames_for(int(MIN_DURATION)), frames_for(int(MAX_DURATION))),
    )
    # The ceiling is the top grid point of a 15-second model, and it is NOT the clock reading
    # the declared ceiling would give: 362 frames is 15.083 s. Stating the envelope in frames
    # is what admits it; stating it in seconds is what excluded it for the whole of se-047.
    check(
        "the ceiling overshoots the declared seconds, and is served anyway",
        (round(MAX_FRAMES / FPS, 3), MAX_FRAMES / FPS > MAX_DURATION, MAX_FRAMES % 17),
        (15.083, True, 5),
    )
    check(
        "the grid has no point at the declared ceiling",
        (int(MAX_DURATION * FPS) % 17, frames_for(15)),
        (360 % 17, 362),
    )

    # Every length end to end through the official preparation blocks, at the release
    # canvas: the packed sequence the DiT would attend over is exactly what `denoise_rows`
    # reports, so the telemetry number is the model's own and not a restatement.
    pipe = meta_h3_pipeline()
    pipe.components["fl2va_dit"] = SimpleNamespace(device=torch.device("cpu"))

    class ReachedDenoise(Exception):
        pass

    def prepare(frames: int, *, ceiling_override: bool = True) -> Any:
        # `ceiling_override=False` restores upstream's own seconds ceiling for one call, so
        # an arm can show what the official block does without `_CEILING_S` in place.
        if not ceiling_override:
            restore = official_module._CEILING_S
            official_module._CEILING_S = MAX_DURATION
            try:
                return prepare(frames)
            finally:
                official_module._CEILING_S = restore
        state = pipe.start_fl2va(
            prompt="Three friends walk in a garden.",
            first_frame=None,
            last_frame=None,
            generator=torch.Generator().manual_seed(7),
            steps=DEFAULT_STEPS,
            frames=frames,
        )
        state.set("prompt_embeds", torch.zeros(1, 4, 5120))
        state.set("text_token_tags", torch.ones(4, dtype=torch.long))
        try:
            pipe.denoise(
                "fl2va",
                state,
                on_step=lambda _: None,
                cancel=lambda: (_ for _ in ()).throw(ReachedDenoise()),
            )
        except ReachedDenoise:
            return state
        raise AssertionError("preparation must stop before any model forward")

    for seconds in DURATIONS:
        frames = frames_for(seconds)
        state = prepare(frames)
        rows = int(state.latents.shape[0]) + int(state.audio_latents.shape[0])
        check(
            f"{seconds}s prepares {frames} frames as one packed sequence",
            (int(state.num_frames), rows),
            (frames, denoise_rows(frames, CANVAS_HEIGHT, CANVAS_WIDTH)),
        )
        del state

    # The floor is upstream's: one grid step below the wire enum is refused by the official
    # block itself, and we do not override that.
    refusal(
        "a clip below the floor refuses in the official preparation",
        partial(prepare, frames_for(4)),
    )
    # The CEILING is ours, deliberately (se-053). Both official layout blocks hold the
    # ceiling against `aligned_frames / fps`, and the `17n + 5` grid has no point at
    # `15.0 * 24 = 360` — so upstream's own bound makes a 15-second model structurally
    # incapable of a 15-second clip, while its floor tolerates exactly the same upward snap
    # (124 frames = 5.167 s passes a 5.0 s floor). That asymmetry is an oversight, not a
    # capability bound: v1's ie#658 shipped, billed and pixel-checked the 362-frame cell.
    # `_CEILING_S` hands the blocks the ceiling in the units the grid actually has, per
    # request, through the same `_ScopedPipeline` seam the schedulers and image edges use.
    # These two arms are the override's evidence: WITHOUT it the top served length refuses
    # upstream, WITH it the same length prepares.
    ceiling = frames_for(max(DURATIONS))
    refusal(
        "the top served length refuses upstream when the ceiling is left in seconds",
        partial(prepare, ceiling, ceiling_override=False),
    )
    state = prepare(ceiling)
    check(
        "and prepares under the scoped frame ceiling",
        (int(state.num_frames), int(state.latents.shape[0]) + int(state.audio_latents.shape[0])),
        (362, denoise_rows(362, CANVAS_HEIGHT, CANVAS_WIDTH)),
    )
    del state
    # One grid step ABOVE the ceiling still refuses — the override raises the bound to the
    # top grid point, it does not remove it.
    refusal(
        "a clip above the served ceiling refuses in the official preparation",
        partial(prepare, frames_for(16)),
    )

    # `ref2va` gates on the same seconds ceiling in a DIFFERENT block — its before-encode
    # setup step, not the denoise layout step — so the scope above proves nothing about it.
    # The wire offers the ceiling on both entrypoints, so both are shown here.
    references = [MiniMaxH3ImageReference(image=np.zeros((64, 64, 3), dtype=np.uint8))]

    def start_ref2va(frames: int, *, ceiling_override: bool = True) -> Any:
        if not ceiling_override:
            restore = official_module._CEILING_S
            official_module._CEILING_S = MAX_DURATION
            try:
                return start_ref2va(frames)
            finally:
                official_module._CEILING_S = restore
        return pipe.start_ref2va(
            prompt="A person in a garden.",
            references=references,
            generator=torch.Generator().manual_seed(7),
            steps=DEFAULT_STEPS,
            frames=frames,
            reference_image_short_edges=[768],
        )

    refusal(
        "ref2va at the top served length refuses upstream without the scoped ceiling",
        partial(start_ref2va, ceiling, ceiling_override=False),
    )
    check(
        "ref2va prepares the top served length under the scoped frame ceiling",
        int(start_ref2va(ceiling).num_frames),
        362,
    )
    refusal(
        "ref2va above the served ceiling still refuses",
        partial(start_ref2va, frames_for(16)),
    )

    field = get_type_hints(package.StandardClipInput, include_extras=True)["duration_s"]
    check(
        "every served second decodes typed",
        [msgspec.convert(seconds, type=field) for seconds in DURATIONS],
        list(DURATIONS),
    )
    for invalid in (0, 4, 16, 60):
        refusal(
            f"an unserved length {invalid} refuses typed at decode",
            partial(msgspec.convert, invalid, type=field),
        )
    refusal(
        "a fractional length refuses typed at decode",
        partial(msgspec.convert, 5.5, type=field),
    )
    # Owner ruling 2026-09-27: every generation call names its length; none defaults.
    for wire in (package.StandardClipInput, package.ClipInput):
        refusal(
            f"{wire.__name__} without duration_s refuses at decode",
            partial(msgspec.convert, {"prompt": "x"}, type=wire),
        )
    # `mute` never skipped audio generation: `decode_audio` ran regardless and the audio rows
    # denoised in the same packed sequence at every step, so the flag only suppressed the mux
    # while costing the caller the same time and money (se-052). It is deleted, and an
    # unknown field is ignored, not refused.
    for name, request in (("standard", package.StandardClipInput), ("turbo", package.ClipInput)):
        check(
            f"{name} ignores a mute field on the wire",
            msgspec.convert({"prompt": "x", "duration_s": 5, "mute": True}, type=request),
            request(prompt="x", duration_s=5),
        )
    # A plan holds one row per (timestep, modality); no row depends on the frame count, so
    # a served length never needs a re-tabled checkpoint.
    for task in ("fl2va", "ref2va"):
        plan = canonical_timestep_plan(cast(Any, task))
        check(f"{task} plan identity is length-independent", plan.digest, PLAN_DIGESTS[task])


def arm_graph_and_dtypes() -> None:
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
    cast(dict[str, Any], wrong_plan["cozy_h3"])["table_keys"] = {}
    refusal(
        "an AdaLN-pruned checkpoint needs valid row labels",
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
        "video VAE constructs uniformly fp32",
        Counter(str(value.dtype) for value in video_vae.state_dict().values()),
        Counter({"torch.float32": 703}),
    )
    with torch.device("meta"):
        supplied = {
            f"video_vae.{name}": "f16" if decode_operand(name, tuple(value.shape)) else "f32"
            for name, value in video_vae.state_dict().items()
        }
        cast_vae = _apply_video_vae_dtype(
            AutoencoderKLMiniMaxH3(), Config({}, tensor_dtypes=supplied)
        )
        original_vae = _apply_video_vae_dtype(AutoencoderKLMiniMaxH3())
    check(
        "video VAE accepts original FP32 without checkpoint rewriting",
        Counter(str(value.dtype) for value in original_vae.state_dict().values()),
        Counter({"torch.float32": 703}),
    )
    cast_state = cast_vae.state_dict()
    cast_counts = Counter(str(value.dtype) for value in cast_state.values())
    check(
        "video VAE stores the fp16-autocast decode operands at fp16",
        cast_counts,
        Counter({"torch.float32": 484, "torch.float16": 219}),
    )
    check(
        "video VAE encode side stays fp32 - encode_vae_condition has no autocast",
        sorted(
            name
            for name, value in cast_state.items()
            if value.dtype is torch.float16 and name.startswith(("encoder.", "quant_conv."))
        ),
        [],
    )
    check(
        "video VAE rotary buffer stays fp32",
        str(cast_vae.decoder.rope.inv_freq.dtype),
        "torch.float32",
    )
    # `vae_tiles.decode_chunks` keys the incoming latents' dtype off the decoder's FIRST
    # parameter, so parameter order decides what the served path casts them to.
    first_name, first_parameter = next(iter(cast_vae.decoder.named_parameters()))
    check(
        "the decoder's first parameter is a float32 one, which decode_chunks reads",
        (first_name, str(first_parameter.dtype)),
        ("register_tokens", "torch.float32"),
    )
    check(
        "video VAE destination bytes",
        (
            sum(value.numel() * 4 for value in cast_state.values()),
            sum(value.numel() * value.element_size() for value in cast_state.values()),
        ),
        (10_415_475_936, 5_570_955_360),
    )
    red("uniform fp16 video VAE cast", cast_counts, Counter({"torch.float16": 703}))
    # The producer stores what the code destines, so the two spellings of "which rows"
    # must be one rule. They live in different wheels, so this is where they meet.
    served = {name for name, value in cast_state.items() if value.dtype is torch.float16}
    produced = {
        name
        for name, value in video_vae.state_dict().items()
        if decode_operand(name, tuple(value.shape))
    }
    check("the producer's cast scope IS the served destination", produced == served, True)
    check("the cast scope selects the 219 decode operands", len(produced), 219)
    check(
        "the producer normalises the video VAE at f16 on every lane",
        {name: (t.cast, t.cast_scope) for name, t in NORMALISED_COMPONENTS.items()},
        {"video_vae": ("f16", "decode_operands")},
    )
    red(
        "an unscoped component cast would store the served destination",
        len(video_vae.state_dict()),
        len(served),
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


def arm_conditioner_lifecycle() -> None:
    print("\n== every conditioner forward releases the autoregressive position cache ==")
    torch.manual_seed(11)
    model = build_text_conditioner(tiny_text_config())
    arguments = {
        "input_ids": torch.tensor([[1, 62, 60, 60, 60, 60, 63, 5]]),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
        "pixel_values": torch.zeros((16, 24), dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "use_cache": False,
        "output_hidden_states": True,
    }
    with torch.inference_mode():
        full = model.model(**arguments).hidden_states
    check("the stack keeps 50 layers plus the embedding state", len(full), 51)
    check("a full forward releases its position cache", model.model.rope_deltas is None, True)
    # The release is a forward hook, so it runs on whichever group rank hosts the conditioner.
    bare = build_text_conditioner(tiny_text_config())
    bare.model._forward_hooks.clear()
    with torch.inference_mode():
        bare.model(**arguments)
    red("upstream without the hook keeps its position cache", bare.model.rope_deltas is None, True)
    conditioner = FinalHiddenState(model)
    for index in range(4):
        with torch.inference_mode():
            hidden = conditioner.model(**arguments).hidden_states
        check(
            f"conditioner call {index + 1} releases its generation cache",
            model.model.rope_deltas is None,
            True,
        )
        check(
            f"conditioner call {index + 1} returns exactly hidden_states[50]",
            (list(hidden), torch.equal(hidden[50], full[50])),
            ([50], True),
        )

    def refuse(*_: Any) -> None:
        raise RuntimeError("conditioner failure")

    handle = model.model.language_model.layers[0].register_forward_pre_hook(refuse)
    try:
        with torch.inference_mode():
            conditioner.model(**arguments)
    except RuntimeError as exc:
        check("conditioning failure is preserved", str(exc), "conditioner failure")
    else:
        fail("conditioning failure", "the fixture unexpectedly succeeded")
    finally:
        handle.remove()
    check(
        "failed conditioning also releases its position cache",
        model.model.rope_deltas is None,
        True,
    )


def arm_text_conditioner() -> None:
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
    expected_vision = Qwen3VLVisionRotaryEmbedding(rotary_model.config.vision_config)
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
    vision_positions = torch.tensor([[0, 31, 68], [0, 31, 90]])
    vision_input = torch.zeros(1, 3, 72, dtype=torch.bfloat16)
    for name, actual, expected in zip(
        ("cos", "sin"),
        rotary_model.model.visual.rotary_pos_emb(vision_input, vision_positions),
        expected_vision(vision_input, vision_positions),
        strict=True,
    ):
        check(
            f"vision {name} matches upstream at reference image positions",
            torch.equal(actual, expected),
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
        check(
            relative,
            hashlib.sha256(
                files("minimax_h3").joinpath(relative).read_bytes()
            ).hexdigest(),
            expected,
        )
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


def _clocked(slots: int, *, soundtrack: Any | None = None, repeat: int = 1) -> Any:
    """A clip as `VideoAsset.read(fps=24)` returns it: one slot per 1/24 s, and a source frame
    that fills `repeat` slots repeated by reference."""
    distinct = [bytes([index, 0, 0]) for index in range(-(-slots // repeat))]
    return DecodedVideo(
        width=1,
        height=1,
        frame_count=slots,
        frames_rgb=tuple(distinct[slot // repeat] for slot in range(slots)),
        frame_pts=tuple(range(slots)),
        frame_durations=(1,) * slots,
        time_base=Fraction(1, FPS),
        pixel_aspect_ratio=Fraction(1),
        soundtrack=soundtrack,
    )


def mp4_clip(
    path: Path,
    *,
    size: tuple[int, int],
    fps: int,
    seconds: float,
    sample_rate: int = 48000,
    channels: int = 2,
    sound: bool = True,
) -> Path:
    """H.264 frames of flat luma `(index * 37) % 256` and AAC from 0, encoded here with PyAV."""
    width, height = size
    with av.open(str(path), "w", format="mp4") as out:
        video = out.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"preset": "ultrafast", "crf": "30", "threads": "2"}
        layout = "stereo" if channels == 2 else "mono"
        audio = out.add_stream("aac", rate=sample_rate, layout=layout) if sound else None
        for index in range(round(seconds * fps)):
            frame = av.VideoFrame(width, height, "yuv420p")
            chroma = frame.planes[1].line_size * (height // 2)
            frame.planes[0].update(bytes([(index * 37) % 256]) * frame.planes[0].line_size * height)
            frame.planes[1].update(bytes([128]) * chroma)
            frame.planes[2].update(bytes([128]) * chroma)
            frame.pts = index
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if audio is not None:
            total = round(seconds * sample_rate)
            tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(total, dtype=np.float32) / sample_rate)
            wave = np.repeat(tone.astype(np.float32)[None], channels, axis=0)
            for offset in range(0, total, 1024):
                chunk = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(wave[:, offset : offset + 1024]), format="fltp", layout=layout
                )
                chunk.sample_rate, chunk.pts = sample_rate, offset
                chunk.time_base = Fraction(1, sample_rate)
                for packet in audio.encode(chunk):
                    out.mux(packet)
            for packet in audio.encode():
                out.mux(packet)
    return path


class ReferencesSeen(msgspec.Struct):
    warnings: list[str]


#: What `read_references` hands back beside its wire result: the prepared references and sizing.
READ: dict[str, Any] = {}


def read_references(
    payload: package.ReferenceRenderInput, assets: package.DirectReferenceAssets
) -> ReferencesSeen:
    """Direct ref2va's reference path, run by the real kernel: hydration under H3's declared
    preparations and bounds, `assets_to_h3_refs`, and the truncation warnings."""
    READ["prepared"], READ["sizing"] = package.assets_to_h3_refs(
        assets, pipe=READ["pipe"], frames=READ["frames"], video_audio=READ["video_audio"]
    )
    return ReferencesSeen(package._direct_truncations(payload, assets))


REFERENCES = App()
REFERENCES.entrypoint(read_references, accelerator=False)


def run_references(
    sources: Sequence[Path | tuple[Path, str, str]],
    *,
    pipe: Any,
    frames: int = MAX_FRAMES,
    video_audio: Sequence[bool] = (),
    work: Path,
) -> tuple[Any, Any, Any]:
    """Each source (or `(path, label, fidelity)`) as one `assets` occurrence of a call;
    `video_audio` says, per video, whether it keeps its sound (`use_audio`)."""
    READ.clear()
    READ.update(pipe=pipe, frames=frames, video_audio=tuple(video_audio))
    grants, wire = {}, []
    for index, source in enumerate(sources):
        path, label, fidelity = source if isinstance(source, tuple) else (source, "", "auto")
        key = f"assets.{index}.asset"
        grants[key] = granted_input(path, input_id=key, order=index)
        wire.append({"asset": grants[key].digest, "label": label, "fidelity": fidelity})
    name = f"references-{len(list(work.iterdir()))}"
    return kernel_attempt(
        REFERENCES.get("read_references"),
        {"prompt": "x", "duration_s": DEFAULT_DURATION_S, "assets": wire},
        # As the executor runs it: each field's declared bound, no narrower deployment cap.
        Invocation(name, work / name, time.monotonic() + 600, assets=grants, max_input_bytes=None),
    )

def tiny_video_vae() -> Any:
    """The real tile-batched VAE class at toy widths, random weights."""
    torch.manual_seed(0)
    vae = TileBatchedVideoVAE(
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=8,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
    ).eval()
    with torch.no_grad():
        for parameter in vae.parameters():
            parameter.normal_(0, 0.02)
    return vae


def tiny_audio_vae() -> Any:
    """The real audio VAE at toy widths; 8 kHz is AAC's floor, so its clip encodes for real."""
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3Audio(
        encoder_dim=8,
        encoder_rates=(2, 2),
        latent_dim=8,
        latent_channels=8,
        decoder_dim=8,
        decoder_rates=(2, 2),
        sampling_rate=8000,
        latents_mean=[0.0] * 8,
        latents_std=[1.0] * 8,
    ).eval()


def finish_scaffold(frames: int) -> tuple[Any, Any]:
    """The official pipeline with tiny REAL VAEs in every workflow and a denoised state:
    `frames` on the 17n + 5 grid at a 96x64 canvas, stereo latents on the release clock."""
    pipe = meta_h3_pipeline()
    video_vae, audio_vae = tiny_video_vae(), tiny_audio_vae()
    for workflow in pipe._pipes.values():
        workflow.update_components(vae=video_vae, audio_vae=audio_vae)
    pipe.components["video_vae"], pipe.components["audio_vae"] = video_vae, audio_vae
    pipe.sample_rate = int(audio_vae.config.sampling_rate)
    latent_frames = 5 * ((frames - 5) // 17) + 2
    audio_latents = frames * pipe.sample_rate // FPS // 4
    state = PipelineState()
    seeds = (torch.Generator().manual_seed(3), torch.Generator().manual_seed(4))
    state.set("latents", torch.randn(1, 24, latent_frames, 4, 6, generator=seeds[0]))
    state.set("audio_latents", torch.randn(2, 8, audio_latents, generator=seeds[1]))
    state.set("output_type", "pt")
    return pipe, state


def arm_media() -> None:
    print("\n== ordered mixed references and exact clocks ==")
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
        "15-second 16:9 video demand at motion size",
        reference_video_vision_tokens(1920, 1080, Fraction(15)),
        3780,
    )
    package.resolve_reference_sizing([], default=1024, video_tokens=32768)
    observe("vision capacity boundary")
    refusal(
        "vision demand above the release budget refuses",
        lambda: package.resolve_reference_sizing(
            [], default=1024, video_tokens=MAX_CONDITIONER_VISION_TOKENS + 1
        ),
        "reference_policy",
    )

    clip = _clocked(48, soundtrack=_audio(1, start=Fraction(1, 2)))
    frames = _frames(clip)
    check("a constant-clock read becomes (T, H, W, 3) uint8 frames", frames.shape, (48, 1, 1, 3))
    check(
        "a source frame repeated by reference fills each of its slots",
        _frames(_clocked(48, repeat=3))[:, 0, 0, 0].tolist(),
        [slot // 3 for slot in range(48)],
    )
    refusal(
        "a read off the 24-fps clock refuses",
        lambda: _frames(replace(clip, time_base=Fraction(1, 30))),
        "reference_policy",
    )
    aligned = _aligned_soundtrack(clip)
    assert aligned is not None
    check("soundtrack aligns to the full video origin", tuple(aligned.shape), (1, 8))
    check(
        "half-second audio offset becomes two silent samples",
        aligned[0, :2].tolist(),
        [0.0, 0.0],
    )

    class Pipe:
        @staticmethod
        def image_reference(value: Any) -> Any:
            return value

        @staticmethod
        def video_reference(value: Any) -> Any:
            return value

        @staticmethod
        def audio_reference(value: Any) -> Any:
            return value

    root = Path(tempfile.mkdtemp(prefix="h3-references-"))
    fixtures = fake_outputs(fake_attempt("h3-reference-fixtures", spool=root / "spool"))
    media: dict[str, Path] = {}
    for name, seconds, sound in (
        ("v1s", 1, False), ("v2", 2, False), ("v7", 7, True), ("v10", 10, True),
        ("v20", 20, True),
    ):
        media[name] = mp4_clip(root / f"{name}.mp4", size=(32, 32), fps=24, seconds=seconds, sound=sound)
    for seconds in (2, 4, 5, 6, 10, 16):
        saved = fixtures.save_audio(
            np.zeros((2, seconds * 16000), np.float32), sample_rate=16000
        )
        media[f"a{seconds}"] = root / f"a{seconds}.flac"
        media[f"a{seconds}"].write_bytes(saved.read_bytes())
    still = fixtures.save_image(PILImage.new("RGB", (32, 32), (90, 30, 200)), format="png")
    media["image"] = root / "image.png"
    media["image"].write_bytes(still.read_bytes())
    work = root / "calls"
    work.mkdir()

    def decode(
        *names: str, frames: int = MAX_FRAMES, video_audio: Sequence[bool] = ()
    ) -> list[Any]:
        _, outcome, _ = run_references(
            [media[name] for name in names],
            pipe=Pipe(),
            frames=frames,
            video_audio=video_audio or [True] * sum(name.startswith("v") for name in names),
            work=work,
        )
        if outcome.terminal != "succeeded":
            raise InvalidRequest(str(outcome.message), code=outcome.code, fields=outcome.fields)
        return list(READ["prepared"])

    def refused(*names: str) -> tuple[str, tuple[str, ...]]:
        _, outcome, _ = run_references([media[name] for name in names], pipe=Pipe(), work=work)
        return outcome.code, tuple(outcome.fields)

    video_ref = VideoAsset("sha256:" + "1" * 64)
    audio_ref = AudioAsset("sha256:" + "2" * 64)
    image_ref = ImageAsset("sha256:" + "3" * 64)
    another_image = ImageAsset("sha256:" + "4" * 64)
    for description, inputs, expected in [
        ("text-only", [], (None, None)),
        ("one unlabelled image", [image_ref], (0, None)),
        ("two positional images", [image_ref, another_image], (0, 1)),
        ("last frame only", [image_ref.with_label("last")], (None, 0)),
        (
            "labels choose roles despite order",
            [image_ref.with_label("last"), another_image.with_label("first")],
            (1, 0),
        ),
        (
            "unlabelled image fills remaining role",
            [image_ref.with_label("last"), another_image],
            (1, 0),
        ),
        (
            "other labels retain positional meaning",
            [image_ref.with_label("woman"), another_image.with_label("product")],
            (0, 1),
        ),
    ]:
        roles = package._keyframe_roles(Assets(inputs))
        check(f"FL2VA {description}", roles, expected)
    mixed: Assets[ImageAsset | VideoAsset | AudioAsset] = Assets(
        [image_ref.with_label("Alice"), video_ref, audio_ref, image_ref.with_label("アリス")]
    )
    policy = package.preflight_reference_media(
        package.StandardClipInput(
            prompt="The two characters meet.", duration_s=DEFAULT_DURATION_S
        ),
        mixed
    )
    check("metadata preflight counts duplicate image occurrences", policy.total, 4)
    check("reference labels preserve arbitrary caller text", mixed.info("アリス").label, "アリス")
    check("naming an occurrence preserves the source handle", image_ref.label, "")
    check(
        "duplicate image references keep separate positions",
        [mixed.info(index).position for index in range(len(mixed))],
        [0, 1, 2, 3],
    )
    prepared = decode("image", "v7", "a2", "image")
    check(
        "model adapter preserves mixed order and both uses of one image",
        (
            [type(value).__name__ for value in prepared],
            prepared[0].tobytes() == prepared[3].tobytes(),
        ),
        (["Image", "DecodedVideo", "DecodedAudio", "Image"], True),
    )
    refusal(
        "metadata preflight refuses audio-only before decoding",
        lambda: package.preflight_reference_media(
            package.StandardClipInput(
                prompt="An audio-only reference.", duration_s=DEFAULT_DURATION_S
            ),
            Assets([audio_ref]),
        ),
        "reference_policy",
    )
    check(
        "videos have no total cap: two 10 s soundtracked clips and 10 s of audio",
        [round(value.duration) for value in decode("v10", "v10", "a10")],
        [10, 10, 10],
    )
    check(
        "standalone audio admits its exact 15-second total",
        len(decode("v2", "a10", "a5")),
        3,
    )
    refusal(
        "16 seconds of standalone audio refuse against the audio modality cap",
        lambda: decode("v2", "a6", "a6", "a4"),
        "reference_policy",
    )
    check("a 16-second audio refuses at hydration, naming it", refused("v2", "a16"),
          ("media_too_long", ("assets.1.asset",)))
    check("a 1-second video refuses at hydration, naming it", refused("v1s", "image"),
          ("media_too_short", ("assets.0.asset",)))
    check(
        "a video keeps its soundtrack only with use_audio",
        [value.soundtrack is None for value in decode("v7", "v7", video_audio=(False, True))],
        [True, False],
    )
    clips = [
        StoryReference("Ann", "character", description="A woman in red."),
        StoryReference("Kick", "video", video=VideoAsset("sha256:" + "5" * 64)),
        StoryReference("Flip", "video", video=VideoAsset("sha256:" + "6" * 64)),
    ]
    check(
        "only a video some call attaches can carry a truncation warning",
        [
            name
            for name, _ in package.named_videos(
                clips, compile_segments(["<Ann> kicks as in <Kick>."], clips)
            )
        ],
        ["<Kick>"],
    )
    long_clip = decode("v20")[0]
    check(
        "a 20-second video reads its first 15 s on the 24-fps clock, soundtrack included",
        (long_clip.frame_count, long_clip.duration, long_clip.soundtrack.duration),
        (360, Fraction(15), Fraction(15)),
    )
    short_call = decode("v20", frames=DEFAULT_FRAMES)[0]
    check(
        "a call reads only the frames the official pipeline keeps",
        (short_call.frame_count, short_call.soundtrack.duration),
        (DEFAULT_FRAMES, Fraction(DEFAULT_FRAMES, FPS)),
    )


    def sizing(
        fidelities: list[str], *, default: int = 1024, video_tokens: int = 0
    ) -> package.ReferenceSizing:
        images = [
            package.ImageSizing(
                f"assets.{index}.asset",
                1024,
                1024,
                None if fidelity == "auto" else package._REFERENCE_FIDELITY_EDGES[fidelity],
            )
            for index, fidelity in enumerate(fidelities)
        ]
        return package.resolve_reference_sizing(images, default=default, video_tokens=video_tokens)

    nine = sizing(["auto"] * 9)
    check("nine default images fit unchanged", (nine.edges, nine.total), ([1024] * 9, 9216))
    stepped = sizing(["auto"] * 9, default=2048)
    check("nine auto images at 2048 step down", (stepped.edges, stepped.total), ([1536] * 9, 20736))
    mixed_sizing = sizing(["low", "high", "auto"])
    check("mixed explicit fidelities retain their sizes", mixed_sizing.edges, [256, 2048, 1024])
    check(
        "each image has its own token demand",
        [image.tokens for image in mixed_sizing.images],
        [64, 4096, 1024],
    )
    check(
        "a smaller request default stays smaller",
        sizing(["high", "auto"], default=512).edges,
        [2048, 512],
    )
    check(
        "off-ladder defaults step down, never up",
        sizing(["auto"] * 9, default=2000).edges,
        [1536] * 9,
    )
    pushed = sizing(["auto"] * 9, default=2048, video_tokens=15120)
    check("video tokens share the image budget", (pushed.edges, pushed.total), ([1024] * 9, 24336))
    for label, hints, video_tokens in (
        ("nine explicit high references", ["high"] * 9, 0),
        ("eight explicit high leave no room for auto", ["high"] * 8 + ["auto"], 0),
        ("explicit images and video exceed budget", ["high"] * 8, 15120),
    ):
        refusal(
            label,
            partial(sizing, hints, video_tokens=video_tokens),
            "reference_policy",
        )
    fidelity_hints: tuple[Literal["low", "medium", "high", "auto"], ...] = (
        "low",
        "medium",
        "high",
        "auto",
    )
    run_references(
        [(media["image"], f"ref-{index}", fidelity) for index, fidelity in enumerate(fidelity_hints)],
        pipe=Pipe(),
        work=work,
    )
    actual_references, actual_sizing = READ["prepared"], READ["sizing"]
    check(
        "public per-occurrence fidelity reaches H3 sizing",
        actual_sizing.edges,
        [256, 1024, 2048, 1024],
    )
    check("fidelity preserves all source occurrences", len(actual_references), 4)
    check(
        "sizing telemetry preserves occurrence identity",
        [item.field for item in actual_sizing.images],
        [f"assets.{index}.asset" for index in range(4)],
    )

    decoded = torch.tensor(
        [
            [
                [[[-0.1, 0.49]], [[0.5, 1.1]], [[0.0, 1.0]]],
                [[[1.0, 0.0]], [[0.25, 0.75]], [[0.1, 0.9]]],
            ]
        ]
    )
    pixels, _ = package._rgb8(torch, decoded)
    check("RGB8 conversion shape", tuple(pixels.shape), (2, 1, 2, 3))
    check("RGB8 clamp and round", pixels[0].flatten().tolist(), [0, 128, 0, 125, 255, 255])

    # The whole tail, for real (h3a-017): tiny official VAEs through the official decode
    # blocks, the real `Outputs` over a real spool, the runtime's own encoder, probe and
    # decoder. The sequential reference is the official whole-clip block quantized whole.
    pipe, state = finish_scaffold(DEFAULT_FRAMES)
    with torch.no_grad():
        reference, _ = package._rgb8(torch, pipe.decode_video("fl2va", state))
    schedule = ScheduleFacts("a" * 64, 30, 31, *[character * 64 for character in "bcde"])
    spool = Path(tempfile.mkdtemp(prefix="h3-finish-"))
    attempt = fake_attempt("h3-finish-receipt", spool=spool)
    telemetry = fake_telemetry(attempt)
    outputs = fake_outputs(attempt)
    finished = package._finish(
        package.H3Model.for_test(pipe=pipe),
        "fl2va",
        state,
        schedule,
        duration_s=DEFAULT_DURATION_S,
        out=outputs,
        tel=telemetry,
        cancel=lambda: None,
    )
    check(
        "finish returns exactly one typed media asset",
        (type(finished.video), len(outputs.saved)),
        (VideoAsset, 1),
    )
    check("the video is the sink's committed mp4", finished.video.media_type, "video/mp4")
    log_events = [event for event in telemetry.events if event.kind == "log"]
    check(
        "finish emits the three ordered proof rows",
        [event.name for event in log_events],
        ["h3 output geometry", "h3 schedule facts", "h3 container facts"],
    )
    logs = {event.name: dict(event.fields) for event in log_events}
    check(
        "Runtime-admitted output geometry receipt",
        logs.get("h3 output geometry"),
        {
            "width": 96,
            "height": 64,
            "frames": DEFAULT_FRAMES,
            "fps": FPS,
            "requested_duration_s": DEFAULT_DURATION_S,
            "duration_seconds": round(DEFAULT_FRAMES / FPS, 3),
            "denoise_rows": denoise_rows(DEFAULT_FRAMES, 64, 96),
            "sample_rate": pipe.sample_rate,
        },
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
    waveform = pipe.decode_audio("fl2va", state)[0].to(torch.float32).contiguous()
    facts = logs.get("h3 container facts") or {}
    check(
        "the container carries every frame and the whole soundtrack as h264 and aac",
        (
            facts.get("video_codec"),
            facts.get("frame_count"),
            facts.get("audio_codec"),
            int(facts.get("audio_decoded_samples") or 0) >= int(waveform.shape[1]),
        ),
        ("h264", DEFAULT_FRAMES, "aac", True),
    )
    decoded = fake_input(finished.video, attempt=attempt, max_decoded_bytes=1 << 30).read()
    played = torch.stack(
        [
            torch.frombuffer(bytearray(frame), dtype=torch.uint8).reshape(64, 96, 3)
            for frame in decoded.frames_rgb
        ]
    )
    error = (played.float() - reference.float()).pow(2).mean()
    psnr = float(10 * torch.log10(255.0**2 / error))
    check(
        "the runtime decodes the mp4 back to the clip's geometry, clock and soundtrack",
        (
            decoded.frame_count,
            (decoded.width, decoded.height),
            decoded.time_base * decoded.frame_durations[0],
            None if decoded.soundtrack is None else decoded.soundtrack.channels,
        ),
        (DEFAULT_FRAMES, (96, 64), Fraction(1, FPS), 2),
    )
    check("the played frames are the source frames within crf 17", psnr > 30, True)
    observe("container round trip", f"psnr={psnr:.1f} dB bytes={finished.video.size_bytes}")
    stages = telemetry.stages()
    check(
        "the tail's stages are all measured",
        [name in stages for name in ("decode_audio", "decode_video")],
        [True] * 2,
    )
    check(
        "finish receipt loses and refuses no Runtime observations",
        (attempt.ring.dropped, attempt.ring.refused),
        (0, 0),
    )

    def residue(spool: Path) -> list[str]:
        return sorted(path.name for path in spool.iterdir() if "video" in path.name)

    def refused_tail(
        name: str,
        pipe: Any,
        state: Any,
        *,
        duration_s: int = DEFAULT_DURATION_S,
        max_output_bytes: int = 256 << 20,
    ) -> Callable[[], Any]:
        spool = Path(tempfile.mkdtemp(prefix=f"h3-finish-{name}-"))
        attempt = fake_attempt(f"h3-finish-{name}", spool=spool, max_output_bytes=max_output_bytes)
        telemetry = fake_telemetry(attempt)

        def run() -> Any:
            try:
                return package._finish(
                    package.H3Model.for_test(pipe=pipe),
                    "fl2va",
                    state,
                    schedule,
                    duration_s=duration_s,
                    out=fake_outputs(attempt),
                    tel=telemetry,
                    cancel=lambda: None,
                )
            finally:
                check(
                    f"{name}: an abandoned stream leaves no video in the spool",
                    residue(spool),
                    [],
                )

        return run

    # The decode is proven against the length the REQUEST asked for, not a constant.
    refusal(
        "a clip whose length is not the requested one refuses before a frame decodes",
        refused_tail("length", pipe, state, duration_s=DURATIONS[1]),
        "output_integrity",
    )
    # Finite weights whose products overflow: the decode refuses the chunk.
    poisoned, poisoned_state = finish_scaffold(DEFAULT_FRAMES)
    with torch.no_grad():
        largest = torch.finfo(torch.float32).max
        poisoned.components["video_vae"].decoder.proj_in.weight.fill_(largest)
    refusal(
        "a non-finite chunk is refused at the chunk, before RGB8 erases it",
        refused_tail("nan", poisoned, poisoned_state),
        "output_integrity",
    )
    refusal(
        "the sink's own refusal reaches the decode thread instead of decoding on",
        refused_tail("bounded", pipe, state, max_output_bytes=4096),
        "output_too_large",
    )

    check("longest single-shot cell", (MAX_FRAMES, FPS), (362, 24))
    check("eight-shot de-duplicated frame count", 8 * MAX_FRAMES - 7, 2889)
    check("eight-shot exact duration", Fraction(8 * MAX_FRAMES - 7, FPS), Fraction(2889, 24))


def _limited_rgb(luma: int) -> int:
    """The RGB value a flat limited-range luma with neutral chroma decodes to."""
    return max(0, min(255, round((luma - 16) * 255 / 219)))


def arm_reference_media() -> None:
    """Reference videos through the real kernel and Runtime handles, at full size.

    A 20 s 1920x1080 30 fps clip with AAC is truncated by Runtime to its first 15 s (the
    adjustment row and the warning log), budgeted at 15 s, and reaches the official pipeline
    at source resolution on the 24-fps clock with 15 s of sound; a 2-minute 1080p60 source
    reads the same way; a 2560x1080 clip is resized once, by the official normalization.
    """
    print("\n== reference videos: Runtime truncation, source resolution, one official resize ==")
    pipe = meta_h3_pipeline()
    root = Path(tempfile.mkdtemp(prefix="h3-reference-media-"))
    work = root / "calls"
    work.mkdir()
    try:
        for seconds, fps in ((20, 30), (120, 60)):
            source = mp4_clip(root / f"{seconds}s.mp4", size=(1920, 1080), fps=fps, seconds=seconds)
            result, outcome, record = run_references(
                [(source, "Kick", "auto")], pipe=pipe, video_audio=[True], work=work
            )
            check(f"{seconds} s {fps} fps 1080p: the reference call succeeds", outcome.terminal, "succeeded")
            if outcome.terminal != "succeeded":
                fail(f"{seconds} s reference call", repr(outcome))
                continue
            (reference,) = READ["prepared"]
            check(
                f"{seconds} s: 360 frames of 15 s at source resolution reach the pipeline",
                (reference.frames.shape, str(reference.frames.dtype), reference.fps),
                ((360, 1080, 1920, 3), "uint8", float(FPS)),
            )
            # Each frame fills the 24-fps slots from its start to the next one's, rounded half up.
            last = next(
                index
                for index in range(15 * fps)
                if math.floor(Fraction(index * FPS, fps) + Fraction(1, 2)) <= 359
                < math.floor(Fraction((index + 1) * FPS, fps) + Fraction(1, 2))
            )
            check(
                f"{seconds} s: the last slot is source frame {last}, by the 24-fps clock rule",
                abs(int(reference.frames[359, 540, 960, 0]) - _limited_rgb(last * 37 % 256)) <= 3,
                True,
            )
            check(
                f"{seconds} s: Runtime records the truncation as a policy adjustment",
                [
                    (row.kind, row.field, float(str(row.requested)), row.applied, row.source)
                    for row in result.adjustments
                ],
                [("adjusted", "assets.0.asset", float(seconds), 15, "policy")],
            )
            check(
                f"{seconds} s: Runtime logs the truncation as a warning",
                [o.name for o in record.ring if o.kind == "log" and o.value == "warn"],
                ["input video truncated"],
            )
            check(
                f"{seconds} s: H3's own result warns",
                result.result.warnings,
                [f"Reference video Kick is {seconds:.1f} s long; H3 used its first 15 s."],
            )
            check(
                f"{seconds} s: the vision budget counts the 15 s Runtime keeps",
                READ["sizing"].video_tokens,
                reference_video_vision_tokens(1920, 1080, Fraction(15)),
            )
            check(
                f"{seconds} s: the soundtrack is 15 s, stereo, at the audio VAE's rate",
                (tuple(reference.audio.shape), reference.sample_rate),
                ((2, 15 * pipe.sample_rate), pipe.sample_rate),
            )
            del reference
            READ.clear()
        wide = mp4_clip(root / "wide.mp4", size=(2560, 1080), fps=24, seconds=3, sound=False)
        result, outcome, _ = run_references([wide], pipe=pipe, frames=DEFAULT_FRAMES, work=work)
        check("2560x1080: the reference call succeeds", outcome.terminal, "succeeded")
        (reference,) = READ["prepared"]
        check(
            "2560x1080: the clip reaches the pipeline unresized, with no adjustment",
            (reference.frames.shape, result.adjustments, result.result.warnings),
            ((72, 1080, 2560, 3), (), []),
        )
        state = pipe.start_ref2va(
            prompt="A dancer spins.",
            references=[reference],
            generator=torch.Generator().manual_seed(7),
            steps=DEFAULT_STEPS,
            frames=DEFAULT_FRAMES,
            reference_image_short_edges=[],
        )
        normalized = state.normalized_references[0].frames
        height, width = (int(v) for v in resolve_canvas_size(2560, 1080, *official.CANVAS))
        canvas = PILImage.fromarray(reference.frames[0]).resize(
            (width, height), PILImage.Resampling.LANCZOS
        )
        motion_height, motion_width = motion_size(height, width)
        guide = np.asarray(canvas.resize((motion_width, motion_height), PILImage.Resampling.LANCZOS))
        check(
            "2560x1080: the official canvas LANCZOS resize, then one to its motion size",
            (normalized.shape, bool(np.array_equal(normalized[0], guide))),
            ((72, motion_height, motion_width, 3), True),
        )
        READ.clear()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def arm_video_stream() -> None:
    """Reject changed channels, decoder infinities and cancellation through the real sink."""

    def stream_at(spool: Path, frames: int, cancel: Callable[[], None]) -> Any:
        attempt = fake_attempt("tail-integrity", spool=spool)
        return package._VideoStream(
            torch,
            fake_outputs(attempt),
            frames=frames,
            waveform=torch.zeros(2, frames * 24000 // FPS),
            sample_rate=24000,
            cancel=cancel,
            progress=lambda _: None,
        )

    with tempfile.TemporaryDirectory(prefix="h3-stream-channels-") as directory:
        spool = Path(directory)
        stream = stream_at(spool, 2, lambda: None)
        try:
            stream.push(torch.full((1, 3, 16, 16), 0.5))
            refusal(
                "later grayscale chunks cannot silently broadcast into RGB",
                lambda: stream.push(torch.full((1, 1, 16, 16), 0.25)),
                "output_integrity",
            )
        finally:
            stream.abandon()
        check("channel refusal leaves no partial video", list(spool.iterdir()), [])

    pipe, state = finish_scaffold(39)
    injected = False

    def infinity(_module: Any, _args: Any, output: Any) -> Any:
        nonlocal injected
        if not injected:
            output = output.clone()
            # This frame survives temporal padding and lies outside the overlap blend.
            output[:, :, 8, 0, 0] = float("inf")
            injected = True
        return output

    handle = pipe.components["video_vae"].decoder.register_forward_hook(infinity)
    with tempfile.TemporaryDirectory(prefix="h3-stream-infinity-") as directory:
        spool = Path(directory)
        stream = stream_at(spool, 39, lambda: None)

        def encode() -> Any:
            pipe.decode_video_chunks("fl2va", state, stream.push)
            return stream.finish()

        try:
            refusal("decoder infinity refuses before clipping", encode, "output_integrity")
        finally:
            stream.abandon()
            handle.remove()
        check("the real decoder emitted the planted infinity", injected, True)
        check("infinity refusal leaves no partial video", list(spool.iterdir()), [])

    caller = get_ident()
    cancelled = Event()
    context = Context("tail-encoding", float("inf"), _cancel=cancelled.is_set)

    def cancel_in_encoder() -> None:
        # Hold the sink at its first cancellation check until the caller signals the
        # real Context. This avoids a timing race with the two-frame encode.
        if get_ident() != caller:
            cancelled.wait()
        context.raise_if_cancelled()

    with tempfile.TemporaryDirectory(prefix="h3-stream-cancel-") as directory:
        spool = Path(directory)
        stream = stream_at(spool, 2, cancel_in_encoder)
        try:
            stream.push(torch.full((2, 3, 16, 16), 0.5))
            cancelled.set()
            refusal("encoding observes cancellation", stream.finish, "cancelled")
        finally:
            cancelled.set()
            stream.abandon()
        check("cancelled encoding leaves no partial video", list(spool.iterdir()), [])


def arm_output_gates() -> None:
    print("\n== structural refusals ==")
    clock = MediaFacts(frames=2, fps=24, sample_rate=24)
    refusal(
        "a non-finite soundtrack refuses before a frame decodes",
        lambda: refuse_before_encode(
            waveform=torch.zeros((1, 2)),
            audio_nonfinite_fraction=0.5,
            requested=clock,
            tel=fake_telemetry(),
        ),
        "output_integrity",
    )
    refusal(
        "audio outside the one-frame A/V tolerance refuses",
        lambda: refuse_before_encode(
            waveform=torch.zeros((1, 5)),
            audio_nonfinite_fraction=0.0,
            requested=clock,
            tel=fake_telemetry(),
        ),
        "output_integrity",
    )
    refusal(
        "a soundtrack that is not (channels, samples) refuses",
        lambda: refuse_before_encode(
            waveform=torch.zeros((1, 1, 2)),
            audio_nonfinite_fraction=0.0,
            requested=clock,
            tel=fake_telemetry(),
        ),
        "output_integrity",
    )
    refuse_before_encode(
        waveform=torch.zeros((2, 3)),
        audio_nonfinite_fraction=0.0,
        requested=clock,
        tel=fake_telemetry(),
    )
    observe("a stereo soundtrack one sample past the clock is within the tolerance")


def arm_vae_tiles() -> None:
    """h3a-017: one chunk's tiles decode as one batch, within one ulp of the tile-at-a-time
    decode, on the real VAE class at the release tile geometry (CPU, random weights)."""
    print("\n== tile-batched video VAE decode ==")
    vae = tiny_video_vae()
    check(
        "the served video VAE is the tile-batched class",
        type(meta_h3_pipeline().components["video_vae"]),
        TileBatchedVideoVAE,
    )
    check("tile-batched VAE keeps the release VAE contract", vae.tokens_chunk_size, 5)

    # 1344x768 at the release tile geometry: latent 84x48 -> 4x7 tiles of (1, 24, 7, 16, 16).
    ratio = vae.spatial_compression_ratio
    rows = vae._split_tiles(768, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    columns = vae._split_tiles(1344, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    check("the release grid is 4x7 tiles, one forward per chunk", len(rows[0]) * len(columns[0]), 28)
    clip = torch.randn(
        1, 24, 7, 768 // ratio, 1344 // ratio, generator=torch.Generator().manual_seed(1)
    )
    with torch.no_grad():
        sequential = AutoencoderKLMiniMaxH3._decode_clip(vae, clip)
        batched = TileBatchedVideoVAE._decode_clip(vae, clip)
    check("batched decode has the clip's pixel shape", tuple(batched.shape), (1, 3, 28, 768, 1344))
    # A larger GEMM may reduce in another order, so the bytes are not the identity; the
    # accuracy is. Against the same decode in float64, the batched fp32 result must sit in
    # the sequential fp32 result's error class — a wrong tile, order or dtype is orders of
    # magnitude out, a re-associated dot product is not.
    with torch.no_grad():
        exact = AutoencoderKLMiniMaxH3._decode_clip(copy.deepcopy(vae).double(), clip.double())
    sequential_error = float((sequential.double() - exact).abs().max())
    batched_error = float((batched.double() - exact).abs().max())
    check("the sequential fp32 decode is itself inexact", sequential_error > 0, True)
    check(
        "batched tiles decode in the sequential decode's error class",
        batched_error <= 2 * sequential_error,
        True,
    )
    observe(
        "tile-batch drift",
        f"sequential_vs_fp64={sequential_error:.3e} batched_vs_fp64={batched_error:.3e} "
        f"batched_vs_sequential={float((sequential - batched).abs().max()):.3e}",
    )
    # Red arm: a tile grid stitched in the wrong order is far outside that class.
    with torch.no_grad():
        wrong = TileBatchedVideoVAE._stitch_tiles(
            vae,
            [
                [
                    AutoencoderKLMiniMaxH3._decode_clip(
                        vae, clip[..., y // ratio : y // ratio + 16, x // ratio : x // ratio + 16]
                    )
                    for x in reversed(columns[0])
                ]
                for y in rows[0]
            ],
            rows[2],
            columns[2],
        )
    red(
        "a reversed tile order still sits in the sequential error class",
        float((wrong.double() - exact).abs().max()) <= 2 * sequential_error,
        True,
    )

    # h3a-017 (c): the temporal loop as a generator IS upstream's `_decode`, piece by piece —
    # bit-identical on the served 5n + 2 grid and on every padding branch off it — so a
    # chunk may be quantized and encoded while the next one decodes.
    pieces_on_grid: list[int] = []
    for latent_frames in (7, 8, 11, 12, 16, 22):
        seed = torch.Generator().manual_seed(latent_frames)
        z = torch.randn(1, 24, latent_frames, 4, 6, generator=seed)
        with torch.no_grad():
            upstream = AutoencoderKLMiniMaxH3._decode(vae, z)
            pieces = list(vae.decode_chunks(z))
            public = vae.decode(z, return_dict=False)[0]
        streamed = torch.cat(pieces, dim=2)
        check(
            f"{latent_frames} latent frames stream exactly as upstream decodes them",
            (tuple(streamed.shape), torch.equal(streamed, upstream), torch.equal(public, upstream)),
            (tuple(upstream.shape), True, True),
        )
        if latent_frames == 12:
            pieces_on_grid = [int(piece.shape[2]) for piece in pieces]
    check(
        "a 39-frame clip streams two 17-frame chunks and the 5-frame tail",
        pieces_on_grid,
        [17, 17, 5],
    )
    z = torch.randn(1, 24, 11, 4, 6, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        untrimmed = torch.cat(list(vae._decode_pieces(z, 2)), dim=2)
        upstream = AutoencoderKLMiniMaxH3._decode(vae, z)
    red(
        "the pieces before the padding hold-back already have upstream's length",
        int(untrimmed.shape[2]),
        int(upstream.shape[2]),
    )


def arm_rgb8_handoff() -> None:
    """h3a-017: the chunked, digested-as-it-lands RGB8 handoff is byte-identical to a
    whole-tensor quantize and its digest is the digest of the finished buffer."""
    print("\n== chunked RGB8 handoff ==")
    frames = 3 * package._RGB8_CHUNK_FRAMES + 1  # three full chunks and a ragged tail
    decoded = torch.rand((1, frames, 3, 16, 32), generator=torch.Generator().manual_seed(2))
    decoded = decoded * 1.2 - 0.1  # excursions past [0, 1] on both sides
    whole = (decoded[0].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    pixels, digest = package._rgb8(torch, decoded.clone())
    check("chunked RGB8 equals the whole-tensor quantize", torch.equal(pixels, whole), True)
    check(
        "chunked RGB8 lands in one contiguous host buffer",
        (pixels.is_contiguous(), pixels.device.type),
        (True, "cpu"),
    )
    check(
        "the running digest is the finished buffer's sha256",
        digest,
        hashlib.sha256(pixels.numpy()).hexdigest(),
    )
    # Red arm: chunks hashed out of order are a different digest, so the check has teeth.
    reordered = hashlib.sha256()
    for start in reversed(range(0, frames, package._RGB8_CHUNK_FRAMES)):
        reordered.update(pixels[start : start + package._RGB8_CHUNK_FRAMES].numpy())
    red("chunks digested out of order still match", reordered.hexdigest(), digest)


def arm_interface() -> None:
    check("wire step values come from both verified task plans", package.SUPPORTED_STEPS, STEPS)
    check("the default is the shortest shipped schedule", package.DEFAULT_STEPS, min(STEPS))
    print("\n== committed public surface ==")
    interface_path = H3 / "metadata" / "package-interface.json"
    interface = json.loads(interface_path.read_text())
    check("one entrypoints list", "jobs" in interface, False)
    functions = {entry["name"]: entry for entry in interface["entrypoints"]}
    workflows = {n: e for n, e in functions.items() if e.get("accelerator") is False}
    entries = {n: e for n, e in functions.items() if n not in workflows}
    surfaces = {surface.name: surface for surface in describe(package.app)}
    check(
        "official actions, turbo functions, the model-bearing segments and encode_text",
        set(entries),
        {
            "encode_text",
            "generate_fl2va", "generate_ref2va", "generate_fl2va_turbo", "generate_ref2va_turbo",
            "cut_segment", "motion_segment",
        },
    )
    check(
        "GPU stages remain internal to their CPU workflows",
        {name for name, entry in entries.items() if entry.get("internal", False)},
        {
            "generate_fl2va", "generate_ref2va", "generate_fl2va_turbo", "generate_ref2va_turbo",
            "cut_segment", "motion_segment",
        },
    )
    check(
        "public serving functions exclude chain implementation details",
        {name for name, entry in entries.items() if not entry.get("internal", False)},
        {"encode_text"},
    )
    check(
        "six workflows over four tasks",
        official._WORKFLOW_TASKS,
        {
            "t2va": "fl2va",
            "fl2va": "fl2va",
            "ref2va": "ref2va",
            "t2va_turbo": "fl2va_turbo",
            "fl2va_turbo": "fl2va_turbo",
            "ref2va_turbo": "ref2va_turbo",
        },
    )
    expected = {
        "fl2va": (["prompt", "seed", "duration_s", "steps", "assets"], ["fl2va_dit"]),
        "ref2va": (["prompt", "seed", "duration_s", "steps", "assets"], ["ref2va_dit"]),
        "fl2va_turbo": (["prompt", "seed", "duration_s", "assets"], ["fl2va_dit", "fl2va_turbo"]),
        "ref2va_turbo": (
            ["prompt", "seed", "duration_s", "assets"],
            ["ref2va_dit", "ref2va_turbo"],
        ),
    }
    for name, (fields, _leased) in expected.items():
        entry = workflows[name]
        child = entries["generate_" + name]
        check(
            f"{name} request fields",
            [field["name"] for field in entry["request"]["fields"]],
            fields[:3] + ["resolution", "upscale"]
            + (["references", "style", "clean_references", "describe_references"]
               if name.startswith("ref2va") else [])
            + fields[3:],
        )
        check(f"{name} resolution is a closed enum",
              next(f["type"] for f in entry["request"]["fields"] if f["name"] == "resolution"),
              {"literal": ["1080p", "1440p", "2160p", "768p"]})
        check(f"{name} has no GPU model slot on its CPU workflow", "models" in entry, False)
        if name.startswith("ref2va"):
            check(
                f"{name} prepares and describes named characters unless told not to",
                [f.get("default") for f in entry["request"]["fields"]
                 if f["name"] in ("clean_references", "describe_references")],
                [True, True],
            )
        if name.endswith("_turbo"):
            check(
                f"{name} has no steps on its wire: the plan fixes {package.TURBO_STEPS}",
                any(field["name"] == "steps" for field in entry["request"]["fields"]),
                False,
            )
        else:
            check(
                f"{name} wire steps come from the shipped plans",
                next(
                    field["type"]
                    for field in entry["request"]["fields"]
                    if field["name"] == "steps"
                ),
                {"literal": list(STEPS)},
            )
        duration = next(
            field for field in entry["request"]["fields"] if field["name"] == "duration_s"
        )
        check(
            f"{name} clip length is required whole seconds bounded by the served envelope",
            (
                duration["type"],
                duration["constraints"],
                duration.get("wire", "required"),
                "default" in duration,
            ),
            ("int", {"ge": min(DURATIONS), "le": max(DURATIONS)}, "required", False),
        )
        # Staged admission reuses a scope's measured peak only within one shape cell, so the
        # cell must name what sizes memory: frames and the reference count, never the prompt.
        cells = {
            (seconds, count): dict(
                normalize(
                    decode_request(
                        cast(type[Any], surfaces[name].payload_type),
                        {
                            "prompt": "a garden",
                            "duration_s": seconds,
                            "assets": [{"asset": f"input:{i}"} for i in range(count)],
                        },
                    )
                ).values
            )
            for seconds in (min(DURATIONS), max(DURATIONS))
            for count in (1, 2)
        }
        steps = {} if name.endswith("_turbo") else {"steps": package.DEFAULT_STEPS}
        check(
            f"{name} shape cell is frames and reference count",
            cells,
            {
                (seconds, count): {
                    "assets": count,
                    "frames": official.frames_for(seconds),
                    **steps,
                }
                for seconds, count in cells
            },
        )
        turbo = name.endswith("_turbo")
        models = child["models"]
        trunk = name.removesuffix("_turbo")
        if turbo:
            check(f"{name} has separately bound base and LoRA models", len(models), 2)
            base_slot, lora_slot = models
            check(f"{name} base model class", base_slot["class"], "H3TurboBase")
            check(f"{name} LoRA model class", lora_slot["class"], "H3TurboLoRA")
            check(
                f"{name} base turbo sampling lease",
                base_slot["component_use"][f"sample_{trunk}_turbo"],
                [f"{trunk}_dit"],
            )
            check(
                f"{name} LoRA sampling lease",
                lora_slot["component_use"][f"sample_{trunk}"],
                [f"{trunk}_turbo"],
            )
            check(
                f"{name} does not advertise uncalled turbo warm scopes",
                any("warm" in method for method in lora_slot["component_use"]),
                False,
            )
            component_use = {**base_slot["component_use"], **lora_slot["component_use"]}
            model_slots = models
        else:
            check(f"{name} one complete model", len(models), 1)
            model_slots = models
            component_use = models[0]["component_use"]
        for model_slot in model_slots:
            check(
                f"{name} {model_slot['class']} encoded leaves",
                model_slot["encoded_leaves"],
                "accept",
            )
        roots = {"fl2va_dit", "ref2va_dit", "text_encoder", "video_vae", "audio_vae"}
        if turbo:
            roots.update(("fl2va_turbo", "ref2va_turbo"))
        check(
            f"{name} declared roots",
            {value for values in component_use.values() for value in values},
            roots,
        )
        check(f"{name} media capability", "media_decode" in surfaces[name].capabilities, True)
        check(
            f"{name} exact customer result fields",
            [field["name"] for field in entry["result"]["fields"]],
            ["video", "originals", "warnings"] + (["references"] if name.startswith("ref2va") else []),
        )
    for wire in (package.ClipInput,):
        check(
            f"{wire.__name__} cannot represent steps; the field is ignored",
            msgspec.convert({"prompt": "x", "duration_s": 5, "steps": 8}, wire),
            wire(prompt="x", duration_s=5),
        )
        check(
            f"{wire.__name__} keeps the base request's other fields",
            list(wire.__struct_fields__),
            ["prompt", "seed", "duration_s"],
        )
    check("the turbo functions run the plans' eight evaluations", package.TURBO_STEPS, 8)
    red("eight is not a base step count", package.TURBO_STEPS in STEPS, True)
    print("\n== long-form composition ==")
    check(
        "composition, assembly and the direct workflows are CPU functions",
        set(workflows),
        {"long_form", "long_form_cuts", "assemble_video", "fl2va", "fl2va_turbo",
         "ref2va", "ref2va_turbo"},
    )
    # The one property decision #601 turns on: the composer holds NO device while its shots
    # render. A model slot here would make one attempt hold eight shots.
    check("long_form declares no model slot", "models" in workflows["long_form"], False)
    check(
        "every segment, turbo or standard, holds one turbo-capable base and its adapter",
        {
            name: [slot["class"] for slot in entries[name]["models"]]
            for name in ("cut_segment", "motion_segment")
        },
        {
            "cut_segment": ["H3TurboBase", "H3TurboLoRA"],
            "motion_segment": ["H3TurboBase", "H3TurboLoRA"],
        },
    )
    check(
        "motion_segment is child-callable by its exact module and export",
        (
            entries["motion_segment"]["invocable"]["module"],
            entries["motion_segment"]["invocable"]["export"],
        ),
        ("h3", "motion_segment"),
    )
    check(
        "a shot's identity is frozen in its own request",
        [field["name"] for field in entries["motion_segment"]["request"]["fields"]],
        ["payload", "assets"],
    )
    # A child call names its whole intent: only context and provenance may be omitted,
    # because a default would put a value into the intent digest that the caller never wrote.
    check(
        "a shot's prompt, seed, length and steps are all named, never defaulted",
        sorted(entries["motion_segment"]["invocable"]["defaults"]),
        [
            "request/assets/[]/fidelity",
            "request/assets/[]/label",
            "request/payload/context",
            "request/payload/context_audio",
            "request/payload/context_frames",
            "request/payload/expected_provenance",
            "request/payload/expected_provenance/union/1/turbo_lora_manifest",
            "request/payload/next_context_audio",
            "request/payload/next_context_frames",
            "request/payload/previous_provenance",
            "request/payload/previous_provenance/union/1/turbo_lora_manifest",
            "result/provenance/turbo_lora_manifest",
        ],
    )
    # Run 1560: a 12 s continuation shared a 6 s opener's shape and ran on its peak.
    opener = ContinuationPlan(6)
    follower = ContinuationPlan(12, Carry(56, True))
    check(
        "a segment's shape is the frames its DiT holds: window plus conditioning context",
        [
            dict(
                normalize(
                    package.MotionInput(
                        prompt="x",
                        seed=1,
                        duration_s=seconds,
                        steps=package.TURBO_STEPS,
                        turbo=True,
                        frames=package.held_frames(plan),
                        video_audio=(),
                    )
                ).values
            )
            for seconds, plan in ((6, opener), (12, follower))
        ],
        [
            {"frames": 158, "steps": 8},
            {"frames": 345 + 56, "steps": 8},
        ],
    )
    adapter = object()

    def sampled(turbo: bool, steps: int) -> object:
        segment = package.MotionInput(
            prompt="x", seed=1, duration_s=6, steps=steps, turbo=turbo, frames=158, video_audio=()
        )
        return package._sampling(segment, adapter)  # type: ignore[arg-type]

    check(
        "one construction samples turbo with its adapter and standard with none",
        [sampled(True, package.TURBO_STEPS) is adapter, sampled(False, 30)],
        [True, None],
    )
    for selected, evaluations in ((True, 30), (False, package.TURBO_STEPS)):
        refusal(
            f"turbo={selected} refuses {evaluations} steps",
            partial(sampled, selected, evaluations),
            "invalid_request",
        )
    segment_fields = next(
        field for field in workflows["long_form"]["request"]["fields"] if field["name"] == "segments"
    )["type"]["list"]["fields"]
    check(
        "long-form segment seed is optional in the published interface",
        next(field for field in segment_fields if field["name"] == "seed"),
        {"name": "seed", "type": {"union": ["int", "null"]}, "wire": "optional"},
    )
    unseeded = msgspec.json.decode(
        b'{"segments":[{"prompt":"x","duration_s":5},'
        b'{"prompt":"y","duration_s":5,"seed":null}],'
        b'"references":[{"name":"Hero","kind":"character"}]}',
        type=package.LongFormInput,
    )
    check(
        "omitted and null segment seeds are automatic",
        [s.seed for s in unseeded.segments],
        [None, None],
    )
    check("long-form defaults to turbo", unseeded.mode, "turbo")
    check("default segments use eight evaluations", package.segment_sampling(unseeded), (True, 8))
    for job in (package.LongFormInput, package.LongFormCutsInput):
        for encoding in (msgspec.json, msgspec.yaml):
            authored = {
                "segments": [{"prompt": "x", "duration_s": 5, "turbo": value}
                             for value in (True, False)],
                "references": [{"name": "Hero", "kind": "character"}],
            }
            request = encoding.decode(encoding.encode(authored), type=job)
            label = f"{job.__name__} {encoding.__name__}"
            check(f"{label} removed segment flags cannot override global turbo",
                  package.segment_sampling(request), (True, 8))
            request.steps = 30
            refusal(f"{label} turbo refuses explicit steps",
                    partial(package.segment_sampling, request), "invalid_request")
            request.mode = "standard"
            request.steps = None
            check(f"{label} global standard applies to every segment",
                  package.segment_sampling(request), (False, 30))
            request.steps = 40
            check(f"{label} global standard preserves its authored step count",
                  package.segment_sampling(request), (False, 40))
    check("long-form has no contradictory standard step default", unseeded.steps, None)
    check("long-form preserves the global context default", unseeded.context_frames, 22)
    check("segment context defaults to true", [s.context_frames for s in unseeded.segments], [True, True])
    check("segment sound defaults to true", [s.context_audio for s in unseeded.segments], [True, True])
    check("segments carry the global picture and the sound except the first",
          package.segment_carries(unseeded), [Carry(), Carry(22, True)])
    for global_frames, windows in ((None, [0, 22, 0, 22]), (0, [0, 0, 0, 0]), (39, [0, 39, 0, 39])):
        expected_windows = [Carry(n, bool(i)) for i, n in enumerate(windows)]
        authored: dict[str, Any] = {
            "segments": [{"prompt": "x", "duration_s": 15,
                          **({"context_frames": False} if index == 2 else {})}
                         for index in range(4)],
            "references": [{"name": "Hero", "kind": "character"}],
        }
        if global_frames is not None:
            authored["context_frames"] = global_frames
        resolved = msgspec.convert(authored, type=package.LongFormInput)
        check(f"global window with a segment picture opt-out {global_frames}",
              package.segment_carries(resolved), expected_windows)
        explicit = copy.deepcopy(authored)
        for segment in explicit["segments"]:
            segment.setdefault("context_frames", True)
        check(f"explicit true matches omission with global {global_frames}",
              package.segment_carries(msgspec.convert(explicit, type=package.LongFormInput)), expected_windows)
    silent = msgspec.convert(
        {"segments": [{"prompt": "x", "duration_s": 5}] + [
            {"prompt": "x", "duration_s": 5, "context_frames": frames, "context_audio": audio}
            for frames, audio in ((True, False), (False, True), (False, False))
        ], "references": [{"name": "Hero", "kind": "character"}]},
        type=package.LongFormInput,
    )
    check("picture and sound carry independently", package.segment_carries(silent),
          [Carry(), Carry(22), Carry(0, True), Carry()])
    plans = [
        ContinuationPlan(14 if carry else 15, carry, successor)
        for carry, successor in zip(
            package.segment_carries(silent),
            [*package.segment_carries(silent)[1:], Carry()],
            strict=True,
        )
    ]
    check("a segment ends its picture where its successor's carry needs it",
          [(plan.sample_frames, plan.delivered_frames) for plan in plans],
          [(362, 362), (362, 339), (362, 338), (362, 362)])
    for field in ("context_frames", "context_audio"):
        for value in (0, 22, 56, None, "false"):
            invalid = {"segments": [{"prompt": "x", "duration_s": 5, field: value}],
                       "references": [{"name": "Hero", "kind": "character"}]}
            refusal(f"segment {field} refuses {value!r}",
                    partial(msgspec.convert, invalid, type=package.LongFormInput),
                    "ValidationError")
    for workflow in ("long_form", "long_form_cuts"):
        segment = next(f for f in workflows[workflow]["request"]["fields"] if f["name"] == "segments")
        check(
            f"a {workflow} segment preserves the turbo request fields",
            [(f["name"], f["type"]) for f in segment["type"]["list"]["fields"]
             if f["name"] not in {"context_frames", "context_audio", "non_diegetic_music"}],
            [
                (f["name"], f["type"])
                for f in workflows["ref2va_turbo"]["request"]["fields"]
                if f["name"] not in {"assets", "resolution", "upscale", "references",
                                      "style", "clean_references", "describe_references"}
            ],
        )
        check(
            f"{workflow} compiler exposes no segment sampling override",
            any(f["name"] == "turbo" for f in segment["type"]["list"]["fields"]),
            False,
        )
        check(
            f"{workflow} segments expose optional music overrides",
            next(f for f in segment["type"]["list"]["fields"] if f["name"] == "non_diegetic_music"),
            {"name": "non_diegetic_music", "type": {"union": ["null", "str"]}, "wire": "optional"},
        )
        check(
            f"per-segment picture and sound carry belong only to long_form ({workflow})",
            [f["name"] for f in segment["type"]["list"]["fields"]
             if f["name"] in {"context_frames", "context_audio"}],
            ["context_frames", "context_audio"] if workflow == "long_form" else [],
        )
    check(
        "long_form request fields",
        [field["name"] for field in workflows["long_form"]["request"]["fields"]],
        [
            "segments",
            "references",
            "style",
            "overall_soundscape",
            "non_diegetic_music",
            "context_frames",
            "mode",
            "steps",
            "clean_references",
            "describe_references",
        ],
    )
    check(
        "long_form returns its video and references, each published as it lands",
        [field["name"] for field in workflows["long_form"]["result"]["fields"]],
        [
            "video",
            "references",
            "delivered_frames",
            "fps",
            "warnings",
        ],
    )
    check(
        "a segment list requires at least one segment without a count cap",
        next(
            field["constraints"]
            for field in workflows["long_form"]["request"]["fields"]
            if field["name"] == "segments"
        ),
        {"min_length": 1},
    )
    check(
        "one replayed frame leaves the chain at every seam",
        package.segment_clock([14, 14, 14, 14, 14, 14, 14, 14]),
        ([0, 345, 689, 1033, 1377, 1721, 2065, 2409], 2753),
    )
    check("H3 permits Runtime encoded linear leaves", package.H3Model.__encoded_leaves__, "accept")
    check(
        "reference files bind to the explicit Assets parameter",
        workflows["ref2va"]["assets"]["parameter"],
        "assets",
    )
    check(
        "reference collection admits only the three H3 media kinds",
        {kind["kind"] for kind in workflows["ref2va"]["assets"]["kinds"]},
        {"image", "video", "audio"},
    )
    check(
        "keyframe endpoint uses the same explicit Assets input",
        workflows["fl2va"]["assets"]["parameter"],
        "assets",
    )
    check(
        "keyframe collection permits at most two images",
        next(
            field["constraints"]
            for field in workflows["fl2va"]["request"]["fields"]
            if field["name"] == "assets"
        ),
        {"max_length": 2},
    )


def tiny_dit() -> Any:
    """The real Diffusers DiT at toy widths: one block, one refiner block, 64 wide."""
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        hidden_size=64,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=128,
        in_channels=4,
        audio_in_channels=8,
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=4,
    ).eval()


# --- PDD-8 turbo -------------------------------------------------------------------------
# The reference (`minimax_h3_pdd.py`, alibaba-pai/MiniMax-H3-Acc-LoRAs rev 335001fb) is the
# oracle: its LoRA wrapper, its 32-interval head bank and its per-step plan, transcribed
# here so a turbo forward of OUR construction is held to a forward of the reference's.


def reference_pdd_plan(step_sizes: Any, start: int, block_size: int) -> Any:
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype)
    span = step_sizes[start : start + block_size].sum()
    plan[0, start : start + block_size] = step_sizes[start : start + block_size] / span
    return plan


class ReferenceParallelHead(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, source: Any, num_steps: int) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.weight = torch.nn.Parameter(
            source.weight.detach()[None].repeat(num_steps, 1, 1).clone()
        )
        self.bias = torch.nn.Parameter(source.bias.detach()[None].repeat(num_steps, 1).clone())
        self.plan = torch.zeros(1, num_steps)
        self.plan[0, 0] = 1.0

    def forward(self, hidden_states: Any) -> Any:
        plan = self.plan.to(device=self.weight.device, dtype=self.weight.dtype)
        weight = torch.einsum("pn,noi->poi", plan, self.weight).flatten(0, 1)
        bias = torch.einsum("pn,no->po", plan, self.bias).flatten()
        return torch.nn.functional.linear(hidden_states, weight, bias)


class ReferenceLoRALinear(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, base: Any, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_down = torch.nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_up = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        torch.nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    def forward(self, hidden_states: Any) -> Any:
        out = self.base(hidden_states)
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(hidden_states, self.lora_down.to(hidden_states.dtype)),
            self.lora_up.to(hidden_states.dtype),
        )
        return out + self.scaling * update.to(out.dtype)


def reference_add_lora(module: Any, targets: Sequence[str], rank: int, alpha: float) -> int:
    sites = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Linear) and any(name.endswith(s) for s in targets)
    ]
    for name, child in sites:
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, ReferenceLoRALinear(child, rank, alpha))
    return len(sites)


TURBO_CONFIG = {
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
TURBO_RANK, TURBO_ALPHA = 4, 4.0


def turbo_layout() -> tuple[Any, TurboSchedule, tuple[float, ...], tuple[tuple[int, int], ...]]:
    plan = canonical_timestep_plan("fl2va_turbo")
    (schedule,) = plan.schedules
    timesteps, block_keys = plan.table_layout()
    return (
        plan,
        TurboSchedule(schedule.video_timesteps, schedule.audio_timesteps),
        timesteps,
        block_keys,
    )


def tiny_overlay(config: Mapping[str, Any]) -> Any:
    _, schedule, timesteps, block_keys = turbo_layout()
    return TurboOverlay.from_official_config(
        config,
        rank=TURBO_RANK,
        alpha=TURBO_ALPHA,
        schedule=schedule,
        table_timesteps=timesteps,
        table_block_keys=block_keys,
        block_table_dtype=torch.float32,
        final_table_dtype=torch.float32,
    ).eval()


def tiny_pruned_dit(config: Mapping[str, Any], task: str = "fl2va") -> Any:
    timesteps, block_keys = canonical_timestep_plan(cast(Any, task)).table_layout()
    return AdaLNPrunedMiniMaxH3Transformer.from_official_config(
        config, table_timesteps=timesteps, table_block_keys=block_keys
    ).eval()


def fill_tables(
    source: Any, tables: Any, timesteps: Sequence[float], keys: Sequence[tuple[int, int]]
) -> None:
    """Table rows from a dynamic (possibly adapter-wrapped) modulation path, the way the
    producer computes them: `adaln_proj(temb)` and `norm_out.linear(silu(temb))` at the
    plan's exact timesteps, gathered in the plan's table order."""
    with torch.no_grad():
        temb = source.time_embedder(source.time_proj(torch.tensor(timesteps, dtype=torch.float32)))
        sparse = torch.tensor([row * 3 + tag for row, tag in keys])
        for source_block, table_block in zip(
            source.transformer_blocks, tables.transformer_blocks, strict=True
        ):
            dense = torch.stack(source_block.adaln_proj(temb), dim=1)
            table_block.adaln_proj.table.copy_(dense.index_select(0, sparse))
        final = source.norm_out.linear(torch.nn.functional.silu(temb))
        tables.norm_out.table.copy_(final.reshape(len(timesteps), 2, -1))


def turbo_forward(step: int, schedule: TurboSchedule, seed: int = 11) -> dict[str, Any]:
    """One packed sequence at turbo evaluation `step`: text and a target video row at the
    video timestep, a clean condition video row, and a target audio row at the audio one."""
    clean = _as_float32(0.999)
    distinct = sorted({schedule.video[step], schedule.audio[step], clean})
    torch.manual_seed(seed)
    return {
        "hidden_states": torch.randn(1, 2, 2),
        "audio_hidden_states": torch.randn(1, 1, 2),
        "encoder_hidden_states": torch.randn(1, 1, 8),
        "timestep": torch.tensor(distinct, dtype=torch.float32),
        "timestep_indices": torch.tensor(
            [
                distinct.index(schedule.video[step]),
                distinct.index(schedule.video[step]),
                distinct.index(clean),
                distinct.index(schedule.audio[step]),
            ]
        ),
        "token_tags": torch.tensor([1, 0, 0, 2]),
        "position_ids": torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        "video_indices": torch.tensor([1, 2]),
        "audio_indices": torch.tensor([3]),
        "text_indices": torch.tensor([0]),
        "return_dict": False,
    }


def arm_turbo_plan() -> None:
    print("\n== PDD-8: the turbo plan is the reference grid read at its block boundaries ==")
    grids = {"video": 12.0, "audio": 3.0}
    steps = PDD_HEADER["pdd_num_steps"]
    block = PDD_HEADER["pdd_block_size"]
    check("eight evaluations", steps // block, 8)
    check("turbo_steps() reads both plans", official.turbo_steps(), 8)
    for task in ("fl2va_turbo", "ref2va_turbo"):
        plan = canonical_timestep_plan(cast(Any, task))
        check(f"{task} plan is stamped with its trunk", plan.task, task.removesuffix("_turbo"))
        check(f"{task} canonical plan digest", plan.digest, TURBO_PLAN_DIGESTS[task])
        committed = files("minimax_h3").joinpath("timestep-plans", f"{task}.json").read_bytes()
        check(f"{task} committed semantic identity", timestep_plan_digest(committed), plan.digest)
        check(f"{task} one fixed schedule", plan.steps, (8,))
        (schedule,) = plan.schedules
        check(f"{task} nfe + 1 grid points", schedule.sigma_grid_points, 9)
        for modality, shift in grids.items():
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(9)
            official_sigmas = scheduler.sigmas.float()
            check(
                f"{task} {modality} sigmas are the scheduler's own at nine points",
                torch.equal(official_sigmas, torch.tensor(getattr(schedule, f"{modality}_sigmas"))),
                True,
            )
            reference = 1.0 - pdd_time_grid(shift, steps)[::block]
            gap = (reference.float() - official_sigmas).abs().max().item()
            check(
                f"{task} {modality} grid equals the reference's block boundaries to float32",
                gap <= 1.5e-7,
                True,
            )
            observe(f"{task} {modality} float64-vs-float32 boundary gap", f"{gap:.3e}")
        document = json.loads(plan.canonical_bytes())
        check(
            f"{task} turbo table rows",
            (
                len(document["table_keys"]["block_modulation"]),
                len(document["table_keys"]["final_normalization"]),
            ),
            (TURBO_BLOCK_ROWS, TURBO_FINAL_ROWS),
        )
    red(
        "the turbo plans are not the base plans",
        set(TURBO_PLAN_DIGESTS.values()) & set(PLAN_DIGESTS.values()),
        {"x"},
    )
    _, turbo, _, _ = turbo_layout()
    check("the two grids share only the origin", set(turbo.video) & set(turbo.audio), {0.0})
    for modality, shift in grids.items():
        plan = pdd_head_plan(shift, steps, block)
        step_sizes = pdd_time_grid(shift, steps).diff()
        check(
            f"{modality} head plan rows are the reference's per-block plans",
            all(
                torch.equal(plan[index], reference_pdd_plan(step_sizes, index * block, block)[0])
                for index in range(steps // block)
            ),
            True,
        )
        check(
            f"{modality} head plan rows sum to one",
            torch.allclose(plan.sum(1), torch.ones(8, dtype=torch.float64)),
            True,
        )
    refusal(
        "a grid the block does not divide refuses", lambda: pdd_head_plan(12.0, 30, 4), "ValueError"
    )
    refusal(
        "grids meeting away from the origin refuse",
        lambda: TurboSchedule((0.0, 0.5, 0.7), (0.0, 0.5, 0.9)),
        "ValueError",
    )
    refusal(
        "timesteps naming no evaluation refuse",
        lambda: turbo.step(torch.tensor([0.123])),
        "artifact_config",
    )
    check("the origin names evaluation zero", turbo.step(torch.tensor([0.0, 0.999])), 0)
    check(
        "every evaluation is named by its own pair",
        [turbo.step(torch.tensor([turbo.video[k], turbo.audio[k]])) for k in range(8)],
        list(range(8)),
    )


def arm_turbo_heads() -> None:
    print("\n== PDD-8: the 32-head bank collapses to one head per evaluation ==")
    torch.manual_seed(5)
    bank = torch.randn(32, 5, 7).bfloat16().float()
    bias = torch.randn(32, 5).bfloat16().float()
    plan = pdd_head_plan(12.0, 32, 4)
    weight, fused_bias = collapse_head_bank(bank, bias, plan)
    check("collapsed shapes", (tuple(weight.shape), tuple(fused_bias.shape)), ((8, 5, 7), (8, 5)))
    heads = TurboHeads(8, 5, 7)
    with torch.no_grad():
        heads.weight.copy_(weight)
        heads.bias.copy_(fused_bias)
    x = torch.randn(3, 7)
    step_sizes = pdd_time_grid(12.0, 32).diff()
    agree = []
    for step in range(8):
        reference = torch.nn.Linear(7, 5)
        head = ReferenceParallelHead(reference, 32)
        with torch.no_grad():
            head.weight.copy_(bank)
            head.bias.copy_(bias)
        head.plan = reference_pdd_plan(step_sizes, step * 4, 4).float()
        agree.append(torch.allclose(head(x), heads(x, step), rtol=1e-6, atol=1e-6))
    check("collapsed heads equal the reference's per-step einsum form", agree, [True] * 8)
    red(
        "a neighbouring evaluation's head differs",
        torch.allclose(heads(x, 1), heads(x, 2), rtol=1e-6, atol=1e-6),
        True,
    )
    refusal(
        "a bank the plan does not describe refuses",
        lambda: collapse_head_bank(bank[:31], bias[:31], plan),
        "ValueError",
    )
    # Odd sequence lengths put partition boundaries inside projection tiles. Small
    # CPU GEMMs may happen to agree without tiling, so also observe their real
    # reduction shapes; accelerator evidence covers the differing numeric kernels.
    x = torch.randn(1, 7, 3079).transpose(1, 2)
    with _GemmRows() as operations:
        expected = heads(x, 7)
        for degree in (2, 4):
            actual = torch.cat(
                [heads(part.clone(), 7) for part in x.tensor_split(degree, dim=1)], dim=1
            )
            check(
                f"ragged degree {degree} heads preserve FP32 values",
                torch.allclose(actual, expected, rtol=1e-6, atol=1e-6),
                True,
            )
    check("head partitions use one GEMM row shape", set(operations.rows), {1024})
    check("head tiles use row-major operands", set(operations.strides), {(7, 1)})
    with _GemmRows() as untiled:
        for degree in (1, 2, 4):
            for part in x.tensor_split(degree, dim=1):
                F.linear(part, heads.weight[7], heads.bias[7])
    red("whole-shard projection changes GEMM row shapes", len(set(untiled.rows)), 1)


class _GemmRows(TorchDispatchMode):  # type: ignore[misc]
    """Observe real projection shapes without substituting any numeric operation."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[int] = []
        self.strides: list[tuple[int, int]] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        # addmm_ too: from Runtime 0.19 the LoRA up projection adds into its output in place
        fused = (torch.ops.aten.addmm.default, torch.ops.aten.addmm_.default)
        if func in (*fused, torch.ops.aten.mm.default):
            operand = args[1] if func in fused else args[0]
            self.rows.append(int(operand.shape[0]))
            self.strides.append((int(operand.stride(0)), int(operand.stride(1))))
        return func(*args, **(kwargs or {}))


class _Allocations(TorchDispatchMode):  # type: ignore[misc]
    """Every tensor an operator allocates: new storage, not a view of one seen before."""

    def __init__(self, *known: Any) -> None:
        super().__init__()
        self.known = {t.untyped_storage().data_ptr() for t in known}
        self.sizes: list[int] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        for value in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(value, torch.Tensor) and value.numel():
                pointer = value.untyped_storage().data_ptr()
                if pointer not in self.known:
                    self.known.add(pointer)
                    self.sizes.append(value.numel())
        return out


def arm_turbo_lora() -> None:
    print("\n== PDD-8: the low-rank update accumulates in place ==")
    rows, width, out_features, rank = 20_000, 64, 48, 8
    # The Runtime's fixed GEMM row count: 256 through 0.18.x, 4096 from 0.19 (TensorD #26:
    # 256-row chunks cost 7.6% of a dense step on an RTX PRO 6000).
    chunk = turbo_module._LORA_ROW_CHUNK
    torch.manual_seed(3)
    factors = LoRAFactors(width, out_features, rank, 1.0)
    with torch.no_grad():
        factors.lora_down.copy_(torch.randn(rank, width))
        factors.lora_up.copy_(torch.randn(out_features, rank) * 0.1)
    x = torch.randn(1, rows, width).bfloat16()
    base = torch.randn(1, rows, out_features).bfloat16()
    down, up = factors.lora_down, factors.lora_up
    # From 0.19 the up projection adds into the base output (addmm: one rounding, of the sum);
    # through 0.18 the update was rounded to BF16 first, then the sum. Either stays within half
    # a BF16 step (2**-8 relative) of the update and of the sum, of the FP32 result.
    low = F.linear(x, down.bfloat16())
    update = low.float() @ up.float().t()
    exact = base.float() + update
    out = base.clone()
    with _Allocations(x, out, factors.lora_down, factors.lora_up) as allocations:
        factors.accumulate(x, out)
    bound = 2**-8 * (update.abs() + exact.abs()) + 2**-12
    check(
        "bf16 operand: the update is within BF16 rounding of the FP32 sum",
        bool(((out.float() - exact).abs() <= bound).all()),
        True,
    )
    red(
        "an update dropped or doubled is far outside that rounding",
        bool(((base.float() - exact).abs() <= bound).all()),
        True,
    )
    check(
        "bf16 operand: every transient is bounded to one chunk of rows",
        max(allocations.sizes) <= chunk * max(width, out_features),
        True,
    )
    red(
        "a materialised [rows, out] update would be larger",
        rows * out_features <= chunk * max(width, out_features),
        True,
    )

    refusal(
        "a prequantized operand cannot silently change the LoRA computation",
        lambda: factors.accumulate(SimpleNamespace(payload=x, scale=1.0), base.clone()),
        "ValueError",
    )
    refusal(
        "an output that is not a plain row-major buffer refuses rather than copying",
        lambda: factors.accumulate(x, base.transpose(1, 2).contiguous().transpose(1, 2)),
        "ValueError",
    )
    transposed = x.transpose(1, 2).contiguous().transpose(1, 2)
    canonical = base.clone()
    factors.accumulate(transposed.contiguous(), canonical)
    out = base.clone()
    with (
        _GemmRows() as operations,
        _Allocations(transposed, out, down, up) as allocations,
    ):
        factors.accumulate(transposed, out)
    check("LoRA input layout does not change values", torch.equal(out, canonical), True)
    check("LoRA tiles use row-major operands", set(operations.strides), {(width, 1), (rank, 1)})
    check(
        "canonicalizing LoRA input layout still bounds every transient to one chunk of rows",
        max(allocations.sizes) <= chunk * max(width, out_features),
        True,
    )


def arm_turbo_forward() -> None:
    print("\n== PDD-8: a turbo forward equals the reference adapter over the official DiT ==")
    torch.manual_seed(7)
    full = MiniMaxH3Transformer3DModel(**TURBO_CONFIG).eval()
    reference = copy.deepcopy(full)
    targets = str(PDD_HEADER["lora_targets"]).split(",")
    sites = reference_add_lora(reference, targets, TURBO_RANK, TURBO_ALPHA)
    check("the reference wraps every family, the tabled one included", sites, 3 * 6 + 2)
    reference.proj_out = ReferenceParallelHead(reference.proj_out, 32)
    reference.audio_proj_out = ReferenceParallelHead(reference.audio_proj_out, 32)
    torch.manual_seed(9)
    with torch.no_grad():
        for module in reference.modules():
            if isinstance(module, ReferenceLoRALinear):
                module.lora_down.copy_(torch.randn_like(module.lora_down).bfloat16().float())
                module.lora_up.copy_((torch.randn_like(module.lora_up) * 0.3).bfloat16().float())
        for head in (reference.proj_out, reference.audio_proj_out):
            head.weight.add_(torch.randn_like(head.weight) * 0.2)
            head.bias.add_(torch.randn_like(head.bias) * 0.2)
            head.weight.copy_(head.weight.bfloat16().float())
            head.bias.copy_(head.bias.bfloat16().float())

    base_plan = canonical_timestep_plan("fl2va")
    base_timesteps, base_keys = base_plan.table_layout()
    pruned = tiny_pruned_dit(TURBO_CONFIG)
    shared = {
        name: value
        for name, value in full.state_dict().items()
        if name in pruned.state_dict() and pruned.state_dict()[name].shape == value.shape
    }
    check(
        "the pruned DiT keeps the official heads by name",
        pruned.load_state_dict(shared, strict=False).unexpected_keys,
        [],
    )
    check(
        "pruned destinations are the base's: no turbo key enters the DiT",
        [name for name in pruned.state_dict() if "lora" in name or "turbo" in name],
        [],
    )
    fill_tables(full, pruned, base_timesteps, base_keys)

    _, schedule, turbo_timesteps, turbo_keys = turbo_layout()
    overlay = tiny_overlay(TURBO_CONFIG)
    with torch.no_grad():
        for path, factors in overlay.lora_sites():
            site = reference.get_submodule(path)
            factors.lora_down.copy_(site.lora_down)
            factors.lora_up.copy_(site.lora_up)
        fill_tables(reference, overlay, turbo_timesteps, turbo_keys)
        for name, shift in (("proj_out", 12.0), ("audio_proj_out", 3.0)):
            bank = getattr(reference, name)
            weight, bias = collapse_head_bank(bank.weight, bank.bias, pdd_head_plan(shift, 32, 4))
            getattr(overlay, name).weight.copy_(weight)
            getattr(overlay, name).bias.copy_(bias)
    paths = [path for path, _ in overlay.lora_sites()]
    families = sorted({next(f for f in LORA_FAMILIES if path.endswith(f)) for path in paths})
    check(
        "the overlay's LoRA sites are exactly the six inference families on every block",
        (len(paths), sorted(set(families)), any("adaln_proj" in path for path in paths)),
        (3 * 6, sorted(LORA_FAMILIES), False),
    )
    check(
        "every site resolves to one official linear of the DiT",
        all(isinstance(pruned.get_submodule(path), torch.nn.Linear) for path in paths),
        True,
    )
    check(
        "overlay state-dict keys are the adapter file's, minus the tabled slice, plus the tables",
        {
            name.replace("transformer_blocks.1.", "transformer_blocks.0.")
            for name in overlay.state_dict()
        },
        {
            *(
                f"transformer_blocks.0.{family}.{end}"
                for family in [f"attn.{f}" for f in LORA_FAMILIES[:4]] + list(LORA_FAMILIES[4:])
                for end in ("lora_down", "lora_up")
            ),
            *(
                f"token_refiner.refiner_blocks.0.{family}.{end}"
                for family in [f"attn.{f}" for f in LORA_FAMILIES[:4]] + list(LORA_FAMILIES[4:])
                for end in ("lora_down", "lora_up")
            ),
            "transformer_blocks.0.adaln_proj.table",
            "norm_out.table",
            "proj_out.weight",
            "proj_out.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
        },
    )
    pruned.install_lora_consumers()

    base_forward = turbo_forward(0, schedule)
    before = pruned(**base_forward)
    check(
        "with the overlay attached, a forward naming no bank is the base's",
        all(
            torch.allclose(a, b, rtol=2e-5, atol=2e-6)
            for a, b in zip(before, full(**base_forward), strict=True)
        ),
        True,
    )
    video_steps = pdd_time_grid(12.0, 32).diff()
    audio_steps = pdd_time_grid(3.0, 32).diff()
    agree, hooked = [], []
    for step in range(8):
        forward = turbo_forward(step, schedule)
        reference.proj_out.plan = reference_pdd_plan(video_steps, step * 4, 4).float()
        reference.audio_proj_out.plan = reference_pdd_plan(audio_steps, step * 4, 4).float()
        with torch.no_grad():
            want = reference(**forward)
            seen: list[int] = []
            handle = pruned.transformer_blocks[0].attn.to_q.register_forward_pre_hook(
                lambda module, args, seen=seen: seen.append(
                    sum(isinstance(hook, _LoRAHook) for hook in module._forward_hooks.values())
                )
            )
            got = pruned(
                **forward, attention_kwargs={ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: overlay}
            )
            handle.remove()
        hooked.append(seen)
        agree.append(
            all(torch.allclose(a, b, rtol=2e-5, atol=2e-6) for a, b in zip(got, want, strict=True))
        )
    check(
        "all eight evaluations equal the reference (video and audio velocities)", agree, [True] * 8
    )
    check("one LoRA hook stood on a site during each turbo forward", hooked, [[1]] * 8)
    check(
        "one permanent hook per site remains installed",
        sum(
            isinstance(hook, _LoRAHook)
            for module in pruned.modules()
            for hook in module._forward_hooks.values()
        ),
        len(paths),
    )
    check(
        "after eight turbo forwards a base forward is bit-identical to before",
        all(torch.equal(a, b) for a, b in zip(pruned(**base_forward), before, strict=True)),
        True,
    )

    # Red: the tabled slice. Turbo tables from the UNADAPTED modulation path lack the
    # adaln_proj LoRA the producer bakes in, and the reference notices.
    stale = tiny_overlay(TURBO_CONFIG)
    stale.load_state_dict(overlay.state_dict())
    fill_tables(full, stale, turbo_timesteps, turbo_keys)
    with torch.no_grad():
        for block in stale.transformer_blocks:
            block.adaln_proj.table.zero_()
        stale.norm_out.table.zero_()
    forward = turbo_forward(3, schedule)
    reference.proj_out.plan = reference_pdd_plan(video_steps, 12, 4).float()
    reference.audio_proj_out.plan = reference_pdd_plan(audio_steps, 12, 4).float()
    with torch.no_grad():
        want = reference(**forward)
        got = pruned(
            **forward, attention_kwargs={ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: stale}
        )
    red(
        "tables without the adaln_proj slice do not reproduce the reference",
        all(torch.allclose(a, b, rtol=2e-5, atol=2e-6) for a, b in zip(got, want, strict=True)),
        True,
    )

    refusal(
        "a turbo forward at a base-only timestep refuses",
        lambda: pruned(
            **{**base_forward, "timestep": torch.tensor([0.0, 0.5])},
            attention_kwargs={ATTENTION_KWARG: TURBO_BANK},
        ),
        "artifact_config",
    )
    check(
        "a refused turbo forward leaves the DiT disarmed for the next base forward",
        all(torch.equal(a, b) for a, b in zip(pruned(**base_forward), before, strict=True)),
        True,
    )
    refusal(
        "an unknown bank refuses",
        lambda: pruned(**base_forward, attention_kwargs={ATTENTION_KWARG: "draft"}),
        "artifact_config",
    )
    # Every turbo boundary sits on the base's 40-step grid (1/8 = 5/40), so the base tables
    # DO hold the turbo timesteps: a forward that names no bank at one is the base's own
    # 40-step evaluation, which is why the bank is named per call and never inferred.
    check(
        "the turbo timesteps are base 40-step rows",
        set(schedule.video) <= set(canonical_timestep_plan("fl2va").schedule(40).video_timesteps),
        True,
    )
    with torch.no_grad():
        forward = turbo_forward(3, schedule)
        check(
            "without the bank, a forward at a turbo timestep is the official base forward",
            all(
                torch.allclose(a, b, rtol=2e-5, atol=2e-6)
                for a, b in zip(pruned(**forward), full(**forward), strict=True)
            ),
            True,
        )


def turbo_h3_config() -> dict[str, Any]:
    """The producer's exact AdaLN-pruned config plus the two overlay sections."""
    sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
    assets = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets"
    sections = parse_production_config((assets / "model-config.json").read_bytes())
    plans = {
        task: parse_plan((assets / f"timestep-plan.{task}.json").read_bytes(), task=task)
        for task in ("fl2va", "ref2va")
    }
    document = canonical_json.decode(
        dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"])
    )
    for trunk in ("fl2va", "ref2va"):
        overlay = copy.deepcopy(document[f"{trunk}_dit"])
        overlay["cozy_h3"] = {
            "task": trunk,
            "modulation": "adaln-pruned",
            "distillation": "pdd",
            "table_keys": canonical_json.decode(
                canonical_timestep_plan(cast(Any, f"{trunk}_turbo")).canonical_bytes()
            )["table_keys"],
            "lora_rank": PDD_HEADER["lora_rank"],
            "lora_alpha": PDD_HEADER["lora_alpha"],
            "pdd_num_steps": PDD_HEADER["pdd_num_steps"],
            "pdd_block_size": PDD_HEADER["pdd_block_size"],
        }
        document[f"{trunk}_turbo"] = overlay
    return cast(dict[str, Any], document)


def arm_turbo_state() -> None:
    """The public two-model sample passes its overlay through the real Diffusers state."""
    torch.manual_seed(512)
    dit = tiny_pruned_dit(TURBO_CONFIG)
    overlay = tiny_overlay(TURBO_CONFIG)
    with torch.no_grad():
        for module in (dit, overlay):
            for parameter in module.parameters():
                parameter.normal_(0, 0.02)
    dit.install_lora_consumers()
    dit.set_attention_backend("native")
    forward = turbo_forward(0, overlay.schedule)
    original = {ATTENTION_KWARG: TURBO_BANK}
    state = PipelineState()
    state.set("attention_kwargs", original)
    state.set("latents", forward["hidden_states"][0])
    state.set("audio_latents", forward["audio_hidden_states"][0])
    state.set("prompt_embeds", forward["encoder_hidden_states"])
    state.set("row_timestep_plan", [(forward["timestep"], forward["timestep_indices"])])
    for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
        state.set(name, forward[name], kwargs_type="denoiser_input_fields")
    loop = MiniMaxH3LoopDenoiser()
    observed: list[Any] = []

    def denoise(task: Any, state: Any, *, on_step: Any, cancel: Any, sol_dense_steps: int) -> Any:
        cancel()
        block_state = loop.get_block_state(state)
        observed.append((block_state.attention_kwargs.get(OVERLAY_KWARG), sol_dense_steps))
        _, block_state = loop(SimpleNamespace(transformer=dit), block_state, 0, forward["timestep"])
        on_step(0)
        return block_state.noise_pred, block_state.audio_noise_pred

    base = package.H3TurboBase.for_test(
        pipe=SimpleNamespace(
            components={"fl2va_dit": dit},
            _dit_specs={"fl2va": (TURBO_CONFIG, "adaln-pruned", None)},
            denoise=denoise,
        )
    )
    lora_pipe = object.__new__(official.OfficialH3TurboLoRA)
    lora_pipe.components = {"fl2va_turbo": overlay}
    layout = TableLayout.parse(
        json.loads(canonical_timestep_plan("fl2va_turbo").canonical_bytes())["table_keys"]
    )
    lora_pipe.specs = {"fl2va": (TURBO_CONFIG, layout)}
    lora = package.H3TurboLoRA.for_test(pipe=lora_pipe)
    with torch.no_grad():
        expected = dit(
            **forward, attention_kwargs={ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: overlay}
        )
        actual = cast(
            Any,
            base.sample_fl2va_turbo(
                state,
                turbo_lora=lora,
                sol_dense_steps=4,
                on_step=lambda _step: None,
                cancel=lambda: None,
            ),
        )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
    check(
        "Diffusers denoiser received the separate overlay",
        len(observed) == 1 and observed[0][0] is overlay,
        True,
    )
    check("sample forwards explicit Sol policy through both models", observed[0][1], 4)
    check(
        "successful sample restores the original selector",
        state.get("attention_kwargs") is original,
        True,
    )
    check("sample creates no shadow state attribute", "attention_kwargs" in vars(state), False)

    def cancel() -> None:
        raise Cancelled("stop before the first forward")

    try:
        base.sample_fl2va_turbo(
            state,
            turbo_lora=lora,
            on_step=lambda _step: None,
            cancel=cancel,
        )
    except Cancelled:
        pass
    else:
        fail("turbo selector cancellation", "cancellation was swallowed")
    check(
        "canceled sample restores the original selector",
        state.get("attention_kwargs") is original,
        True,
    )
    check("DiT remains disarmed after both calls", dit._arming, None)


def arm_turbo_artifact() -> None:
    """Separate exact base/turbo constructors and independent Runtime census scopes."""
    document = turbo_h3_config()
    base_only = {name: value for name, value in document.items() if not name.endswith("_turbo")}
    overlay_only = {name: value for name, value in document.items() if name.endswith("_turbo")}
    refusal(
        "ordinary construction rejects turbo roots",
        lambda: OfficialH3Pipeline(Config(document)),
        "artifact_config",
    )
    refusal(
        "LoRA construction requires both overlay roots",
        lambda: official.OfficialH3TurboLoRA(Config({"fl2va_turbo": overlay_only["fl2va_turbo"]})),
        "artifact_config",
    )
    for field, value in (
        ("lora_rank", 32),
        ("lora_alpha", 32.0),
        ("pdd_num_steps", 16),
        ("pdd_block_size", 2),
        ("task", "ref2va"),
    ):
        changed = copy.deepcopy(overlay_only)
        changed["fl2va_turbo"]["cozy_h3"][field] = value
        refusal(
            f"unsupported PDD {field} refuses",
            partial(official._overlay_spec, changed, "fl2va"),
            "artifact_config",
        )
    with torch.device("meta"):
        base = official.build_h3_turbo_base(Config(base_only))
        lora = official.OfficialH3TurboLoRA(Config(overlay_only))
    check("ordinary model has exactly five real roots", set(base.components), set(base_only))
    check("turbo LoRA has exactly two overlay roots", set(lora.components), set(overlay_only))
    for cls, config in ((package.H3TurboBase, base_only), (package.H3TurboLoRA, overlay_only)):
        derived = derive(cls(), Artifact("constructor-audit", {}, Config(config)))
        check(
            f"{cls.__name__} Runtime census matches its exact checkpoint",
            {row.component for row in derived.destination_sets},
            set(config),
        )
    for trunk in ("fl2va", "ref2va"):
        dit = base.components[f"{trunk}_dit"]
        overlay = lora.components[f"{trunk}_turbo"]
        check(
            f"{trunk} base installs permanent LoRA consumers",
            len(dit.transformer_blocks[0].attn.to_q._forward_hooks),
            1,
        )
        check(f"{trunk} overlay has independent weights", id(dit) != id(overlay), True)
        check(f"{trunk} overlay head geometry", tuple(overlay.proj_out.weight.shape), (8, 96, 5376))


def arm_attention_scope() -> None:
    """Real Diffusers layouts and CPU DiT steps supply/reset request-local attention facts."""
    refs = [
        MiniMaxH3ImageReference(image=PILImage.new("RGB", (16, 16))),
        MiniMaxH3AudioReference(audio=torch.zeros(2, 8), sample_rate=32000),
        MiniMaxH3VideoReference(
            frames=torch.zeros(1, 3, 16, 16),
            fps=24,
            audio=torch.zeros(2, 8),
            sample_rate=32000,
        ),
    ]
    packed = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        text_token_tags=torch.ones(3, dtype=torch.long),
        references=refs,
        condition_latents=[torch.zeros(1, 24, 1, 4, 6), torch.zeros(1, 24, 2, 4, 6)],
        audio_condition_latents=[torch.zeros(6, 32), torch.zeros(4, 32)],
        num_latent_frames=2,
        latent_height=4,
        latent_width=6,
        num_audio_latents=4,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
    )
    _, tags, video, audio, text, condition_video, condition_audio = packed
    fields = {"token_tags": tags, "text_indices": text, "audio_indices": audio}
    state = SimpleNamespace(
        denoiser_input_fields=fields,
        num_condition_video_rows=condition_video,
        num_condition_audio_rows=condition_audio,
    )
    layout = _attention_layout(state, 12)
    protected = torch.cat((text, video[:condition_video], audio)).sort().values
    check(
        "all non-target-video rows occupy the protected prefix", protected.tolist(), list(range(39))
    )
    check("mixed reference layout facts", (layout.live_tokens, layout.protected_prefix), (51, 39))
    state.denoiser_input_fields = {name: value.to("meta") for name, value in fields.items()}
    check(
        "layout uses shape metadata without tensor-value reads",
        _attention_layout(state, 12),
        layout,
    )

    pipe = meta_h3_pipeline()
    dit = MiniMaxH3Transformer3DModel.from_config(
        dict(tiny_dit().config),
        in_channels=24,
        audio_in_channels=32,
    ).eval()
    pipe.components["fl2va_dit"] = dit
    for workflow in ("t2va", "fl2va"):
        pipe._pipes[workflow].update_components(transformer=dit)

    def start() -> Any:
        state = pipe.start_fl2va(
            prompt="Two fighters.",
            first_frame=None,
            last_frame=None,
            generator=torch.Generator().manual_seed(7),
            steps=30,
            frames=124,
        )
        for name, value in {
            "height": 64,
            "width": 96,
            "prompt_embeds": torch.zeros(1, 4, 32),
            "text_token_tags": torch.ones(4, dtype=torch.long),
        }.items():
            state.set(name, value)
        return state

    seen: list[AttentionLayout | None] = []
    between: list[AttentionLayout | None] = []
    handle = dit.register_forward_pre_hook(lambda _module, _args: seen.append(_ACTIVE_LAYOUT.get()))
    outer = AttentionLayout(live_tokens=1, protected_prefix=0, step=99)
    with torch.no_grad(), attention_scope(outer):
        pipe.denoise(
            "fl2va",
            start(),
            on_step=lambda _: between.append(_ACTIVE_LAYOUT.get()),
            cancel=lambda: None,
        )
        check("each DiT step restores the surrounding scope", between, [outer] * 30)
        check("completed request restores the surrounding scope", _ACTIVE_LAYOUT.get(), outer)
    handle.remove()
    check(
        "30 actual CPU DiT forwards have ordered scope indices",
        [x.step for x in seen if x],
        list(range(30)),
    )
    check(
        "actual text-only forward layout",
        [(x.live_tokens, x.protected_prefix) for x in seen if x],
        [(640, 418)] * 30,
    )
    check("request scope does not leak", _ACTIVE_LAYOUT.get(), None)

    class StopForward(Exception):
        pass

    def stop(_module: Any, _args: Any) -> None:
        assert _ACTIVE_LAYOUT.get() is not None
        raise StopForward("scoped CPU forward interrupted")

    handle = dit.register_forward_pre_hook(stop)
    try:
        with attention_scope(outer):
            refusal(
                "failed DiT step unwinds its scope",
                lambda: pipe.denoise(
                    "fl2va",
                    start(),
                    on_step=lambda _: None,
                    cancel=lambda: None,
                ),
                "StopForward",
            )
            check("failed step restores outer scope", _ACTIVE_LAYOUT.get(), outer)
    finally:
        handle.remove()
    check("failure scope does not leak", _ACTIVE_LAYOUT.get(), None)
    handle = dit.register_forward_pre_hook(lambda _module, _args: seen.append(_ACTIVE_LAYOUT.get()))
    try:
        pipe.warm_dit("fl2va")
    finally:
        handle.remove()
    check(
        "warm is explicitly dense",
        seen[-1],
        AttentionLayout(
            live_tokens=24,
            protected_prefix=0,
            step=0,
            dense_until_step=10,
            dense_paths=("token_refiner", "transformer_blocks.0", "transformer_blocks.1"),
            exact_paths=("token_refiner", "transformer_blocks.1", "transformer_blocks.45", "transformer_blocks.49"),
        ),
    )

def _code(call: Callable[[], object]) -> str:
    try:
        call()
    except (glue.GlueRefusal, ValueError) as exc:
        return str(getattr(exc, "code", type(exc).__name__))
    return ""


def arm_glue() -> None:
    """The fused glue's Diffusers contract and topology match on a real (narrow) H3 DiT; a
    device without the kernels serves eager, reported, with nothing substituted."""
    torch.manual_seed(76182)
    dit = (
        MiniMaxH3Transformer3DModel(
            num_layers=2,
            num_refiner_layers=1,
            hidden_size=256,
            ffn_dim=512,
            num_attention_heads=2,
            attention_head_dim=128,
            time_embed_hidden_dim=256,
            time_embed_dim=128,
        )
        .eval()
        .bfloat16()
    )
    glue._require_calls()
    before = glue._tensor_census(dit)
    glue._validate_transformer(dit, "dit")
    check(
        "glue matches the Diffusers H3 blocks, tensors untouched", glue._tensor_census(dit), before
    )
    check(
        "a narrow DiT is not the compiled widths",
        _code(lambda: glue._widths(dit, "dit")),
        "fusion_block_shape",
    )
    handle = dit.transformer_blocks[1].ff.register_forward_hook(lambda *_: None)
    check(
        "a hooked later block refuses before any change",
        _code(lambda: glue._validate_transformer(dit, "dit")),
        "fusion_block_shape",
    )
    handle.remove()
    model = SimpleNamespace()

    def never(kernels: object) -> Any:
        raise AssertionError("a CPU plan loaded kernels")

    served = glue.install(model, SimpleNamespace(components={"dit": dit}), Plan("cpu", never))
    check(
        "a CPU executor serves eager, reported, unsubstituted",
        (served, model._cozy_fusion_report["code"], type(dit.transformer_blocks[0]).__name__),
        ("", "fusion_device_incapable", "MiniMaxH3TransformerBlock"),
    )
    check("the plan's identity is stable", glue.identity(), glue.identity())


def arm_producer() -> None:
    """The shared-QKV producers: registrations, carrier views, rotary presentation, step policy,
    adjacent-step statistics, request-state cleanup and the processors they refuse."""
    check(
        "producers ride the Runtime's Kitchen Sol bases; only fixed means shards",
        {name: (p.base, p.context_parallel) for name, p in producer.PRODUCERS.items()},
        {
            producer.FIXED_MEANS: ("kitchen-sol-fp4-fixed-means", True),
            producer.FP4_LOWMEM: ("kitchen-sol-fp4-lowmem", False),
            producer.FAST: ("kitchen-sol", False),
        },
    )
    payload, scale = torch.arange(88).reshape(11, 8), torch.arange(11).reshape(11, 1)
    part = producer.rows(PreQuantized(payload, scale, (1, 11, 8)), 3, 8)
    check(
        "carrier rows are views of the payload and scales",
        (
            part.shape,
            part.payload.data_ptr() - payload.data_ptr(),
            part.scale.data_ptr() - scale.data_ptr(),
        ),
        ((1, 5, 8), 3 * 8 * payload.element_size(), 3 * scale.element_size()),
    )
    cos, sin = torch.randn(1, 17, 1, 96), torch.randn(1, 17, 1, 96)
    state = producer.RequestState()
    freqs = producer.rope_presentation((cos, sin), 17, state)
    x = torch.randn(17, 96)
    f = freqs[:, 0]
    turned = torch.cat(
        (
            f[:, :, 0, 0] * x[:, :48] + f[:, :, 0, 1] * x[:, 48:],
            f[:, :, 1, 1] * x[:, 48:] + f[:, :, 1, 0] * x[:, :48],
        ),
        dim=-1,
    )
    expected = x * cos.reshape(17, 96) + torch.cat((-x[:, 48:], x[:, :48]), dim=-1) * sin.reshape(
        17, 96
    )
    check(
        "the rotary matrix is split-half RoPE, reused within a step",
        (torch.equal(turned, expected), producer.rope_presentation((cos, sin), 17, state) is freqs),
        (True, True),
    )
    layout = AttentionLayout(100, 17, 4, 4, ("transformer_blocks.1", "token_refiner"))
    check(
        "sparse steps skip the protected paths",
        [
            producer.sparse_step(layout, p)
            for p in ("transformer_blocks.2.attn", "transformer_blocks.1.attn", "token_refiner.0")
        ],
        [True, False, False],
    )
    red(
        "a dense step is never sparse",
        producer.sparse_step(replace(layout, step=3), "transformer_blocks.2.attn"),
        True,
    )
    key, geometry, mean = ("dit", "block.2"), (129, 56, 17), torch.ones(1)
    reuse = []
    for step, shape in ((5, geometry), (6, geometry), (5, (129, 56, 18))):
        state.statistics[key] = (4, geometry, mean, mean)
        reuse.append(producer.previous_statistics(state, key, step, shape)[0] is mean)
    check(
        "statistics carry to the adjacent step of the same geometry only",
        reuse,
        [True, False, False],
    )
    state.close()
    check("closing the request drops its state", (state.statistics, state.rope), ({}, None))
    attention = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128)
    check(
        "an unencoded projection shares no quantization",
        producer.shared_qkv_refusal(attention),
        "non-rowwise-leaf",
    )
    producer.validate(attention)
    attention.to_q.register_forward_hook(lambda *_: None)
    check(
        "an unknown projection hook is refused",
        _code(lambda: producer.validate(attention)),
        "ValueError",
    )


ARMS = {
    "attention-scope": arm_attention_scope,
    "checkpoint-table-layout": arm_checkpoint_table_layout,
    "turbo-plan": arm_turbo_plan,
    "turbo-heads": arm_turbo_heads,
    "turbo-lora": arm_turbo_lora,
    "turbo-forward": arm_turbo_forward,
    "turbo-state": arm_turbo_state,
    "turbo-artifact": arm_turbo_artifact,
    "producer-configs": arm_producer_configs,
    "producer-construction-order": arm_producer_construction_order,
    "schedule": arm_schedule,
    "zero-reference": arm_zero_reference_preparation,
    "clip-length": arm_clip_length,
    "reference-resolution": arm_reference_resolution,
    "graph": arm_graph_and_dtypes,
    "conditioner": arm_text_conditioner,
    "conditioner-lifecycle": arm_conditioner_lifecycle,
    "adaln-pruned": arm_adaln_pruned,
    "processor": arm_processor,
    "media": arm_media,
    "reference-media": arm_reference_media,
    "video-stream": arm_video_stream,
    "gates": arm_output_gates,
    "vae-tiles": arm_vae_tiles,
    "rgb8-handoff": arm_rgb8_handoff,
    "interface": arm_interface,
    "glue": arm_glue,
    "attention-producer": arm_producer,
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
