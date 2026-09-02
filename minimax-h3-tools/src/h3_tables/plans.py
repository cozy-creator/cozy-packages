"""Read the canonical MiniMax-H3 timestep-plan handoff.

The plan, not a step-count label, is the numerical contract. Its exact bytes bind every
sigma, forward evaluation, row class, and table order consumed by the producer.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

from cozy_runtime.author import canonical_json

Task = Literal["fl2va", "ref2va"]

LAUNCH_PLAN_DIGESTS: dict[Task, str] = {
    "fl2va": "sha256:8da103b9b09629f9f4bcc7c3311929a83c4bc76d5ac2a49fa8ad6c08a140d99b",
    "ref2va": "sha256:f99dec0b673105a6b7cabdc57a62df9653afd943a9092eef6018aa48095a9487",
}

_TOP_LEVEL = {
    "adaln_row_index",
    "audio_shift",
    "table_keys",
    "table_order",
    "evaluations",
    "final_norm_row_index",
    "fps",
    "frames",
    "row_timestep_reduction",
    "scalar_encoding",
    "scheduler_semantics",
    "sigma_grid_points",
    "task",
    "terminal",
    "transformer_evaluations",
    "video_shift",
}


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    task: Task
    digest: str
    canonical_bytes: bytes
    timesteps: tuple[float, ...]
    block_rows: tuple[tuple[int, int], ...]
    sigma_grid_points: int
    transformer_evaluations: int


def _f32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _hex_f32(value: object, field: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field} is not an IEEE-754 hexadecimal string")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed != _f32(parsed) or parsed.hex() != value:
        raise ValueError(f"{field} is not one exact canonical float32 value")
    return parsed


def _closed(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{field} has fields {actual}, expected {sorted(keys)}")
    return cast(dict[str, Any], value)


def _canonical(document: dict[str, Any]) -> bytes:
    return canonical_json.encode(document)


def parse_plan(raw: bytes, *, task: str, launch: bool = True) -> TimestepPlan:
    """Validate one complete plan and return its exact table selectors.

    `launch=True` pins the currently approved 30-point oracle bytes. The structural
    validator itself admits a future exact step bucket, so widening the approved digest set
    does not require a second numerical implementation.
    """
    if task not in LAUNCH_PLAN_DIGESTS:
        raise ValueError(f"unknown MiniMax-H3 task {task!r}")
    try:
        decoded = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("timestep plan is not JSON") from exc
    document = _closed(decoded, _TOP_LEVEL, "TimestepPlan")
    canonical = _canonical(document)
    typed_task: Task = task
    if document["task"] != task:
        raise ValueError(f"timestep plan does not declare task {task!r}")
    fixed = {
        "scalar_encoding": "ieee754-binary32-hex",
        "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
        "row_timestep_reduction": "unique-sorted-return-inverse",
        "adaln_row_index": "timestep_index*3+modality_tag",
        "final_norm_row_index": "timestep_index",
        "table_order": "first-distinct-evaluation-class-occurrence",
        "frames": 345,
        "fps": 24,
    }
    for name, expected in fixed.items():
        if document[name] != expected:
            raise ValueError(
                f"timestep plan {name} is {document[name]!r}, expected {expected!r}"
            )

    grid_points = document["sigma_grid_points"]
    forwards = document["transformer_evaluations"]
    evaluations = document["evaluations"]
    if (
        not isinstance(grid_points, int)
        or not isinstance(forwards, int)
        or grid_points < 2
        or forwards != grid_points - 1
        or not isinstance(evaluations, list)
        or len(evaluations) != forwards
    ):
        raise ValueError(
            "the plan must have N sigma points, N-1 indexed forward evaluations"
        )
    _hex_f32(document["video_shift"], "video_shift")
    _hex_f32(document["audio_shift"], "audio_shift")
    terminal = _closed(
        document["terminal"],
        {"video_sigma", "audio_sigma", "transformer_evaluation"},
        "terminal",
    )
    if (
        _hex_f32(terminal["video_sigma"], "terminal.video_sigma") != 0.0
        or _hex_f32(terminal["audio_sigma"], "terminal.audio_sigma") != 0.0
        or terminal["transformer_evaluation"] is not False
    ):
        raise ValueError("terminal sigma zero is not a transformer evaluation")

    expected_classes = (
        ("target_video", "video", 0, "always"),
        ("text", "text", 1, "always"),
        ("target_audio", "audio", 2, "always"),
        ("condition_video", "video", 0, "if_condition_video_rows"),
        ("condition_audio", "audio", 2, "if_condition_audio_rows"),
    )
    video_sigmas: list[float] = []
    audio_sigmas: list[float] = []
    class_rows: list[dict[str, Any]] = []
    clean_video = _f32(0.999)
    for index, value in enumerate(evaluations):
        row = _closed(
            value,
            {"index", "video_sigma", "audio_sigma", "modulation_classes"},
            f"evaluations[{index}]",
        )
        if row["index"] != index:
            raise ValueError("evaluation indexes are not contiguous from zero")
        video_sigma = _hex_f32(row["video_sigma"], f"evaluations[{index}].video_sigma")
        audio_sigma = _hex_f32(row["audio_sigma"], f"evaluations[{index}].audio_sigma")
        video_sigmas.append(video_sigma)
        audio_sigmas.append(audio_sigma)
        classes = row["modulation_classes"]
        if not isinstance(classes, list) or len(classes) != len(expected_classes):
            raise ValueError(
                f"evaluation {index} does not declare the five H3 row classes"
            )
        wanted_timesteps = (
            _f32(1.0 - video_sigma),
            _f32(1.0 - video_sigma),
            _f32(1.0 - audio_sigma),
            max(_f32(1.0 - video_sigma), clean_video),
            _f32(1.0),
        )
        for position, (value, expected, timestep) in enumerate(
            zip(classes, expected_classes, wanted_timesteps, strict=True)
        ):
            item = _closed(
                value,
                {"name", "modality", "modality_tag", "presence", "timestep"},
                f"evaluations[{index}].modulation_classes[{position}]",
            )
            if (
                tuple(
                    item[name]
                    for name in ("name", "modality", "modality_tag", "presence")
                )
                != expected
            ):
                raise ValueError(
                    f"evaluation {index} row class {position} changed meaning"
                )
            if _hex_f32(item["timestep"], f"evaluation {index} timestep") != timestep:
                raise ValueError(
                    f"evaluation {index} row class {position} has the wrong timestep"
                )
            class_rows.append(item)
    for name, sigmas in (("video", video_sigmas), ("audio", audio_sigmas)):
        if sigmas[0] != 1.0 or any(a <= b or b <= 0.0 for a, b in pairwise(sigmas)):
            raise ValueError(
                f"{name} sigmas are not strictly descending from one toward zero"
            )

    keys = _closed(
        document["table_keys"],
        {"block_modulation", "final_normalization"},
        "table_keys",
    )
    block_expected: list[dict[str, Any]] = []
    final_expected: list[dict[str, Any]] = []
    seen_blocks: set[tuple[str, int]] = set()
    seen_final: set[str] = set()
    for item in class_rows:
        pair = (str(item["timestep"]), int(item["modality_tag"]))
        if pair not in seen_blocks:
            seen_blocks.add(pair)
            block_expected.append(
                {
                    "index": len(block_expected),
                    "timestep": pair[0],
                    "modality": item["modality"],
                    "modality_tag": pair[1],
                }
            )
        if pair[0] not in seen_final:
            seen_final.add(pair[0])
            final_expected.append({"index": len(final_expected), "timestep": pair[0]})
    if (
        keys["block_modulation"] != block_expected
        or keys["final_normalization"] != final_expected
    ):
        raise ValueError(
            "table keys are not the first-distinct exact evaluation coverage"
        )

    timesteps = tuple(
        _hex_f32(row["timestep"], "final_normalization.timestep")
        for row in final_expected
    )
    timestep_index = {value: index for index, value in enumerate(timesteps)}
    block_rows = tuple(
        (
            timestep_index[_hex_f32(row["timestep"], "block_modulation.timestep")],
            row["modality_tag"],
        )
        for row in block_expected
    )
    digest = canonical_json.digest(document)
    if launch and digest != LAUNCH_PLAN_DIGESTS[typed_task]:
        raise ValueError(
            f"{task} plan {digest} is structurally valid but is not the launch oracle "
            f"{LAUNCH_PLAN_DIGESTS[typed_task]}"
        )
    return TimestepPlan(
        typed_task,
        digest,
        canonical,
        timesteps,
        block_rows,
        grid_points,
        forwards,
    )


def parse_declared_plan(raw: bytes) -> TimestepPlan:
    """Parse one plan whose own closed task field selects the strict task validator."""
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("timestep plan is not JSON") from exc
    if not isinstance(value, dict) or value.get("task") not in LAUNCH_PLAN_DIGESTS:
        raise ValueError("timestep plan does not declare fl2va or ref2va")
    return parse_plan(raw, task=str(value["task"]))
