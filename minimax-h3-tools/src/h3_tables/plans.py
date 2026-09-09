"""Read the canonical MiniMax-H3 timestep-plan handoff.

The plan, not a step-count label, is the numerical contract. Its exact bytes bind every
schedule's sigmas and forward evaluations, every row class, and the one union table order
the producer emits so a single pruned checkpoint serves every listed step count.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import torch
from cozy_runtime.author import canonical_json

Task = Literal["fl2va", "ref2va"]
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")

#: The served 30/40/50-evaluation union plans, stamped into the committed model config.
LAUNCH_PLAN_DIGESTS: dict[Task, str] = {
    "fl2va": "sha256:9a48803d17d7bb5499ca8c018f60c86c9890eb91496249ac7cb199bc9e45201e",
    "ref2va": "sha256:565a164cbf0cefa58d4976cb4c84887cf9807cc7c4be62263d5e0e0c5d9bc50e",
}
#: PDD-8 (`alibaba-pai/MiniMax-H3-Acc-LoRAs`): `Schedule(9)` at the released shifts, which is
#: the 33-point training grid at its block-4 boundaries — eight transformer evaluations. The
#: pipeline is called with `num_inference_steps = 9` because the scheduler counts the terminal
#: sigma; the plan counts evaluations.
TURBO_PLAN_DIGESTS: dict[Task, str] = {
    "fl2va": "sha256:d2eb1605c1c1e01a4c5fdaaf1912ab43f33d9ce4772febf1c75b9bbfc8f7ba9d",
    "ref2va": "sha256:896a10881805e3047f6df514dd4e70fcea078837469eeb22ce67f0bd3f679cb7",
}

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
LAUNCH_GRID_POINTS = (31, 41, 51)
TURBO_GRID_POINTS = (9,)
FPS = 24

_TOP_LEVEL = {
    "adaln_row_index",
    "audio_shift",
    "table_keys",
    "table_order",
    "final_norm_row_index",
    "fps",
    "row_timestep_reduction",
    "scalar_encoding",
    "scheduler_semantics",
    "schedules",
    "task",
    "video_shift",
}
_SCHEDULE = {"transformer_evaluations", "sigma_grid_points", "evaluations", "terminal"}


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    task: Task
    digest: str
    canonical_bytes: bytes
    timesteps: tuple[float, ...]
    block_rows: tuple[tuple[int, int], ...]
    steps: tuple[int, ...]

    @property
    def table_keys(self) -> dict[str, Any]:
        """The exact ordered row meanings already validated in the producer handoff."""
        return cast(dict[str, Any], canonical_json.decode(self.canonical_bytes)["table_keys"])


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


_EXPECTED_CLASSES = (
    ("target_video", "video", 0, "always"),
    ("text", "text", 1, "always"),
    ("target_audio", "audio", 2, "always"),
    ("condition_video", "video", 0, "if_condition_video_rows"),
    ("condition_audio", "audio", 2, "if_condition_audio_rows"),
)


def _parse_schedule(value: object, field: str, class_rows: list[dict[str, Any]]) -> int:
    """Validate one schedule, append its row classes in order, return its step count."""
    schedule = _closed(value, _SCHEDULE, field)
    grid_points = schedule["sigma_grid_points"]
    forwards = schedule["transformer_evaluations"]
    evaluations = schedule["evaluations"]
    if (
        not isinstance(grid_points, int)
        or not isinstance(forwards, int)
        or forwards < 1
        or forwards + 1 > grid_points
        or not isinstance(evaluations, list)
        or len(evaluations) != forwards
    ):
        raise ValueError(f"{field} needs N indexed evaluations under at least N+1 grid points")
    terminal = _closed(
        schedule["terminal"],
        {"video_sigma", "audio_sigma", "transformer_evaluation"},
        f"{field}.terminal",
    )
    if (
        _hex_f32(terminal["video_sigma"], f"{field}.terminal.video_sigma") != 0.0
        or _hex_f32(terminal["audio_sigma"], f"{field}.terminal.audio_sigma") != 0.0
        or terminal["transformer_evaluation"] is not False
    ):
        raise ValueError("terminal sigma zero is not a transformer evaluation")
    video_sigmas: list[float] = []
    audio_sigmas: list[float] = []
    clean_video = _f32(0.999)
    for index, value in enumerate(evaluations):
        name = f"{field}.evaluations[{index}]"
        row = _closed(value, {"index", "video_sigma", "audio_sigma", "modulation_classes"}, name)
        if row["index"] != index:
            raise ValueError("evaluation indexes are not contiguous from zero")
        video_sigma = _hex_f32(row["video_sigma"], f"{name}.video_sigma")
        audio_sigma = _hex_f32(row["audio_sigma"], f"{name}.audio_sigma")
        video_sigmas.append(video_sigma)
        audio_sigmas.append(audio_sigma)
        classes = row["modulation_classes"]
        if not isinstance(classes, list) or len(classes) != len(_EXPECTED_CLASSES):
            raise ValueError(f"{name} does not declare the five H3 row classes")
        wanted_timesteps = (
            _f32(1.0 - video_sigma),
            _f32(1.0 - video_sigma),
            _f32(1.0 - audio_sigma),
            max(_f32(1.0 - video_sigma), clean_video),
            _f32(1.0),
        )
        for position, (item_value, expected, timestep) in enumerate(
            zip(classes, _EXPECTED_CLASSES, wanted_timesteps, strict=True)
        ):
            item = _closed(
                item_value,
                {"name", "modality", "modality_tag", "presence", "timestep"},
                f"{name}.modulation_classes[{position}]",
            )
            if (
                tuple(item[key] for key in ("name", "modality", "modality_tag", "presence"))
                != expected
            ):
                raise ValueError(f"{name} row class {position} changed meaning")
            if _hex_f32(item["timestep"], f"{name} timestep") != timestep:
                raise ValueError(f"{name} row class {position} has the wrong timestep")
            class_rows.append(item)
    for modality, sigmas in (("video", video_sigmas), ("audio", audio_sigmas)):
        if sigmas[0] != 1.0 or any(a <= b or b <= 0.0 for a, b in pairwise(sigmas)):
            raise ValueError(
                f"{field} {modality} sigmas are not strictly descending from one toward zero"
            )
    return forwards


def parse_plan(
    raw: bytes, *, task: str, digests: Mapping[Task, str] = LAUNCH_PLAN_DIGESTS
) -> TimestepPlan:
    """Validate one complete plan and return its exact table selectors.

    `digests` pins the approved oracle bytes per task. The structural validator itself
    admits any exact schedule set, so admitting another plan is one more pinned digest, never
    a second numerical implementation.
    """
    if task not in TASKS:
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
        "fps": FPS,
    }
    for name, expected in fixed.items():
        if document[name] != expected:
            raise ValueError(
                f"timestep plan {name} is {document[name]!r}, expected {expected!r}"
            )
    _hex_f32(document["video_shift"], "video_shift")
    _hex_f32(document["audio_shift"], "audio_shift")

    schedules = document["schedules"]
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("the plan must list at least one schedule")
    class_rows: list[dict[str, Any]] = []
    steps = tuple(
        _parse_schedule(value, f"schedules[{index}]", class_rows)
        for index, value in enumerate(schedules)
    )
    if any(left >= right for left, right in pairwise(steps)):
        raise ValueError("schedules must be listed by strictly ascending step count")

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
    if digest != digests[typed_task]:
        raise ValueError(
            f"{task} plan {digest} is structurally valid but is not the pinned oracle "
            f"{digests[typed_task]}"
        )
    return TimestepPlan(typed_task, digest, canonical, timesteps, block_rows, steps)


def parse_declared_plan(
    raw: bytes, *, digests: Mapping[Task, str] = LAUNCH_PLAN_DIGESTS
) -> TimestepPlan:
    """Parse one plan whose own closed task field selects the strict task validator."""
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("timestep plan is not JSON") from exc
    if not isinstance(value, dict) or value.get("task") not in TASKS:
        raise ValueError("timestep plan does not declare fl2va or ref2va")
    return parse_plan(raw, task=str(value["task"]), digests=digests)


def sigma_grid(shift: float, points: int) -> tuple[float, ...]:
    """`MiniMaxH3Scheduler.set_timesteps(points)`: the float32 linspace through the
    exponential shift, float32 collisions collapsed, terminal zero included."""
    base = torch.linspace(1.0, 0.0, points, dtype=torch.float32)
    sigmas = shift * base / (1 + (shift - 1) * base)
    return tuple(float(value) for value in torch.unique_consecutive(sigmas))


def _modulation_class(
    name: str, timestep: float, modality: str, tag: int, presence: str
) -> dict[str, Any]:
    return {
        "name": name,
        "modality": modality,
        "modality_tag": tag,
        "presence": presence,
        "timestep": _f32(timestep).hex(),
    }


def compose_plan(
    task: Task,
    *,
    video_shift: float = VIDEO_SHIFT,
    audio_shift: float = AUDIO_SHIFT,
    grid_points: Sequence[int],
) -> bytes:
    """The canonical plan bytes for these sigma grids, ascending by evaluation count.

    This is the producer's own derivation of `minimax-h3`'s `TimestepPlan.canonical_bytes()`
    without Diffusers: the committed assets must reproduce from it byte for byte.
    """
    clean_video = _f32(0.999)
    schedules: list[dict[str, Any]] = []
    block_keys: list[dict[str, Any]] = []
    final_keys: list[dict[str, Any]] = []
    seen_blocks: set[tuple[str, int]] = set()
    seen_final: set[str] = set()
    for points in grid_points:
        video = sigma_grid(video_shift, points)
        audio = sigma_grid(audio_shift, points)
        if len(video) != len(audio) or len(video) < 2:
            raise ValueError(f"{points} grid points collide differently per modality")
        evaluations: list[dict[str, Any]] = []
        for index, (video_sigma, audio_sigma) in enumerate(
            zip(video[:-1], audio[:-1], strict=True)
        ):
            video_t = _f32(1.0 - video_sigma)
            audio_t = _f32(1.0 - audio_sigma)
            classes = [
                _modulation_class("target_video", video_t, "video", 0, "always"),
                _modulation_class("text", video_t, "text", 1, "always"),
                _modulation_class("target_audio", audio_t, "audio", 2, "always"),
                _modulation_class(
                    "condition_video",
                    max(video_t, clean_video),
                    "video",
                    0,
                    "if_condition_video_rows",
                ),
                _modulation_class(
                    "condition_audio", 1.0, "audio", 2, "if_condition_audio_rows"
                ),
            ]
            evaluations.append(
                {
                    "index": index,
                    "video_sigma": _f32(video_sigma).hex(),
                    "audio_sigma": _f32(audio_sigma).hex(),
                    "modulation_classes": classes,
                }
            )
            for item in classes:
                timestep = str(item["timestep"])
                tag = int(item["modality_tag"])
                if (timestep, tag) not in seen_blocks:
                    seen_blocks.add((timestep, tag))
                    block_keys.append(
                        {
                            "index": len(block_keys),
                            "timestep": timestep,
                            "modality": item["modality"],
                            "modality_tag": tag,
                        }
                    )
                if timestep not in seen_final:
                    seen_final.add(timestep)
                    final_keys.append({"index": len(final_keys), "timestep": timestep})
        schedules.append(
            {
                "transformer_evaluations": len(video) - 1,
                "sigma_grid_points": points,
                "evaluations": evaluations,
                "terminal": {
                    "video_sigma": _f32(video[-1]).hex(),
                    "audio_sigma": _f32(audio[-1]).hex(),
                    "transformer_evaluation": False,
                },
            }
        )
    return canonical_json.encode(
        {
            "task": task,
            "scalar_encoding": "ieee754-binary32-hex",
            "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
            "row_timestep_reduction": "unique-sorted-return-inverse",
            "adaln_row_index": "timestep_index*3+modality_tag",
            "final_norm_row_index": "timestep_index",
            "table_order": "first-distinct-evaluation-class-occurrence",
            "fps": FPS,
            "video_shift": _f32(video_shift).hex(),
            "audio_shift": _f32(audio_shift).hex(),
            "schedules": schedules,
            "table_keys": {"block_modulation": block_keys, "final_normalization": final_keys},
        }
    )
