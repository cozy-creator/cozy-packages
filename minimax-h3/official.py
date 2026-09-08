"""The official Diffusers 0.40 MiniMax-H3 graph, staged by weighted component.

This module contains no media decoder and no checkpoint loader. Runtime supplies immutable
decoded values and fills the component roots constructed here. Diffusers owns every model
operation: presentation, conditioning, layout, schedules, solver, and decode. Artifact config
selects the official FULL modulation modules or AdaLN-pruned table replacements before fill.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import msgspec
import numpy as np
import torch
from cozy_runtime.author import (
    Config,
    ConformanceError,
    DecodedAudio,
    DecodedImage,
    DecodedVideo,
    InvalidRequest,
    OutputError,
    Telemetry,
    canonical_json,
)
from diffusers import (
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from torch.utils._pytree import keystr, tree_flatten_with_path
from transformers import (
    AddedToken,
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from adaln_pruned import AdaLNPrunedMiniMaxH3Transformer
from conditioner import build_text_conditioner
from vae_tiles import TileBatchedVideoVAE

Task = Literal["fl2va", "ref2va"]
_TASKS: tuple[Task, ...] = ("fl2va", "ref2va")
_WORKFLOW_TASKS: dict[str, Task] = {"t2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}

FPS = 24
FRAMES = 345
MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3
MAX_AUDIO_REFERENCES = 3
MAX_REFERENCES = 12
MAX_CONDITIONER_VISION_TOKENS = 32768
REFERENCE_IMAGE_SHORT_EDGE = 2048
_WEIGHTED_CONFIG_SECTIONS = {
    "audio_vae",
    "fl2va_dit",
    "ref2va_dit",
    "text_encoder",
    "video_vae",
}
_DIT_COMPONENT = {"fl2va": "fl2va_dit", "ref2va": "ref2va_dit"}
_DIFFUSERS_DIT = {"fl2va": "transformer", "ref2va": "transformer_ref"}
_ASSETS = Path(__file__).resolve().parent
#: The dry warm forward (`warm_dit`): rows per modality, and the modality tags the DiT
#: reads off `token_tags`. Eight rows is the smallest count that still runs both patch
#: projections, the refiner, the packed block stack and its glue at least once; the noise
#: levels are the plan's own first evaluation, so an AdaLN-pruned table holds every
#: (timestep, modality) pair the step gathers.
_WARM_ROWS = 8
_WARM_VIDEO_TAG, _WARM_TEXT_TAG, _WARM_AUDIO_TAG = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Schedule:
    """One official sigma grid per modality: `MiniMaxH3Scheduler.set_timesteps(points)`.

    `linspace(1, 0, sigma_grid_points)` through each modality's exponential shift, float32
    collisions collapsed; the terminal zero is a grid point with no transformer evaluation.
    """

    sigma_grid_points: int
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]

    def __post_init__(self) -> None:
        points = len(self.video_sigmas)
        if points < 2 or len(self.audio_sigmas) != points or points > self.sigma_grid_points:
            raise ValueError(
                "a MiniMax-H3 schedule holds equally many video and audio sigmas, "
                "at most one per grid point"
            )
        for sigmas in (self.video_sigmas, self.audio_sigmas):
            if (
                sigmas[0] != 1.0
                or sigmas[-1] != 0.0
                or any(not math.isfinite(value) or value != _as_float32(value) for value in sigmas)
                or any(left <= right for left, right in pairwise(sigmas))
            ):
                raise ValueError(
                    "MiniMax-H3 sigmas must be exact float32 values descending from one to zero"
                )

    @property
    def transformer_evaluations(self) -> int:
        return len(self.video_sigmas) - 1

    @property
    def video_timesteps(self) -> tuple[float, ...]:
        return tuple(_as_float32(1.0 - sigma) for sigma in self.video_sigmas[:-1])

    @property
    def audio_timesteps(self) -> tuple[float, ...]:
        return tuple(_as_float32(1.0 - sigma) for sigma in self.audio_sigmas[:-1])


@dataclass(frozen=True, slots=True)
class ScheduleFacts:
    """The receipt of the one schedule an attempt executed, under its plan identity."""

    timestep_plan_digest: str
    transformer_evaluations: int
    sigma_grid_points: int
    video_sigma_digest: str
    audio_sigma_digest: str
    video_timestep_digest: str
    audio_timestep_digest: str

    @classmethod
    def of(cls, plan: TimestepPlan, schedule: Schedule) -> ScheduleFacts:
        return cls(
            timestep_plan_digest=plan.digest,
            transformer_evaluations=schedule.transformer_evaluations,
            sigma_grid_points=schedule.sigma_grid_points,
            video_sigma_digest=_float32_digest(schedule.video_sigmas),
            audio_sigma_digest=_float32_digest(schedule.audio_sigmas),
            video_timestep_digest=_float32_digest(schedule.video_timesteps),
            audio_timestep_digest=_float32_digest(schedule.audio_timesteps),
        )


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """The exact official two-modality schedules one task serves; no weights required.

    An AdaLN-pruned checkpoint carries modulation rows for the union of these schedules,
    so the plan identity names both the step choices a request may make and the table
    layout the checkpoint must contain.
    """

    task: Task
    video_shift: float
    audio_shift: float
    schedules: tuple[Schedule, ...]

    def __post_init__(self) -> None:
        if self.task not in ("fl2va", "ref2va"):
            raise ValueError(f"unknown MiniMax-H3 task {self.task!r}")
        if any(
            not math.isfinite(shift) or shift <= 0 or shift != _as_float32(shift)
            for shift in (self.video_shift, self.audio_shift)
        ):
            raise ValueError("MiniMax-H3 shifts must be finite positive float32 values")
        if not self.schedules or any(
            left.transformer_evaluations >= right.transformer_evaluations
            for left, right in pairwise(self.schedules)
        ):
            raise ValueError(
                "a MiniMax-H3 plan lists its schedules by strictly ascending evaluation count"
            )

    @property
    def steps(self) -> tuple[int, ...]:
        """The denoise step counts (transformer evaluations) this plan serves."""
        return tuple(schedule.transformer_evaluations for schedule in self.schedules)

    def schedule(self, steps: int) -> Schedule:
        for schedule in self.schedules:
            if schedule.transformer_evaluations == steps:
                return schedule
        raise InvalidRequest(
            f"this release serves {', '.join(map(str, self.steps))} denoise steps, not {steps}",
            code="steps",
            fields=["steps"],
        )

    def executed(
        self, video_timesteps: Sequence[float], audio_timesteps: Sequence[float]
    ) -> Schedule:
        """The one schedule whose exact timesteps the official pipeline just built."""
        for schedule in self.schedules:
            if (
                tuple(video_timesteps) == schedule.video_timesteps
                and tuple(audio_timesteps) == schedule.audio_timesteps
            ):
                return schedule
        raise ConformanceError(
            f"the official {self.task} schedule differs from every plan schedule",
            code="artifact_config",
        )

    @property
    def digest(self) -> str:
        return timestep_plan_digest(self.canonical_bytes())

    def canonical_bytes(self) -> bytes:
        """Canonical producer handoff; float32 values are exact hex strings."""
        return canonical_json.encode(self._document())

    def table_layout(self) -> tuple[tuple[float, ...], tuple[tuple[int, int], ...]]:
        """The table order named by these exact canonical plan bytes."""
        keys = self._document()["table_keys"]
        final_rows = keys["final_normalization"]
        timesteps = tuple(float.fromhex(row["timestep"]) for row in final_rows)
        timestep_rows = {value: index for index, value in enumerate(timesteps)}
        block_rows = tuple(
            (timestep_rows[float.fromhex(row["timestep"])], int(row["modality_tag"]))
            for row in keys["block_modulation"]
        )
        return timesteps, block_rows

    def _document(self) -> dict[str, Any]:
        clean_video = _as_float32(0.999)
        condition_audio = _as_float32(1.0)
        schedules: list[dict[str, Any]] = []
        block_keys: list[dict[str, Any]] = []
        final_keys: list[dict[str, Any]] = []
        seen_blocks: set[tuple[str, int]] = set()
        seen_final: set[str] = set()
        for schedule in self.schedules:
            evaluations: list[dict[str, Any]] = []
            for index, (video_timestep, audio_timestep) in enumerate(
                zip(schedule.video_timesteps, schedule.audio_timesteps, strict=True)
            ):
                classes = [
                    _modulation_class("target_video", video_timestep, "video", 0, "always"),
                    _modulation_class("text", video_timestep, "text", 1, "always"),
                    _modulation_class("target_audio", audio_timestep, "audio", 2, "always"),
                    _modulation_class(
                        "condition_video",
                        max(video_timestep, clean_video),
                        "video",
                        0,
                        "if_condition_video_rows",
                    ),
                    _modulation_class(
                        "condition_audio",
                        condition_audio,
                        "audio",
                        2,
                        "if_condition_audio_rows",
                    ),
                ]
                evaluations.append(
                    {
                        "index": index,
                        "video_sigma": _float_hex(schedule.video_sigmas[index]),
                        "audio_sigma": _float_hex(schedule.audio_sigmas[index]),
                        "modulation_classes": classes,
                    }
                )
                for modulation in classes:
                    timestep = modulation["timestep"]
                    block_key = (timestep, modulation["modality_tag"])
                    if block_key not in seen_blocks:
                        seen_blocks.add(block_key)
                        block_keys.append(
                            {
                                "index": len(block_keys),
                                "timestep": timestep,
                                "modality": modulation["modality"],
                                "modality_tag": modulation["modality_tag"],
                            }
                        )
                    if timestep not in seen_final:
                        seen_final.add(timestep)
                        final_keys.append({"index": len(final_keys), "timestep": timestep})
            schedules.append(
                {
                    "transformer_evaluations": schedule.transformer_evaluations,
                    "sigma_grid_points": schedule.sigma_grid_points,
                    "evaluations": evaluations,
                    "terminal": {
                        "video_sigma": _float_hex(schedule.video_sigmas[-1]),
                        "audio_sigma": _float_hex(schedule.audio_sigmas[-1]),
                        "transformer_evaluation": False,
                    },
                }
            )
        return {
            "task": self.task,
            "scalar_encoding": "ieee754-binary32-hex",
            "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
            "row_timestep_reduction": "unique-sorted-return-inverse",
            "adaln_row_index": "timestep_index*3+modality_tag",
            "final_norm_row_index": "timestep_index",
            "table_order": "first-distinct-evaluation-class-occurrence",
            "frames": FRAMES,
            "fps": FPS,
            "video_shift": _float_hex(self.video_shift),
            "audio_shift": _float_hex(self.audio_shift),
            "schedules": schedules,
            "table_keys": {
                "block_modulation": block_keys,
                "final_normalization": final_keys,
            },
        }


def timestep_plan_digest(raw: bytes) -> str:
    """Formatting-invariant identity for one parsed timestep-plan document."""
    document = canonical_json.decode(raw)
    if not isinstance(document, dict):
        raise ValueError("a timestep plan must be a JSON object")
    return canonical_json.digest(document).removeprefix("sha256:")


class ReferencePolicyFacts(msgspec.Struct, frozen=True):
    """Preflight facts: the one reference-cardinality record both modules share."""

    images: int
    videos: int
    audios: int
    total: int


def validate_reference_policy(kinds: Sequence[str]) -> ReferencePolicyFacts:
    """Validate the official ordered-reference cardinality policy before hydration."""
    unknown = [kind for kind in kinds if kind not in {"image", "video", "audio"}]
    if unknown:
        raise ValueError(f"unknown MiniMax-H3 reference kind {unknown[0]!r}")
    facts = ReferencePolicyFacts(
        images=kinds.count("image"),
        videos=kinds.count("video"),
        audios=kinds.count("audio"),
        total=len(kinds),
    )
    for name, value, limit in (
        ("image", facts.images, MAX_IMAGE_REFERENCES),
        ("video", facts.videos, MAX_VIDEO_REFERENCES),
        ("audio", facts.audios, MAX_AUDIO_REFERENCES),
    ):
        if value > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {name} references, got {value}")
    if not 1 <= facts.total <= MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 needs 1..{MAX_REFERENCES} references, got {facts.total}")
    if facts.audios == facts.total:
        raise ValueError(
            "an audio reference must be paired with at least one image or video reference"
        )
    return facts


@lru_cache(maxsize=2)
def canonical_timestep_plan(task: Task) -> TimestepPlan:
    """Read and self-verify the committed official-oracle plan without a tensor device."""
    raw = (_ASSETS / "timestep-plans" / f"{task}.json").read_bytes()
    try:
        document = canonical_json.decode(raw)
        if not isinstance(document, dict):
            raise TypeError("timestep plan is not an object")
        schedules = []
        for row in document["schedules"]:
            evaluations = row["evaluations"]
            schedules.append(
                Schedule(
                    sigma_grid_points=int(row["sigma_grid_points"]),
                    video_sigmas=tuple(
                        [float.fromhex(entry["video_sigma"]) for entry in evaluations]
                        + [float.fromhex(row["terminal"]["video_sigma"])]
                    ),
                    audio_sigmas=tuple(
                        [float.fromhex(entry["audio_sigma"]) for entry in evaluations]
                        + [float.fromhex(row["terminal"]["audio_sigma"])]
                    ),
                )
            )
        plan = TimestepPlan(
            task=task,
            video_shift=float.fromhex(document["video_shift"]),
            audio_shift=float.fromhex(document["audio_shift"]),
            schedules=tuple(schedules),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"committed MiniMax-H3 {task} TimestepPlan is malformed") from exc
    if plan.canonical_bytes() != canonical_json.encode(document):
        raise ValueError(f"committed MiniMax-H3 {task} TimestepPlan has unexpected semantics")
    return plan


def supported_steps() -> tuple[int, ...]:
    """The denoise step counts both committed plans serve."""
    steps = {task: canonical_timestep_plan(task).steps for task in _TASKS}
    if len(set(steps.values())) != 1:
        raise ValueError("the FL2VA and Ref2VA plans must serve the same denoise step counts")
    return steps["fl2va"]


def reference_image_vision_tokens(
    width: int, height: int, short_edge: int = REFERENCE_IMAGE_SHORT_EDGE
) -> int:
    """Official short-edge, 16-patch, 2x2-merge image demand."""
    scale = short_edge / min(width, height)
    target_height = max(32, round(height * scale / 32) * 32)
    target_width = max(32, round(width * scale / 32) * 32)
    return target_height * target_width // (16 * 16 * 2 * 2)


def reference_video_vision_tokens(width: int, height: int, duration: Fraction) -> int:
    """Exact official 2-fps, pair-merged vision demand after the target canvas rule."""
    canvas = resolve_canvas_size(width, height, 32, 768, 768 * 1344)
    canvas_height, canvas_width = (int(value) for value in canvas)
    frames_at_24fps = _round_fraction(duration * FPS)
    sampled_frames = (frames_at_24fps + 11) // 12
    temporal_blocks = (sampled_frames + 1) // 2
    spatial_tokens = canvas_height * canvas_width // (16 * 16 * 2 * 2)
    return temporal_blocks * spatial_tokens


def _parked(tensor: Any) -> bool:
    """Runtime parks an evicted or paged-out weight on ``meta`` or in a zero-byte storage.
    A check claims only the resident values it counted, never unread storage."""
    return bool(tensor.is_meta or (tensor.numel() and not tensor.untyped_storage().nbytes()))


def _inspectable(tensor: Any) -> bool:
    return bool((tensor.is_floating_point() or tensor.is_complex()) and not _parked(tensor))


class ResidentWeights:
    """Finite-weight verdicts that live exactly as long as the bytes they describe.

    Weights are immutable once filled, so their scan runs once per fill rather than once
    per request (h3a-016). A verdict is keyed on the storage object itself, held weakly:
    eviction frees the storage and its verdict with it, and a re-stage fills a fresh
    storage that has none. The fill writes in place, which bumps the tensor's version
    counter, so a paged block refilled into a storage that was resized rather than
    replaced is rescanned too. Nothing here holds a strong reference to a weight.
    """

    def __init__(self) -> None:
        self._verdicts: weakref.WeakKeyDictionary[Any, tuple[int, int, int]] = (
            weakref.WeakKeyDictionary()
        )

    @staticmethod
    def _fill(tensor: Any) -> tuple[Any, tuple[int, int, int]]:
        storage = tensor.untyped_storage()
        return storage, (storage.data_ptr(), storage.nbytes(), tensor._version)

    def verified(self, tensor: Any) -> bool:
        storage, fill = self._fill(tensor)
        return self._verdicts.get(storage) == fill

    def record(self, tensor: Any) -> None:
        storage, fill = self._fill(tensor)
        self._verdicts[storage] = fill


@dataclass
class _Observation:
    """One stage's queued device reductions; nothing is read until it settles."""

    stage: str
    tensors: int = 0
    elements: int = 0
    parked: int = 0
    chunks: list[tuple[str, int]] = field(default_factory=list)
    nonfinite: list[Any] = field(default_factory=list)
    absmax: list[Any] = field(default_factory=list)


class NumericalChecks:
    """Request-local, bounded observations; no tensor is replaced or modified.

    A check queues its reductions on the device and reads them in one sync: at once for
    conditioning, decode and resident weights, and once after the denoise loop for the
    per-step hooks (``defer=True``), which used to drain the stream on every chunk of every
    step. Deferral changes nothing the customer can see: every step's tensors are still
    inspected, the verdict is read before any decode, and the first non-finite stage in
    step order is the one named.
    """

    def __init__(self, telemetry: Telemetry, resident: ResidentWeights | None = None) -> None:
        self.telemetry = telemetry
        self.resident = ResidentWeights() if resident is None else resident
        self._pending: list[_Observation] = []

    def tensors(
        self,
        stage: str,
        values: Sequence[tuple[str, Any]],
        *,
        required: bool = False,
        defer: bool = False,
    ) -> None:
        observation = self._observe(stage, values, required=required)
        if defer:
            self._pending.append(observation)
        else:
            self._settle([observation])

    def settle(self) -> None:
        """Read every deferred observation in order: one sync for a whole denoise loop."""
        pending, self._pending = self._pending, []
        self._settle(pending)

    def _observe(
        self, stage: str, values: Sequence[tuple[str, Any]], *, required: bool
    ) -> _Observation:
        observation = _Observation(stage)
        with torch.no_grad():
            for name, value in values:
                leaves, _ = tree_flatten_with_path(value)
                for path, tensor in leaves:
                    label = name + keystr(path)
                    if not isinstance(tensor, torch.Tensor):
                        if required and not isinstance(
                            tensor, (bool, int, float, str, bytes, type(None))
                        ):
                            raise OutputError(
                                f"uninspectable tensor output at {stage}/{label}: "
                                f"{type(tensor).__name__} needs registered PyTree flattening",
                                code="numerical_uninspectable",
                            )
                        continue
                    if not (tensor.is_floating_point() or tensor.is_complex()):
                        continue
                    if _parked(tensor):
                        observation.parked += 1
                        continue
                    observation.tensors += 1
                    # Stored FP8 reductions are not supported on every backend.
                    # Widen only this bounded observation, never the weight itself.
                    widen = tensor.element_size() == 1
                    chunk_elements = 1024 * 1024 if widen else 16 * 1024 * 1024
                    chunks = [tensor.detach()]
                    while chunks:
                        part = chunks.pop()
                        if part.numel() > chunk_elements:
                            axis = max(range(part.ndim), key=lambda dim: part.shape[dim])
                            chunks.extend(part.split(max(1, part.shape[axis] // 2), dim=axis))
                            continue
                        size = part.numel()
                        observation.elements += size
                        if not size:
                            continue
                        if widen:
                            part = part.float()
                        observation.chunks.append((label, size))
                        observation.nonfinite.append((~torch.isfinite(part)).sum())
                        observation.absmax.append(part.abs().amax().float())
        if required and not observation.elements:
            raise OutputError(
                f"no resident floating tensor values to inspect at {stage}",
                code="numerical_uninspectable",
            )
        return observation

    def _settle(self, observations: list[_Observation]) -> None:
        counts = self._read([count for o in observations for count in o.nonfinite])
        maxima = self._read([maximum for o in observations for maximum in o.absmax])
        position = 0
        for observation in observations:
            maximum, maximum_name = -1.0, ""
            for label, size in observation.chunks:
                bad, observed = int(counts[position]), float(maxima[position])
                position += 1
                if bad:
                    self.telemetry.log(
                        "h3 non-finite tensor",
                        stage=observation.stage,
                        tensor=label[:512],
                        chunk_elements=size,
                        chunk_nonfinite=bad,
                    )
                    raise OutputError(
                        f"non-finite tensor at {observation.stage}/{label}: "
                        f"{bad} of {size} values in the checked chunk",
                        code="numerical_nonfinite",
                    )
                if observed > maximum:
                    maximum, maximum_name = observed, label
            self.telemetry.log(
                "h3 numerical check",
                stage=observation.stage,
                status="finite_resident" if observation.elements else "no_resident_float_values",
                tensors=observation.tensors,
                elements=observation.elements,
                parked_tensors=observation.parked,
                absmax=max(maximum, 0.0),
                absmax_tensor=maximum_name[:512],
            )

    @staticmethod
    def _read(scalars: list[Any]) -> list[float]:
        """Copy queued scalars back in one sync per device, whatever mix they sit on."""
        values: dict[int, float] = {}
        by_device: dict[Any, list[int]] = {}
        for index, scalar in enumerate(scalars):
            by_device.setdefault(scalar.device, []).append(index)
        for indices in by_device.values():
            read = torch.stack([scalars[index] for index in indices]).tolist()
            values.update(zip(indices, read, strict=True))
        return [values[index] for index in range(len(scalars))]

    def component(self, name: str, module: Any) -> None:
        """Refuse non-finite resident weights before the first forward after a fill.

        Derived rotary/config buffers are not checkpoint payloads but can poison the same
        computation, so nonpersistent buffers are inspected under their exact names.
        """
        values = [*module.named_parameters(), *module.named_buffers()]
        resident = [(label, tensor) for label, tensor in values if _inspectable(tensor)]
        pending = [
            (label, tensor) for label, tensor in resident if not self.resident.verified(tensor)
        ]
        if resident and not pending:
            self.telemetry.log(
                "h3 numerical check",
                stage=f"resident.{name}",
                status="verified_fill",
                tensors=len(resident),
            )
            return
        self.tensors(f"resident.{name}", values if len(pending) == len(resident) else pending)
        for _, tensor in pending:
            self.resident.record(tensor)

    def outputs(self, stage: str, state: Any, outputs: Any, *, required: bool = False) -> None:
        self.tensors(
            stage,
            [(item.name, getattr(state, item.name, None)) for item in outputs if item.name],
            required=required,
        )

    @contextmanager
    def forwards(self, module: Any, component: str) -> Iterator[None]:
        """Observe every forward of one module; the verdicts are read when the block closes.

        A body that raises discards its pending observations with the attempt.
        """
        handles: list[Any] = []
        step = 0

        def before(_module: Any, args: Any, kwargs: Any) -> None:
            self.tensors(
                f"{component}.input.{step}",
                [("args", args), ("kwargs", kwargs)],
                required=True,
                defer=True,
            )

        def after(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            nonlocal step
            if output is not None:
                self.tensors(
                    f"{component}.prediction.{step}",
                    [("output", output)],
                    required=True,
                    defer=True,
                )
                step += 1

        try:
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
            yield
        except BaseException:
            self._pending.clear()
            raise
        finally:
            for handle in handles:
                handle.remove()
        self.settle()


class _ScopedPipeline:
    """A request-local view whose execution device follows the admitted component.

    Diffusers normally finds a device by scanning every registered module. Runtime stages
    one weighted root at a time, so that scan can see an inactive sibling first. The view
    changes no placement; it merely reports the device of the root whose Runtime scope is
    already active.
    """

    def __init__(
        self,
        pipe: Any,
        component: Any = None,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._pipe = pipe
        self._component = component
        self._overrides = {} if overrides is None else dict(overrides)

    @property
    def _execution_device(self) -> Any:
        return self._pipe._execution_device if self._component is None else self._component.device

    @property
    def device(self) -> Any:
        return self._pipe.device if self._component is None else self._component.device

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._pipe, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Blocks communicate through PipelineState. A write landing on this per-call view
        # would evaporate silently, and forwarding it would mutate the shared pipeline
        # across requests — so a non-private write is a typed refusal, never a silent one.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        raise ConformanceError(
            f"a wrapped Diffusers block set {name!r} on the request-local pipeline view",
            code="artifact_config",
        )


class _ImageEdges:
    """Request-local configuration view for the official ``ref2va`` setup step.

    The step reads ``reference_image_short_edge`` once per image reference, in packed order,
    so this view hands every image its own resolved edge and the official resize does the
    per-image work — no copied geometry, and the shared pipeline config is never touched.
    A read count that differs from the image count is upstream drift and refuses typed.
    """

    def __init__(self, config: Mapping[str, Any], edges: Sequence[int]) -> None:
        self._config = config
        self._edges = iter(edges)
        self._count = len(edges)

    def __getattr__(self, name: str) -> Any:
        if name != "reference_image_short_edge":
            return getattr(self._config, name)
        edge = next(self._edges, None)
        if edge is None:
            raise ConformanceError(
                f"official ref2va setup read more than {self._count} image short edges",
                code="artifact_config",
                fields=["pipeline", "reference_image_short_edge"],
            )
        return edge

    def settle(self) -> None:
        if next(self._edges, None) is not None:
            raise ConformanceError(
                f"official ref2va setup read fewer than {self._count} image short edges",
                code="artifact_config",
                fields=["pipeline", "reference_image_short_edge"],
            )


class OfficialH3Pipeline:
    """Both official task workflows over one shared, dual-DiT construction."""

    def __init__(self, config: Config) -> None:
        mapping = _artifact_sections(config.mapping())
        blocks = {name: MiniMaxH3Blocks().get_workflow(name) for name in _WORKFLOW_TASKS}
        pipes = {name: MiniMaxH3ModularPipeline(blocks=block) for name, block in blocks.items()}
        dit_specs = _dit_specs(mapping)
        dits = {
            task: _build_dit(upstream, structure, plan)
            for task, (upstream, structure, plan) in dit_specs.items()
        }
        _validate_dual_dit_topology(dits)
        text_encoder = build_text_conditioner(_section(mapping, "text_encoder"))
        video_vae = TileBatchedVideoVAE.from_config(_section(mapping, "video_vae")).eval()
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_config(_section(mapping, "audio_vae")).eval()
        for task in _TASKS:
            _validate_model_contract(pipes[task], dits[task], video_vae, audio_vae)
        tokenizer, processor = _processor()

        for workflow, task in _WORKFLOW_TASKS.items():
            pipes[workflow].register_components(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                processor=processor,
                vae=video_vae,
                audio_vae=audio_vae,
                scheduler=MiniMaxH3Scheduler(shift=12.0),
                audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
                **{_DIFFUSERS_DIT[task]: dits[task]},
            )

        # Runtime reads this mapping and nothing under ``pipe`` when deriving/filling
        # checkpoint destinations. Config-only processors and schedulers are deliberately
        # absent; ``video_vae`` is the artifact name while official Diffusers calls it
        # ``vae``.
        self.components: dict[str, Any] = {
            "fl2va_dit": dits["fl2va"],
            "ref2va_dit": dits["ref2va"],
            "text_encoder": text_encoder,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
        }
        # Finite-weight verdicts outlive requests exactly as the resident bytes do.
        self.resident = ResidentWeights()
        # The one release audio clock, read off the contract-checked artifact config
        # rather than respelled by callers.
        self.sample_rate = int(audio_vae.config.sampling_rate)
        self._blocks = blocks
        self._pipes = pipes
        self._plans = {task: spec[2] for task, spec in dit_specs.items()}

    def generator(self, source: object) -> Any:
        """Adapt Runtime's public request generator to Diffusers' torch generator."""
        if isinstance(source, torch.Generator):
            return source
        if not isinstance(source, random.Random):
            raise ConformanceError(
                f"request generator has unsupported type {type(source).__name__}",
                code="artifact_config",
            )
        return torch.Generator().manual_seed(source.getrandbits(63))

    def image_reference(self, image: DecodedImage) -> Any:
        pixels = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        return MiniMaxH3ImageReference(image=pixels)

    def keyframe(self, image: DecodedImage) -> Any:
        """Public RGB bytes into the input type required by the official resize block."""
        pixels = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        return self._pipes["fl2va"].image_processor.numpy_to_pil(pixels.astype(np.float32) / 255.0)[
            0
        ]

    def audio_reference(self, audio: DecodedAudio) -> Any:
        return MiniMaxH3AudioReference(audio=_audio_tensor(audio), sample_rate=audio.sample_rate)

    def video_reference(self, video: DecodedVideo) -> Any:
        soundtrack = _aligned_soundtrack(video)
        return MiniMaxH3VideoReference(
            frames=_video_at_24fps(video),
            fps=float(FPS),
            audio=soundtrack,
            sample_rate=None if video.soundtrack is None else video.soundtrack.sample_rate,
        )

    def start_fl2va(
        self,
        *,
        prompt: str,
        first_frame: Any | None,
        last_frame: Any | None,
        generator: Any,
        steps: int,
    ) -> Any:
        # With no anchor, state the official default canvas. With an anchor, leaving the
        # dimensions absent makes the first supplied keyframe the geometry authority.
        height, width = (768, 1344) if first_frame is None and last_frame is None else (None, None)
        return self._start(
            "fl2va",
            prompt=prompt,
            generator=generator,
            steps=steps,
            image=first_frame,
            last_image=last_frame,
            height=height,
            width=width,
        )

    def start_ref2va(
        self,
        *,
        prompt: str,
        references: Sequence[Any],
        generator: Any,
        steps: int,
        reference_image_short_edges: Sequence[int],
    ) -> Any:
        """``reference_image_short_edges`` is one resolved edge per image reference, in
        packed order; the official setup step resizes each image to its own."""
        return self._start(
            "ref2va",
            prompt=prompt,
            generator=generator,
            steps=steps,
            references=list(references),
            image_short_edges=tuple(reference_image_short_edges),
        )

    def _start(
        self,
        task: Task,
        *,
        prompt: str,
        generator: Any,
        steps: int,
        image_short_edges: Sequence[int] = (),
        **values: Any,
    ) -> Any:
        # The plan, not the request, spells the official grid: `steps` transformer
        # evaluations are the schedule's grid points less the terminal zero.
        schedule = self._plans[task].schedule(steps)
        state = PipelineState()
        fixed = {
            "prompt": prompt,
            "generator": generator,
            "num_frames": FRAMES,
            "num_inference_steps": schedule.sigma_grid_points,
            "output_type": "pt",
            **values,
        }
        for name, value in fixed.items():
            state.set(name, value)
        workflow = self._workflow(task, state)
        if "before_encode" in self._blocks[workflow].sub_blocks:
            pipe = self._pipes[workflow]
            edges = None
            if task == "ref2va":
                # Only setup uses this per-request geometry. A view leaves the shared
                # pipeline unchanged, even if preprocessing raises or calls overlap.
                edges = _ImageEdges(pipe.config, image_short_edges)
                pipe = _ScopedPipeline(pipe, overrides={"config": edges})
            self._run_with(task, pipe, "before_encode", state)
            if edges is not None:
                edges.settle()
        return state

    def condition_text(
        self, task: Task, state: Any, *, checks: NumericalChecks | None = None
    ) -> None:
        self._run(task, "text_encoder", state, component="text_encoder")
        if checks is not None:
            checks.outputs(
                "condition_text",
                state,
                self._blocks[self._workflow(task, state)]
                .sub_blocks["text_encoder"]
                .intermediate_outputs,
                required=True,
            )

    def condition_media(
        self, task: Task, state: Any, *, checks: NumericalChecks | None = None
    ) -> None:
        self._run(task, "vae_encoder", state, component="video_vae")
        if checks is not None:
            checks.outputs(
                "condition_media",
                state,
                self._blocks[self._workflow(task, state)]
                .sub_blocks["vae_encoder"]
                .intermediate_outputs,
                required=True,
            )

    def warm_dit(self, task: Task) -> None:
        """One dry forward of this task's DiT over a packed sequence of every modality.

        Not a generation (h3a-018 #709: an H3 warm case would be a 30-step request, and the
        runtime calls `warm` once per construction fill). What a first request would
        otherwise pay is shape-independent: the fused glue's first launches on the
        substituted blocks (h3a-015 — the executor loaded their cubins before `warm`), the
        rotary tables, and each projection's first GEMM plan. The rows are the plan's own
        first evaluation — target video and text at the first video timestep, target audio
        at the first audio timestep — which is the pairing an AdaLN-pruned table is built
        from. Inputs are zeros off the DiT's own rotary buffer, so nothing names a device
        and no generator moves; outputs are dropped.
        """
        dit = self.components[_DIT_COMPONENT[task]]
        schedule = self._plans[task].schedules[0]
        rows, tags = _WARM_ROWS, (_WARM_VIDEO_TAG, _WARM_TEXT_TAG, _WARM_AUDIO_TAG)
        packed = rows * len(tags)
        anchor = dit.rope.inv_freq
        rowwise = {
            "token_tags": [tag for tag in tags for _ in range(rows)],
            # Video and text rows ride the video timestep (index 0), audio rows the audio
            # timestep (index 1) — `timestep` below holds exactly those two values.
            "timestep_indices": [0] * (2 * rows) + [1] * rows,
            "video_indices": range(rows),
            "text_indices": range(rows, 2 * rows),
            "audio_indices": range(2 * rows, packed),
        }
        with torch.no_grad():
            dit(
                hidden_states=anchor.new_zeros(
                    1, rows, dit.config.in_channels * math.prod(dit.config.patch_size)
                ),
                audio_hidden_states=anchor.new_zeros(1, rows, dit.config.audio_in_channels),
                encoder_hidden_states=anchor.new_zeros(1, rows, dit.config.text_dim),
                timestep=anchor.new_tensor(
                    [schedule.video_timesteps[0], schedule.audio_timesteps[0]]
                ),
                position_ids=anchor.new_zeros(packed, 3, dtype=torch.int64),
                return_dict=False,
                **{
                    name: anchor.new_tensor(list(value), dtype=torch.int64)
                    for name, value in rowwise.items()
                },
            )

    def denoise(
        self,
        task: Task,
        state: Any,
        *,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
        checks: NumericalChecks | None = None,
    ) -> ScheduleFacts:
        blocks = self._blocks[self._workflow(task, state)].sub_blocks
        scoped = _ScopedPipeline(
            self._pipes[self._workflow(task, state)],
            self.components[_DIT_COMPONENT[task]],
            overrides={
                "scheduler": MiniMaxH3Scheduler(shift=12.0),
                "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
            },
        )
        # The selected upstream workflow owns the ordered preparation steps.
        # Text-only requests have no keyframe tensors to encode or concatenate.
        for name in blocks:
            if name == "denoise.denoise":
                break
            if not name.startswith("denoise."):
                continue
            self._run_with(task, scoped, name, state)
            if checks is not None:
                checks.outputs(
                    name,
                    state,
                    blocks[name].intermediate_outputs,
                    required=name.startswith("denoise.prepare_latents"),
                )

        # The receipt names the schedule the official blocks actually built, proven
        # against the plan by exact float32 equality of both modality vectors.
        plan = self._plans[task]
        facts = ScheduleFacts.of(
            plan,
            plan.executed(_float_tuple(state.timesteps), _float_tuple(state.audio_timesteps)),
        )
        loop = blocks["denoise.denoise"]
        block_state = loop.get_block_state(state)
        for index, timestep in enumerate(block_state.timesteps):
            cancel()
            _, block_state = loop.loop_step(scoped, block_state, i=index, t=timestep)
            if checks is not None:
                checks.tensors(
                    f"{_DIT_COMPONENT[task]}.updated.{index}",
                    [
                        ("latents", block_state.latents),
                        ("audio_latents", block_state.audio_latents),
                    ],
                    required=True,
                    defer=True,
                )
            on_step(index)
        if checks is not None:
            # One read for every step's inputs, predictions and updates, before any decode.
            # Waiting loses nothing: the official update is an affine blend of the current
            # latents and the prediction with no clamp, so a non-finite value that enters at
            # step k is in every later latent, and the first offending stage is still named.
            checks.settle()
        loop.set_block_state(state, block_state)
        self._run_with(task, scoped, "denoise.after_denoise", state)
        return facts

    def decode_audio(self, task: Task, state: Any) -> Any:
        self._run(task, "decode.audio", state, component="audio_vae")
        return state.audio

    def decode_video(self, task: Task, state: Any) -> Any:
        self._run(task, "decode.video", state, component="video_vae")
        return state.videos

    def _run(self, task: Task, name: str, state: Any, *, component: str | None = None) -> None:
        workflow = self._workflow(task, state)
        pipe = (
            self._pipes[workflow]
            if component is None
            else _ScopedPipeline(self._pipes[workflow], self.components[component])
        )
        self._run_with(task, pipe, name, state)

    def _run_with(self, task: Task, pipe: Any, name: str, state: Any) -> None:
        block = self._blocks[self._workflow(task, state)].sub_blocks[name]
        block(pipe, state)

    @staticmethod
    def _workflow(task: Task, state: Any) -> str:
        if task == "fl2va" and state.get("image") is None and state.get("last_image") is None:
            return "t2va"
        return task


def build_h3_pipeline(config: Config) -> OfficialH3Pipeline:
    return OfficialH3Pipeline(config)


def _section(mapping: Mapping[str, object], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise ConformanceError(
            f"artifact config has no {name!r} mapping", code="artifact_config", fields=[name]
        )
    return dict(value)


def _artifact_sections(mapping: Mapping[str, object]) -> Mapping[str, object]:
    present = set(mapping)
    if present != _WEIGHTED_CONFIG_SECTIONS:
        missing = sorted(_WEIGHTED_CONFIG_SECTIONS - present)
        unexpected = sorted(present - _WEIGHTED_CONFIG_SECTIONS)
        raise ConformanceError(
            f"artifact config sections differ from the uniform dual FULL contract: "
            f"missing={missing}, unexpected={unexpected}",
            code="artifact_config",
            fields=[*missing, *unexpected],
        )
    return mapping


def _dit_spec(
    mapping: Mapping[str, object], task: Task
) -> tuple[dict[str, Any], str, TimestepPlan]:
    component = _DIT_COMPONENT[task]
    upstream = _section(mapping, component)
    extension = upstream.pop("cozy_h3", None)
    if not isinstance(extension, Mapping):
        raise ConformanceError(
            f"artifact config {component!r} has no closed cozy_h3 structure",
            code="artifact_config",
            fields=[component, "cozy_h3"],
        )
    structure = extension.get("modulation")
    if structure not in {"full", "adaln-pruned"}:
        raise ConformanceError(
            f"artifact config {component!r} has unknown modulation structure {structure!r}",
            code="artifact_config",
            fields=[component, "cozy_h3", "modulation"],
        )
    # FULL computes modulation from live weights; only pruned tables bind a plan.
    fields = {"task", "modulation"}
    if structure == "adaln-pruned":
        fields.add("timestep_plan_digest")
    if set(extension) != fields:
        raise ConformanceError(
            f"artifact config {component!r} has no closed cozy_h3 structure",
            code="artifact_config",
            fields=[component, "cozy_h3"],
        )
    plan = canonical_timestep_plan(task)
    expected: dict[str, str] = {"task": task}
    if structure == "adaln-pruned":
        expected["timestep_plan_digest"] = f"sha256:{plan.digest}"
    for name, want in expected.items():
        if extension[name] != want:
            raise ConformanceError(
                f"artifact config {component!r} {name!r} is {extension[name]!r}, expected {want!r}",
                code="artifact_config",
                fields=[component, "cozy_h3", name],
            )
    return upstream, str(structure), plan


def _dit_specs(
    mapping: Mapping[str, object],
) -> dict[Task, tuple[dict[str, Any], str, TimestepPlan]]:
    specs = {task: _dit_spec(mapping, task) for task in _TASKS}
    if len({spec[1] for spec in specs.values()}) != 1:
        raise ConformanceError(
            "FL2VA and Ref2VA DiTs use different modulation structures",
            code="artifact_config",
            fields=["fl2va_dit", "ref2va_dit"],
        )
    if specs["fl2va"][0] != specs["ref2va"][0]:
        raise ConformanceError(
            "FL2VA and Ref2VA DiTs do not carry the same upstream architecture config",
            code="artifact_config",
            fields=["fl2va_dit", "ref2va_dit"],
        )
    return specs


def _build_dit(config: Mapping[str, Any], structure: str, plan: TimestepPlan) -> Any:
    if structure == "full":
        transformer = MiniMaxH3Transformer3DModel.from_config(dict(config))
    else:
        timesteps, block_keys = plan.table_layout()
        transformer = AdaLNPrunedMiniMaxH3Transformer.from_official_config(
            config,
            table_timesteps=timesteps,
            table_block_keys=block_keys,
        )
    return _apply_transformer_dtype(transformer)


def _validate_dual_dit_topology(dits: Mapping[Task, Any]) -> None:
    """Prove one architecture instantiated twice for two independent weight identities."""
    fl2va, ref2va = dits["fl2va"], dits["ref2va"]
    if fl2va is ref2va:
        raise ConformanceError(
            "FL2VA and Ref2VA component names alias one module instance",
            code="artifact_config",
            fields=["fl2va_dit", "ref2va_dit"],
        )

    def topology(module: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        return tuple(
            (name, tuple(int(value) for value in tensor.shape), str(tensor.dtype))
            for name, tensor in module.state_dict().items()
        )

    if type(fl2va) is not type(ref2va) or topology(fl2va) != topology(ref2va):
        raise ConformanceError(
            "FL2VA and Ref2VA DiTs do not have one identical class and destination topology",
            code="artifact_config",
            fields=["fl2va_dit", "ref2va_dit"],
        )


def _apply_transformer_dtype(transformer: Any) -> Any:
    """Reproduce Diffusers' mixed FULL compute policy on Runtime destinations."""
    # Cast each root directly to its final dtype. In particular, RoPE frequencies
    # are real config-derived buffers: BF16 followed by FP32 cannot restore them.
    for name, component in transformer.named_children():
        dtype = torch.float32 if name in transformer._keep_in_fp32_modules else torch.bfloat16
        component.to(dtype=dtype)
    return transformer.eval()


def _validate_model_contract(pipe: Any, transformer: Any, video_vae: Any, audio_vae: Any) -> None:
    """Refuse shape-preserving scalar drift in the one official release geometry."""
    contracts = (
        (
            "pipeline",
            pipe.config,
            {"canvas_short_edge": 768, "canvas_max_pixels": 768 * 1344},
        ),
        (
            "transformer",
            transformer.config,
            {
                "num_attention_heads": 56,
                "attention_head_dim": 128,
                "hidden_size": 5376,
                "num_layers": 50,
                "num_refiner_layers": 2,
                "ffn_dim": 14336,
                "in_channels": 24,
                "audio_in_channels": 32,
                "patch_size": (1, 2, 2),
                "text_dim": 5120,
                "freq_dim": 256,
                "time_embed_hidden_dim": 5376,
                "time_embed_dim": 2688,
                "rope_freq_dim": 16,
                "rope_theta": 10000.0,
                "norm_eps": 1e-5,
                "qk_norm_eps": 1e-5,
                "final_norm_eps": 1e-5,
            },
        ),
        (
            "video_vae",
            video_vae.config,
            {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 24,
                "spatial_downsample_factors": (2, 2, 2, 2, 1, 1),
                "temporal_downsample_factors": (1, 2, 2, 1, 1, 1),
                "clip_length": 17,
                "token_drop": 3,
            },
        ),
        (
            "audio_vae",
            audio_vae.config,
            {
                "encoder_rates": (2, 4, 4, 5, 5),
                "latent_dim": 2048,
                "latent_channels": 32,
                "decoder_rates": (5, 5, 2, 2, 2, 2, 2),
                "sampling_rate": 32000,
            },
        ),
    )
    for component, config, expected in contracts:
        for name, want in expected.items():
            got = config.get(name)
            if isinstance(want, tuple) and got is not None:
                got = tuple(got)
            if got != want:
                raise ConformanceError(
                    f"official H3 {component} config {name!r} is {got!r}, expected {want!r}",
                    code="artifact_config",
                    fields=[component, name],
                )
    if pipe.config.get("reference_image_short_edge", 2048) != 2048:
        raise ConformanceError(
            "official H3 reference image short edge differs from 2048",
            code="artifact_config",
            fields=["pipeline", "reference_image_short_edge"],
        )
    if video_vae.tokens_chunk_size != 5 or video_vae.spatial_compression_ratio != 16:
        raise ConformanceError(
            "official H3 video VAE derived geometry differs from 5-token/16-pixel release",
            code="artifact_config",
            fields=["video_vae"],
        )


def _processor() -> tuple[Any, Any]:
    vocab = _json_mapping(_ASSETS / "tokenizer" / "vocab.json")
    config = _json_mapping(_ASSETS / "tokenizer" / "tokenizer_config.json")
    config.pop("tokenizer_class", None)
    added_tokens = config.get("added_tokens_decoder")
    if not isinstance(added_tokens, Mapping) or any(
        not isinstance(value, Mapping) for value in added_tokens.values()
    ):
        raise ConformanceError(
            "bundled tokenizer added-token table is missing or malformed",
            code="artifact_config",
        )
    try:
        config["added_tokens_decoder"] = {
            int(index): AddedToken(**dict(value)) for index, value in added_tokens.items()
        }
    except (TypeError, ValueError) as exc:
        raise ConformanceError(
            "bundled tokenizer added-token table is malformed", code="artifact_config"
        ) from exc
    merge_lines = (_ASSETS / "tokenizer" / "merges.txt").read_text().splitlines()
    merges = [tuple(line.split(" ")) for line in merge_lines]
    if not merges or any(len(pair) != 2 for pair in merges):
        raise ConformanceError(
            "bundled tokenizer merges are empty or malformed", code="artifact_config"
        )
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=merges, **config)

    image_config = _json_mapping(_ASSETS / "processor" / "preprocessor_config.json")
    video_config = _json_mapping(_ASSETS / "processor" / "video_preprocessor_config.json")
    image_config.pop("processor_class", None)
    image_config.pop("image_processor_type", None)
    video_config.pop("processor_class", None)
    video_config.pop("video_processor_type", None)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(**image_config),
        video_processor=Qwen3VLVideoProcessor(**video_config),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
    )
    return tokenizer, processor


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ConformanceError(
            f"bundled processor asset {path.name!r} is unreadable",
            code="artifact_config",
        ) from exc
    if not isinstance(value, dict):
        raise ConformanceError(
            f"bundled processor asset {path.name!r} is not a JSON object",
            code="artifact_config",
        )
    return value


def _audio_tensor(audio: DecodedAudio) -> Any:
    channels = [
        torch.frombuffer(bytearray(channel), dtype=torch.float32) for channel in audio.pcm_f32le
    ]
    return torch.stack(channels)


def _aligned_soundtrack(video: DecodedVideo) -> Any | None:
    """Put embedded PCM on the video's origin without rounding either native clock."""
    audio = video.soundtrack
    if audio is None:
        return None
    exact_offset = (audio.start_time - video.start_time) * audio.sample_rate
    exact_samples = video.duration * audio.sample_rate
    if exact_offset.denominator != 1:
        raise ConformanceError(
            "reference soundtrack offset is not aligned to its sample clock",
            code="reference_policy",
        )
    if exact_samples.denominator != 1:
        raise ConformanceError(
            "reference video duration is not aligned to its soundtrack sample clock",
            code="reference_policy",
        )
    offset = exact_offset.numerator
    target_samples = exact_samples.numerator
    waveform = _audio_tensor(audio)
    start = max(offset, 0)
    end = min(offset + audio.sample_count, target_samples)
    if start >= end:
        raise ConformanceError(
            "reference soundtrack has no samples on its video's timeline",
            code="reference_policy",
        )
    aligned = torch.zeros((audio.channels, target_samples), dtype=torch.float32)
    source_start = start - offset
    aligned[:, start:end] = waveform[:, source_start : source_start + end - start]
    return aligned


def _video_at_24fps(video: DecodedVideo) -> Any:
    """Exact presentation boundaries onto the official whole-frame 24-fps clock."""
    starts = video.frame_pts
    durations = video.frame_durations
    for index in range(len(starts) - 1):
        if starts[index] + durations[index] != starts[index + 1]:
            raise ConformanceError(
                f"reference video clock has a gap or overlap before frame {index + 1}",
                code="reference_policy",
            )

    origin = starts[0]
    boundaries = [Fraction(value - origin) * video.time_base for value in starts]
    boundaries.append(Fraction(starts[-1] + durations[-1] - origin) * video.time_base)
    slots = [_round_fraction(boundary * FPS) for boundary in boundaries]
    selected: list[int] = []
    for index in range(video.frame_count):
        selected.extend([index] * (slots[index + 1] - slots[index]))
    if not selected:
        raise ConformanceError(
            "reference video has no frame on the 24-fps clock", code="reference_policy"
        )

    frames = []
    for index in selected:
        raw = video.frames_rgb[index]
        frame = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        frames.append(frame.reshape(video.height, video.width, 3).permute(2, 0, 1))
    return torch.stack(frames)


def _round_fraction(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _float_tuple(tensor: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().float().cpu())


def _float32_digest(values: Sequence[float]) -> str:
    raw = b"".join(struct.pack("<f", value) for value in values)
    return hashlib.sha256(raw).hexdigest()


def _as_float32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _float_hex(value: float) -> str:
    return _as_float32(value).hex()


def _modulation_class(
    name: str,
    timestep: float,
    modality: str,
    tag: int,
    presence: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "timestep": _float_hex(timestep),
        "modality": modality,
        "modality_tag": tag,
        "presence": presence,
    }
