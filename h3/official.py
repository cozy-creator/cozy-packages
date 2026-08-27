"""The official Diffusers 0.40 MiniMax-H3 graph, staged by weighted component.

This module contains no media decoder and no checkpoint loader. Runtime supplies immutable
decoded values and fills the component roots constructed here. Diffusers owns every model
operation: presentation, conditioning, layout, schedules, FULL AdaLN, solver, and decode.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from cozy_runtime.author import (
    Config,
    ConformanceError,
    DecodedAudio,
    DecodedImage,
    DecodedVideo,
)

Task = Literal["fl2va", "ref2va"]

FPS = 24
FRAMES = 345
SIGMA_GRID_POINTS = 30
TRANSFORMER_EVALUATIONS = 29
MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3
MAX_AUDIO_REFERENCES = 3
MAX_REFERENCES = 12
MAX_CONDITIONER_VISION_TOKENS = 32768
_WEIGHTED_CONFIG_SECTIONS = {
    "audio_vae",
    "text_encoder",
    "transformer",
    "transformer_ref",
    "video_vae",
}
_ASSETS = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class ScheduleFacts:
    timestep_plan_digest: str
    video_sigma_digest: str
    audio_sigma_digest: str
    video_timestep_digest: str
    audio_timestep_digest: str


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """The exact official 30-point, two-modality schedule; no weights required."""

    task: Task
    video_shift: float
    audio_shift: float
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    video_timesteps: tuple[float, ...]
    audio_timesteps: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.task not in ("fl2va", "ref2va"):
            raise ValueError(f"unknown MiniMax-H3 task {self.task!r}")
        if any(
            not math.isfinite(shift) or shift <= 0 or shift != _as_float32(shift)
            for shift in (self.video_shift, self.audio_shift)
        ):
            raise ValueError("MiniMax-H3 shifts must be finite positive float32 values")
        if (
            len(self.video_sigmas) != SIGMA_GRID_POINTS
            or len(self.audio_sigmas) != SIGMA_GRID_POINTS
        ):
            raise ValueError("a MiniMax-H3 plan must contain exactly 30 video and audio sigmas")
        if (
            len(self.video_timesteps) != TRANSFORMER_EVALUATIONS
            or len(self.audio_timesteps) != TRANSFORMER_EVALUATIONS
        ):
            raise ValueError("a MiniMax-H3 plan must contain exactly 29 video and audio timesteps")
        if self.video_sigmas[-1] != 0.0 or self.audio_sigmas[-1] != 0.0:
            raise ValueError("the terminal video and audio sigmas must both be zero")
        if self.video_sigmas != _shifted_sigmas(
            self.video_shift
        ) or self.audio_sigmas != _shifted_sigmas(self.audio_shift):
            raise ValueError("MiniMax-H3 sigmas must exactly match their official float32 shifts")
        for sigmas, timesteps in (
            (self.video_sigmas, self.video_timesteps),
            (self.audio_sigmas, self.audio_timesteps),
        ):
            expected = tuple(_as_float32(1.0 - sigma) for sigma in sigmas[:-1])
            if timesteps != expected:
                raise ValueError("MiniMax-H3 timesteps must be float32 one-minus-sigma values")

    @property
    def video_sigma_digest(self) -> str:
        return _float32_digest(self.video_sigmas)

    @property
    def audio_sigma_digest(self) -> str:
        return _float32_digest(self.audio_sigmas)

    @property
    def video_timestep_digest(self) -> str:
        return _float32_digest(self.video_timesteps)

    @property
    def audio_timestep_digest(self) -> str:
        return _float32_digest(self.audio_timesteps)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        """Canonical job-010 handoff; float32 values are exact hex strings."""
        clean_video = _as_float32(0.999)
        condition_audio = _as_float32(1.0)
        evaluations: list[dict[str, Any]] = []
        for index, (video_timestep, audio_timestep) in enumerate(
            zip(self.video_timesteps, self.audio_timesteps, strict=True)
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
                    "video_sigma": _float_hex(self.video_sigmas[index]),
                    "audio_sigma": _float_hex(self.audio_sigmas[index]),
                    "modulation_classes": classes,
                }
            )
        block_keys: list[dict[str, Any]] = []
        final_keys: list[dict[str, Any]] = []
        seen_blocks: set[tuple[str, int]] = set()
        seen_final: set[str] = set()
        for evaluation in evaluations:
            for modulation in evaluation["modulation_classes"]:
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
        document = {
            "schema": "cozy.minimax_h3.timestep_plan/1",
            "task": self.task,
            "scalar_encoding": "ieee754-binary32-hex",
            "scheduler_semantics": "minimax-h3-data-ward-rf-euler/1",
            "row_timestep_reduction": "unique-sorted-return-inverse",
            "adaln_row_index": "timestep_index*3+modality_tag",
            "final_norm_row_index": "timestep_index",
            "baked_table_order": "first-distinct-evaluation-class-occurrence",
            "frames": FRAMES,
            "fps": FPS,
            "sigma_grid_points": SIGMA_GRID_POINTS,
            "transformer_evaluations": TRANSFORMER_EVALUATIONS,
            "video_shift": _float_hex(self.video_shift),
            "audio_shift": _float_hex(self.audio_shift),
            "evaluations": evaluations,
            "baked_table_keys": {
                "block_modulation": block_keys,
                "final_normalization": final_keys,
            },
            "terminal": {
                "video_sigma": _float_hex(self.video_sigmas[-1]),
                "audio_sigma": _float_hex(self.audio_sigmas[-1]),
                "transformer_evaluation": False,
            },
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


@dataclass(frozen=True, slots=True)
class ReferencePolicyFacts:
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


def canonical_timestep_plan(task: Task) -> TimestepPlan:
    """Derive Diffusers' exact float32 FULL schedule without constructing any model."""
    video_sigmas = _shifted_sigmas(12.0)
    audio_sigmas = _shifted_sigmas(3.0)
    return TimestepPlan(
        task=task,
        video_shift=12.0,
        audio_shift=3.0,
        video_sigmas=video_sigmas,
        audio_sigmas=audio_sigmas,
        video_timesteps=_timesteps(video_sigmas),
        audio_timesteps=_timesteps(audio_sigmas),
    )


def reference_image_vision_tokens(width: int, height: int) -> int:
    """Exact official 2048-short-edge, 16-patch, 2x2-merge image demand."""
    scale = 2048 / min(width, height)
    target_height = max(32, round(height * scale / 32) * 32)
    target_width = max(32, round(width * scale / 32) * 32)
    return target_height * target_width // (16 * 16 * 2 * 2)


def reference_video_vision_tokens(width: int, height: int, duration: Fraction) -> int:
    """Exact official 2-fps, pair-merged vision demand after the target canvas rule."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

    canvas = resolve_canvas_size(width, height, 32, 768, 768 * 1344)
    canvas_height, canvas_width = (int(value) for value in canvas)
    frames_at_24fps = _round_fraction(duration * FPS)
    sampled_frames = (frames_at_24fps + 11) // 12
    temporal_blocks = (sampled_frames + 1) // 2
    spatial_tokens = canvas_height * canvas_width // (16 * 16 * 2 * 2)
    return temporal_blocks * spatial_tokens


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
        component: Any,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._pipe = pipe
        self._component = component
        self._overrides = {} if overrides is None else dict(overrides)

    @property
    def _execution_device(self) -> Any:
        return self._component.device

    @property
    def device(self) -> Any:
        return self._component.device

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._pipe, name)


class OfficialH3Pipeline:
    """One task-pruned official workflow and its four weighted component roots."""

    def __init__(self, config: Config, *, task: Task) -> None:
        import torch
        from diffusers import (
            AutoencoderKLMiniMaxH3,
            AutoencoderKLMiniMaxH3Audio,
            MiniMaxH3Blocks,
            MiniMaxH3ModularPipeline,
            MiniMaxH3Scheduler,
            MiniMaxH3Transformer3DModel,
        )
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        mapping = _artifact_sections(config.mapping())
        blocks = MiniMaxH3Blocks().get_workflow(task)
        pipe = MiniMaxH3ModularPipeline(blocks=blocks)

        transformer_name = "transformer" if task == "fl2va" else "transformer_ref"
        transformer = _apply_transformer_dtype(
            MiniMaxH3Transformer3DModel.from_config(_section(mapping, transformer_name))
        )
        text_encoder = (
            Qwen3VLForConditionalGeneration(Qwen3VLConfig(**_section(mapping, "text_encoder")))
            .to(dtype=torch.bfloat16)
            .eval()
        )
        video_vae = AutoencoderKLMiniMaxH3.from_config(_section(mapping, "video_vae")).eval()
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_config(_section(mapping, "audio_vae")).eval()
        _validate_model_contract(pipe, transformer, video_vae, audio_vae)
        tokenizer, processor = _processor()

        registered = {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "vae": video_vae,
            "audio_vae": audio_vae,
            "scheduler": MiniMaxH3Scheduler(shift=12.0),
            "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
            transformer_name: transformer,
        }
        pipe.register_components(**registered)

        # Runtime reads this mapping and nothing under ``pipe`` when deriving/filling
        # checkpoint destinations. Config-only processors and schedulers are deliberately
        # absent; ``video_vae`` is the artifact name while official Diffusers calls it
        # ``vae``.
        self.components: dict[str, Any] = {
            transformer_name: transformer,
            "text_encoder": text_encoder,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
        }
        self.task = task
        self._blocks = blocks
        self._pipe = pipe
        self._transformer_name = transformer_name

    def generator(self, source: object) -> Any:
        """Adapt Runtime's public request generator to Diffusers' torch generator."""
        import random

        import torch

        if isinstance(source, torch.Generator):
            return source
        if not isinstance(source, random.Random):
            raise ConformanceError(
                f"request generator has unsupported type {type(source).__name__}",
                code="generator_type",
            )
        return torch.Generator().manual_seed(source.getrandbits(63))

    def image_reference(self, image: DecodedImage) -> Any:
        import numpy as np
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

        pixels = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        return MiniMaxH3ImageReference(image=pixels)

    def keyframe(self, image: DecodedImage) -> Any:
        """Public RGB bytes into the input type required by the official resize block."""
        import numpy as np

        pixels = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        return self._pipe.image_processor.numpy_to_pil(pixels.astype(np.float32) / 255.0)[0]

    def audio_reference(self, audio: DecodedAudio) -> Any:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference

        return MiniMaxH3AudioReference(audio=_audio_tensor(audio), sample_rate=audio.sample_rate)

    def video_reference(self, video: DecodedVideo) -> Any:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference

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
    ) -> Any:
        # With no anchor, state the official default canvas. With an anchor, leaving the
        # dimensions absent makes the first supplied keyframe the geometry authority.
        height, width = (768, 1344) if first_frame is None and last_frame is None else (None, None)
        return self._start(
            prompt=prompt,
            generator=generator,
            image=first_frame,
            last_image=last_frame,
            height=height,
            width=width,
        )

    def start_ref2va(self, *, prompt: str, references: Sequence[Any], generator: Any) -> Any:
        return self._start(prompt=prompt, generator=generator, references=list(references))

    def _start(self, *, prompt: str, generator: Any, **values: Any) -> Any:
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        state = PipelineState()
        fixed = {
            "prompt": prompt,
            "generator": generator,
            "num_frames": FRAMES,
            "num_inference_steps": SIGMA_GRID_POINTS,
            "output_type": "pt",
            **values,
        }
        for name, value in fixed.items():
            state.set(name, value)
        self._run("before_encode", state)
        return state

    def condition_text(self, state: Any) -> None:
        self._run("text_encoder", state, component="text_encoder")

    def condition_media(self, state: Any) -> None:
        self._run("vae_encoder", state, component="video_vae")

    def denoise(
        self,
        state: Any,
        *,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> ScheduleFacts:
        from diffusers import MiniMaxH3Scheduler

        scoped = _ScopedPipeline(
            self._pipe,
            self.components[self._transformer_name],
            overrides={
                "scheduler": MiniMaxH3Scheduler(shift=12.0),
                "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
            },
        )
        names = [
            "denoise.prepare_layout",
            "denoise.prepare_condition_latents",
            "denoise.prepare_latents",
            (
                "denoise.prepare_latents_fl2va"
                if self.task == "fl2va"
                else "denoise.prepare_latents_ref2va"
            ),
            "denoise.set_timesteps",
        ]
        for name in names:
            self._run_with(scoped, name, state)

        facts = self._schedule_facts(scoped, state)
        loop = self._blocks.sub_blocks["denoise.denoise"]
        block_state = loop.get_block_state(state)
        for index, timestep in enumerate(block_state.timesteps):
            cancel()
            _, block_state = loop.loop_step(scoped, block_state, i=index, t=timestep)
            on_step(index)
        loop.set_block_state(state, block_state)
        self._run_with(scoped, "denoise.after_denoise", state)
        return facts

    def decode_audio(self, state: Any) -> Any:
        self._run("decode.audio", state, component="audio_vae")
        return state.audio

    def decode_video(self, state: Any) -> Any:
        self._run("decode.video", state, component="video_vae")
        return state.videos

    def _run(self, name: str, state: Any, *, component: str | None = None) -> None:
        pipe = (
            self._pipe
            if component is None
            else _ScopedPipeline(self._pipe, self.components[component])
        )
        self._run_with(pipe, name, state)

    def _run_with(self, pipe: Any, name: str, state: Any) -> None:
        block = self._blocks.sub_blocks[name]
        block(pipe, state)

    def _schedule_facts(self, pipe: Any, state: Any) -> ScheduleFacts:
        video_sigmas = pipe.scheduler.sigmas
        audio_sigmas = pipe.audio_scheduler.sigmas
        expected = canonical_timestep_plan(self.task)
        if (
            video_sigmas is None
            or audio_sigmas is None
            or _float_tuple(video_sigmas) != expected.video_sigmas
            or _float_tuple(audio_sigmas) != expected.audio_sigmas
            or _float_tuple(state.timesteps) != expected.video_timesteps
            or _float_tuple(state.audio_timesteps) != expected.audio_timesteps
        ):
            raise ConformanceError(
                "official H3 scheduler did not produce the canonical 30-sigma/29-forward plan",
                code="canonical_schedule",
            )
        _validate_row_timestep_plan(state, expected)
        return ScheduleFacts(
            timestep_plan_digest=expected.digest,
            video_sigma_digest=expected.video_sigma_digest,
            audio_sigma_digest=expected.audio_sigma_digest,
            video_timestep_digest=expected.video_timestep_digest,
            audio_timestep_digest=expected.audio_timestep_digest,
        )


def build_fl2va_pipeline(config: Config) -> OfficialH3Pipeline:
    return OfficialH3Pipeline(config, task="fl2va")


def build_ref2va_pipeline(config: Config) -> OfficialH3Pipeline:
    return OfficialH3Pipeline(config, task="ref2va")


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


def _apply_transformer_dtype(transformer: Any) -> Any:
    """Reproduce Diffusers' mixed FULL compute policy on Runtime destinations."""
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="There are modules in MiniMaxH3Transformer3DModel.*"
        )
        transformer.to(dtype=torch.bfloat16)
    for name in transformer._keep_in_fp32_modules:
        getattr(transformer, name).to(dtype=torch.float32)
    return transformer.eval()


def _validate_row_timestep_plan(state: Any, expected: TimestepPlan) -> None:
    """Prove the live packed rows consume exactly the classes named by the receipt."""
    import torch

    plans = state.row_timestep_plan
    if len(plans) != TRANSFORMER_EVALUATIONS:
        raise ConformanceError(
            "official H3 row plan has the wrong length", code="canonical_schedule"
        )
    tags = state.token_tags.detach().long().cpu()
    video_indices = state.video_indices.detach().long().cpu()
    audio_indices = state.audio_indices.detach().long().cpu()
    text_indices = state.text_indices.detach().long().cpu()
    expected_tags = torch.full_like(tags, -1)
    expected_tags[video_indices] = 0
    expected_tags[audio_indices] = 2
    text_tags = state.text_token_tags.detach().long().cpu()
    if text_tags.shape != text_indices.shape or bool(((text_tags != 0) & (text_tags != 1)).any()):
        raise ConformanceError(
            "official H3 presentation tags are not the canonical vision/text 0/1 convention",
            code="canonical_schedule",
        )
    expected_tags[text_indices] = text_tags
    if not torch.equal(tags, expected_tags):
        raise ConformanceError(
            "official H3 packed-row modality tags differ from the canonical 0/1/2 convention",
            code="canonical_schedule",
        )
    for index, (unique, inverse) in enumerate(plans):
        row_timesteps = unique.detach().float().cpu().index_select(0, inverse.detach().long().cpu())
        want = torch.full_like(row_timesteps, expected.video_timesteps[index])
        want[video_indices[: state.num_condition_video_rows]] = max(
            expected.video_timesteps[index], _as_float32(0.999)
        )
        want[audio_indices[state.num_condition_audio_rows :]] = expected.audio_timesteps[index]
        want[audio_indices[: state.num_condition_audio_rows]] = _as_float32(1.0)
        if not torch.equal(row_timesteps, want):
            raise ConformanceError(
                f"official H3 packed-row timestep assignment differs at evaluation {index}",
                code="canonical_schedule",
            )


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
    from transformers import (
        AddedToken,
        Qwen2Tokenizer,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocab = _json_mapping(_ASSETS / "tokenizer" / "vocab.json")
    config = _json_mapping(_ASSETS / "tokenizer" / "tokenizer_config.json")
    config.pop("tokenizer_class", None)
    added_tokens = config.get("added_tokens_decoder")
    if not isinstance(added_tokens, Mapping) or any(
        not isinstance(value, Mapping) for value in added_tokens.values()
    ):
        raise ConformanceError(
            "bundled tokenizer added-token table is missing or malformed",
            code="asset_config",
        )
    try:
        config["added_tokens_decoder"] = {
            int(index): AddedToken(**dict(value)) for index, value in added_tokens.items()
        }
    except (TypeError, ValueError) as exc:
        raise ConformanceError(
            "bundled tokenizer added-token table is malformed", code="asset_config"
        ) from exc
    merge_lines = (_ASSETS / "tokenizer" / "merges.txt").read_text().splitlines()
    merges = [tuple(line.split(" ")) for line in merge_lines]
    if not merges or any(len(pair) != 2 for pair in merges):
        raise ConformanceError(
            "bundled tokenizer merges are empty or malformed", code="asset_config"
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
            code="asset_config",
        ) from exc
    if not isinstance(value, dict):
        raise ConformanceError(
            f"bundled processor asset {path.name!r} is not a JSON object",
            code="asset_config",
        )
    return value


def _audio_tensor(audio: DecodedAudio) -> Any:
    import torch

    channels = [
        torch.frombuffer(bytearray(channel), dtype=torch.float32) for channel in audio.pcm_f32le
    ]
    return torch.stack(channels)


def _aligned_soundtrack(video: DecodedVideo) -> Any | None:
    """Put embedded PCM on the video's origin without rounding either native clock."""
    import torch

    audio = video.soundtrack
    if audio is None:
        return None
    exact_offset = (audio.start_time - video.start_time) * audio.sample_rate
    exact_samples = video.duration * audio.sample_rate
    if exact_offset.denominator != 1:
        raise ConformanceError(
            "reference soundtrack offset is not aligned to its sample clock",
            code="reference_av_clock",
        )
    if exact_samples.denominator != 1:
        raise ConformanceError(
            "reference video duration is not aligned to its soundtrack sample clock",
            code="reference_av_clock",
        )
    offset = exact_offset.numerator
    target_samples = exact_samples.numerator
    waveform = _audio_tensor(audio)
    start = max(offset, 0)
    end = min(offset + audio.sample_count, target_samples)
    if start >= end:
        raise ConformanceError(
            "reference soundtrack has no samples on its video's timeline",
            code="reference_av_clock",
        )
    aligned = torch.zeros((audio.channels, target_samples), dtype=torch.float32)
    source_start = start - offset
    aligned[:, start:end] = waveform[:, source_start : source_start + end - start]
    return aligned


def _video_at_24fps(video: DecodedVideo) -> Any:
    """Exact presentation boundaries onto the official whole-frame 24-fps clock."""
    import torch

    starts = video.frame_pts
    durations = video.frame_durations
    for index in range(len(starts) - 1):
        if starts[index] + durations[index] != starts[index + 1]:
            raise ConformanceError(
                f"reference video clock has a gap or overlap before frame {index + 1}",
                code="reference_clock",
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
            "reference video has no frame on the 24-fps clock", code="reference_clock"
        )

    frames = []
    for index in selected:
        raw = video.frames_rgb[index]
        frame = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        frames.append(frame.reshape(video.height, video.width, 3).permute(2, 0, 1))
    return torch.stack(frames)


def _round_fraction(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _shifted_sigmas(shift: float) -> tuple[float, ...]:
    import torch

    base = torch.linspace(1.0, 0.0, SIGMA_GRID_POINTS, dtype=torch.float32)
    sigmas = shift * base / (1 + (shift - 1) * base)
    return _float_tuple(torch.unique_consecutive(sigmas))


def _timesteps(sigmas: Sequence[float]) -> tuple[float, ...]:
    import torch

    values = torch.tensor(sigmas, dtype=torch.float32)
    return _float_tuple(1.0 - values[:-1])


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
