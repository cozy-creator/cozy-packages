"""MiniMax-H3's two official serving actions over one dual-DiT construction.

Runtime decodes typed assets and encodes outputs. Official Diffusers owns H3 presentation,
conditioning, AdaLN, solver, and decode math. This module validates the product contract,
stages weighted roots, and joins those two boundaries.
"""

from __future__ import annotations

import hashlib
import queue
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from fractions import Fraction
from functools import partial
from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    ChildCallError,
    Context,
    DecodedAudio,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaHeader,
    DecodedVideo,
    DecodedVideoFormat,
    DecodedVideoFrame,
    Image,
    ImageAsset,
    ImageFrame,
    ImagePreparation,
    InvalidRequest,
    Loader,
    MediaDecoder,
    Mixed,
    Model,
    OutputError,
    Outputs,
    Preflight,
    SavedVideo,
    Telemetry,
    Tree,
    UnsupportedInput,
    VideoAsset,
    data_values,
    invocable,
    sequence_parallel,
    uses_components,
)
from msgspec.structs import replace

from assembly import MAX_SHOTS, AssembleVideoRequest, assemble, assemble_video
from gates import MediaFacts, refuse_before_encode, report_after_encode
from long_form_state import (
    MAX_PREFIX_BYTES,
    RenderProvenance,
    ShotIntent,
    StoredShot,
    check_intent,
    compatible,
    provenance,
    read_prefix,
    save_prefix,
)
from official import (
    FPS,
    MAX_AUDIO_REFERENCES,
    MAX_CONDITIONER_VISION_TOKENS,
    MAX_IMAGE_REFERENCES,
    MAX_REFERENCES,
    MAX_VIDEO_REFERENCES,
    REFERENCE_IMAGE_SHORT_EDGE,
    NumericalChecks,
    OfficialH3Pipeline,
    OfficialH3TurboLoRA,
    ReferencePolicyFacts,
    ScheduleFacts,
    Task,
    assert_duration_envelope,
    build_h3_pipeline,
    build_h3_turbo_base,
    build_h3_turbo_lora,
    denoise_rows,
    frames_for,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    turbo_steps,
    validate_reference_policy,
)
from turbo import ATTENTION_KWARG, OVERLAY_KWARG, TURBO_BANK

app = App()

_MIB = 1 << 20
_MIN_REFERENCE_DURATION = Fraction(2, 1)
_MAX_REFERENCE_DURATION = Fraction(15, 1)


ReferenceAssets = Annotated[
    Assets[Mixed],
    AssetLimits(
        images=MAX_IMAGE_REFERENCES,
        videos=MAX_VIDEO_REFERENCES,
        audio=MAX_AUDIO_REFERENCES,
        total=MAX_REFERENCES,
    ),
    ImagePreparation(max_edge=8192, max_pixels=16_777_216),
    msgspec.Meta(min_length=1),
]
KeyframeAssets = Annotated[Assets[Image], AssetLimits(images=2)]
DEFAULT_REFERENCE_IMAGE_SHORT_EDGE = 1024
_SHORT_EDGE_LADDER = (2048, 1536, 1024, 768, 512, 256)
_REFERENCE_FIDELITY_EDGES = {"low": 256, "medium": 1024, "high": REFERENCE_IMAGE_SHORT_EDGE}
Prompt = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
# The shipped plans own the enum and must agree for both inference tasks.
SUPPORTED_STEPS = data_values(
    __file__, "timestep-plans/fl2va.json", "schedules", "transformer_evaluations"
)
if (
    data_values(__file__, "timestep-plans/ref2va.json", "schedules", "transformer_evaluations")
    != SUPPORTED_STEPS
):
    raise ValueError("H3 task plans declare different supported step counts")
DEFAULT_STEPS = 30  # Static and imported descriptors must expose the same default.
#: PDD-8: what the turbo functions run, fixed by their plans and absent from their wire.
TURBO_STEPS = turbo_steps()
Steps = Annotated[
    Literal[SUPPORTED_STEPS],  # type: ignore[valid-type]
    msgspec.Meta(description="Denoise steps (transformer evaluations); fewer is faster."),
]
# Length is the request's largest cost lever: the DiT attends over ONE packed sequence whose
# rows scale with the frame count, and attention is quadratic in it. Every whole second in
# the envelope is served, and the SHORTEST is the default — a caller that says nothing pays
# the cheapest clip, not the longest (se-047).
# Declared, because the wire IS the declaration: `describe` reads this file and runs none of
# it (#713). `assert_duration_envelope` refuses unless the official 17n+5 snap and the frame
# envelope serve exactly these whole seconds, so the pair below cannot drift from the
# geometry it advertises. The ceiling is 15: 362 frames is the top of the `17n + 5` grid a
# 15-second model reaches, and delivering it needs the scoped ceiling in `official.py`
# (se-053), because upstream states its own bound in seconds and the grid has no 360.
MIN_DURATION_S = 5
MAX_DURATION_S = 15
assert_duration_envelope((MIN_DURATION_S, MAX_DURATION_S))
DEFAULT_DURATION_S = MIN_DURATION_S
DurationSeconds = Annotated[
    int,
    msgspec.Meta(
        ge=MIN_DURATION_S,
        le=MAX_DURATION_S,
        description=(
            "Clip length in whole seconds, snapped up to the video VAE's own 17n+5 frame "
            "grid; shorter is quadratically faster."
        ),
    ),
]


class FirstLastFrameToVideoInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    seed: int | None = None
    steps: Steps = DEFAULT_STEPS
    duration_s: DurationSeconds = DEFAULT_DURATION_S


class ReferenceMediaToVideoInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    seed: int | None = None
    steps: Steps = DEFAULT_STEPS
    duration_s: DurationSeconds = DEFAULT_DURATION_S


# The turbo functions carry no `steps`: PDD-8 fixes eight transformer evaluations, so the
# parameter is unrepresentable on the wire rather than refused at runtime.
class FirstLastFrameToVideoTurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    seed: int | None = None
    duration_s: DurationSeconds = DEFAULT_DURATION_S


class ReferenceMediaToVideoTurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    seed: int | None = None
    duration_s: DurationSeconds = DEFAULT_DURATION_S


class H3VideoOutput(msgspec.Struct):
    """The catalog wire shape. Checkpoint, plan, geometry, and digest facts remain
    attempt observations (se-012): they ride Telemetry, never the customer result."""

    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    continuation_frame: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    warnings: list[str]


def preflight_reference_media(
    payload: ReferenceMediaToVideoInput, assets: ReferenceAssets
) -> ReferencePolicyFacts:
    """Refuse cross-field count errors before Runtime hydrates a single asset."""
    del payload
    return _reference_policy(assets)


def preflight_reference_media_turbo(
    payload: ReferenceMediaToVideoTurboInput, assets: ReferenceAssets
) -> ReferencePolicyFacts:
    del payload
    return _reference_policy(assets)


def _reference_policy(assets: ReferenceAssets) -> ReferencePolicyFacts:
    kinds = [assets.info(index).kind for index in range(len(assets))]
    try:
        return validate_reference_policy(kinds)
    except ValueError as exc:
        raise UnsupportedInput(str(exc), code="reference_policy", fields=["assets"]) from exc


@sequence_parallel(degrees=(2, 4))
class H3Model(Model[OfficialH3Pipeline], encoded_leaves="accept", fusion="accept"):
    """H3 is head-shardable at 2 and 4.

    The DiT declares 56 attention heads, which divide by 2, 4 and 8, and its upstream
    `_cp_plan` shards the packed sequence itself -- every block's GEMMs, norms, RoPE and
    SwiGLU run on S/K rows, not just attention. `EncodedLinear` quantizes per TOKEN with
    rowwise `_scaled_mm` scales, so a row's numbers do not depend on which rank holds it;
    a per-tensor activation scale would be derived from the local shard and is refused by
    the runtime rather than served.

    2 and 4 are declared because 2 and 4 are what has been RUN: degree 2 and degree 4
    reproduce the degree-1 video and audio velocities to 1.2e-7 and 2.4e-7 max absolute
    error on this pipeline's own AdaLN-pruned DiT (2026-09-08), which is float32 round-off.
    8 divides the heads too and stays undeclared until an 8-wide arm exists.
    """

    pipe: OfficialH3Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3Pipeline, factory=build_h3_pipeline)

    def unload(self, loader: Loader) -> None:
        return None

    def warm(self, ctx: Context) -> None:
        """One dry DiT forward per entrypoint DiT this device ADMITS, before serving.

        Both DiTs are offered, because one construction carries both entrypoints and a
        switch between them must not pay a first call either (h3a-018). The runtime has
        already applied the fused glue and loaded its cubins for this device by the time
        `warm` runs (h3a-015, `fusion="accept"` above); the dry forward is what pays their
        first launches, the rotary tables and the projections' first GEMM plans.

        Offered, not required. A fill that could not hold both DiTs PARKED one, and `warm`
        runs before any attempt fixes a placement rung, so admission there evicts nothing
        and staging the parked DiT is a measured `device_shortfall`. That is a capacity
        fact about the card, not a construction failure: the parked DiT is staged by the
        ladder on its entrypoint's first request, under a rung that may evict, and pays its
        first launches there. So a shortfall on one entrypoint's DiT is recorded and
        skipped while the other is still warmed — consent, not requirement, the shape
        h3a-015 gave the fused lane itself.
        """
        warmers: dict[Task, Callable[[], None]] = {
            "fl2va": self.warm_fl2va,
            "ref2va": self.warm_ref2va,
        }
        for warm_one in warmers.values():
            ctx.raise_if_cancelled()
            try:
                warm_one()
            except Exception as exc:
                if getattr(exc, "code", "") != "device_shortfall":
                    raise
                print(
                    f"[minimax-h3] {warm_one.__name__} not applied: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    @uses_components("fl2va_dit")
    def warm_fl2va(self) -> None:
        self.pipe.warm_dit("fl2va")

    @uses_components("ref2va_dit")
    def warm_ref2va(self) -> None:
        self.pipe.warm_dit("ref2va")

    @uses_components("text_encoder")
    def condition_text(self, task: Task, state: Any, *, checks: NumericalChecks) -> None:
        checks.component("text_encoder", self.pipe.components["text_encoder"])
        self.pipe.condition_text(task, state, checks=checks)

    @uses_components("audio_vae")
    def decode_audio(
        self, task: Task, state: Any, *, checks: NumericalChecks | None = None
    ) -> tuple[Any, int]:
        if checks is not None:
            checks.component("audio_vae", self.pipe.components["audio_vae"])
        audio = self.pipe.decode_audio(task, state)
        if checks is not None:
            checks.tensors("decode_audio", [("audio", audio)])
        return audio, int(state.sampling_rate)

    @uses_components("video_vae")
    def decode_video(
        self,
        task: Task,
        state: Any,
        *,
        on_chunk: Callable[[Any], None],
        checks: NumericalChecks | None = None,
    ) -> int:
        """Hand every decoded temporal chunk to `on_chunk` inside the VAE's component scope;
        a generator would run its body after the scope had closed."""
        if checks is not None:
            checks.component("video_vae", self.pipe.components["video_vae"])

        def observed(chunk: Any) -> None:
            if checks is not None:
                checks.tensors("decode_video", [("video", chunk)])
            on_chunk(chunk)

        return self.pipe.decode_video_chunks(task, state, observed)

    @uses_components("video_vae")
    def condition_fl2va_media(self, task: Task, state: Any, *, checks: NumericalChecks) -> None:
        checks.component("video_vae", self.pipe.components["video_vae"])
        self.pipe.condition_media(task, state, checks=checks)

    @uses_components("video_vae", "audio_vae")
    def condition_ref2va_media(self, task: Task, state: Any, *, checks: NumericalChecks) -> None:
        for name in ("video_vae", "audio_vae"):
            checks.component(name, self.pipe.components[name])
        self.pipe.condition_media(task, state, checks=checks)

    @uses_components("fl2va_dit")
    def sample_fl2va(
        self, state: Any, *, on_step: Any, cancel: Any, checks: NumericalChecks
    ) -> ScheduleFacts:
        root = self.pipe.components["fl2va_dit"]
        checks.component("fl2va_dit", root)
        with checks.forwards(root, "fl2va_dit"):
            return self.pipe.denoise("fl2va", state, on_step=on_step, cancel=cancel, checks=checks)

    @uses_components("ref2va_dit")
    def sample_ref2va(
        self, state: Any, *, on_step: Any, cancel: Any, checks: NumericalChecks
    ) -> ScheduleFacts:
        root = self.pipe.components["ref2va_dit"]
        checks.component("ref2va_dit", root)
        with checks.forwards(root, "ref2va_dit"):
            return self.pipe.denoise("ref2va", state, on_step=on_step, cancel=cancel, checks=checks)


@sequence_parallel(degrees=(2, 4))
class H3TurboBase(H3Model, encoded_leaves="accept", fusion="accept"):
    """The five-root base checkpoint with construction-time turbo consumers."""

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3Pipeline, factory=build_h3_turbo_base)

    @uses_components("fl2va_dit")
    def sample_fl2va_turbo(
        self,
        state: Any,
        *,
        turbo_lora: H3TurboLoRA,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
    ) -> ScheduleFacts:
        return turbo_lora.sample_fl2va(self, state, on_step=on_step, cancel=cancel, checks=checks)

    @uses_components("ref2va_dit")
    def sample_ref2va_turbo(
        self,
        state: Any,
        *,
        turbo_lora: H3TurboLoRA,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
    ) -> ScheduleFacts:
        return turbo_lora.sample_ref2va(self, state, on_step=on_step, cancel=cancel, checks=checks)


@sequence_parallel(degrees=(2, 4))
class H3TurboLoRA(Model[OfficialH3TurboLoRA], encoded_leaves="accept"):
    """Independent PDD weights, replicated alongside the base on every CP rank."""

    pipe: OfficialH3TurboLoRA

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3TurboLoRA, factory=build_h3_turbo_lora)

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("fl2va_turbo")
    def sample_fl2va(
        self, base: H3Model, state: Any, *, on_step: Any, cancel: Any, checks: NumericalChecks
    ) -> ScheduleFacts:
        return self._sample(base, "fl2va", state, on_step=on_step, cancel=cancel, checks=checks)

    @uses_components("ref2va_turbo")
    def sample_ref2va(
        self, base: H3Model, state: Any, *, on_step: Any, cancel: Any, checks: NumericalChecks
    ) -> ScheduleFacts:
        return self._sample(base, "ref2va", state, on_step=on_step, cancel=cancel, checks=checks)

    def _sample(
        self,
        base: H3Model,
        trunk: Literal["fl2va", "ref2va"],
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
    ) -> ScheduleFacts:
        task: Task = "fl2va_turbo" if trunk == "fl2va" else "ref2va_turbo"
        overlay = self.pipe.overlay(base.pipe, trunk)
        root = base.pipe.components[f"{trunk}_dit"]
        checks.component(f"{trunk}_dit", root)
        checks.component(f"{trunk}_turbo", overlay)
        original = state.get("attention_kwargs")
        state.set(
            "attention_kwargs",
            {
                **(original or {}),
                ATTENTION_KWARG: TURBO_BANK,
                OVERLAY_KWARG: overlay,
            },
        )
        try:
            with checks.forwards(root, f"{trunk}_dit"):
                return base.pipe.denoise(task, state, on_step=on_step, cancel=cancel, checks=checks)
        finally:
            state.set("attention_kwargs", original)


def _keyframe_roles(assets: KeyframeAssets) -> tuple[int | None, int | None]:
    """Explicit first/last labels choose roles; other images fill remaining roles in order."""
    named = {
        assets.info(index).label: index
        for index in range(len(assets))
        if assets.info(index).label in {"first", "last"}
    }
    remaining = iter(
        index for index in range(len(assets)) if assets.info(index).label not in {"first", "last"}
    )
    first = named["first"] if "first" in named else next(remaining, None)
    last = named["last"] if "last" in named else next(remaining, None)
    if next(remaining, None) is not None:
        raise InvalidRequest("FL2VA accepts at most two images", fields=["assets"])
    return first, last


def _keyframe_image(
    assets: KeyframeAssets,
    index: int | None,
    *,
    field: str,
) -> Image | None:
    if index is None:
        return None
    image = assets[index]
    _validate_ratio(image.width, image.height, field)
    return image


class ImageSizing(msgspec.Struct, frozen=True):
    """One image reference's fidelity: its explicit short edge, or the auto-sized default."""

    field: str
    width: int
    height: int
    requested: int | None
    resolved: int = 0

    @property
    def tokens(self) -> int:
        return reference_image_vision_tokens(self.width, self.height, self.resolved)


class ReferenceSizing(msgspec.Struct, frozen=True):
    images: tuple[ImageSizing, ...]
    video_tokens: int
    default: int

    @property
    def edges(self) -> list[int]:
        return [image.resolved for image in self.images]

    @property
    def total(self) -> int:
        return self.video_tokens + sum(image.tokens for image in self.images)

    @property
    def summary(self) -> str:
        """Per image: field, requested (or ``default``) -> resolved short edge, tokens."""
        return "; ".join(
            f"{image.field} {image.requested or 'default'}->{image.resolved} {image.tokens} tokens"
            for image in self.images
        )


def resolve_reference_sizing(
    images: Sequence[ImageSizing], *, default: int, video_tokens: int
) -> ReferenceSizing:
    """An explicit fidelity is never changed. The images without one share the package
    default and step down the ladder together until the total fits the budget; auto-sizing
    never rises above the default."""

    def sized(edge: int) -> ReferenceSizing:
        resolved = tuple(replace(image, resolved=image.requested or edge) for image in images)
        return ReferenceSizing(resolved, video_tokens, default)

    ladder = [default, *(rung for rung in _SHORT_EDGE_LADDER if rung < default)]
    for edge in ladder:
        sizing = sized(edge)
        if sizing.total <= MAX_CONDITIONER_VISION_TOKENS:
            return sizing
    floor = sized(ladder[-1])
    explicit = [image for image in floor.images if image.requested]
    raise UnsupportedInput(
        f"reference vision presentation needs {floor.total} tokens even with every default-sized "
        f"image at {ladder[-1]} px ({video_tokens} from videos, "
        f"{sum(image.tokens for image in explicit)} from the explicit short edges "
        f"{[image.requested for image in explicit]}); this release admits at most "
        f"{MAX_CONDITIONER_VISION_TOKENS}",
        code="reference_policy",
        fields=["assets"],
    )


def assets_to_h3_refs(
    references: ReferenceAssets,
    *,
    pipe: OfficialH3Pipeline,
) -> tuple[list[Any], ReferenceSizing]:
    prepared: list[Any] = []
    video_duration = Fraction(0)
    audio_duration = Fraction(0)
    video_tokens = 0
    images: list[ImageSizing] = []

    for index, reference in enumerate(references):
        info = references.info(index)
        field = info.id
        if isinstance(reference, Image):
            image = reference
            _validate_ratio(image.width, image.height, field)
            requested = (
                None if info.fidelity == "auto" else _REFERENCE_FIDELITY_EDGES[info.fidelity]
            )
            images.append(ImageSizing(field, image.width, image.height, requested))
            prepared.append(pipe.image_reference(image))
        elif isinstance(reference, DecodedVideo):
            video = reference
            _validate_video(video, field)
            video_duration += video.duration
            if video_duration > _MAX_REFERENCE_DURATION:
                raise InvalidRequest(
                    "reference videos total more than 15 seconds",
                    code="reference_policy",
                    fields=["assets"],
                )
            video_tokens += reference_video_vision_tokens(video.width, video.height, video.duration)
            prepared.append(pipe.video_reference(video))
        else:
            audio = reference
            _validate_audio(audio, field)
            audio_duration += audio.duration
            _validate_audio_aggregate(audio_duration)
            prepared.append(pipe.audio_reference(audio))
    return prepared, resolve_reference_sizing(
        images, default=DEFAULT_REFERENCE_IMAGE_SHORT_EDGE, video_tokens=video_tokens
    )


def _validate_ratio(width: int, height: int, field: str) -> None:
    if width > 4 * height or height > 4 * width:
        raise UnsupportedInput(
            f"{field} must have an aspect ratio between 1:4 and 4:1, got {width}x{height}",
            code="reference_policy",
            fields=[field],
        )


def _validate_audio(audio: DecodedAudio, field: str) -> None:
    _validate_audio_channels(audio, field)
    _validate_duration(audio.duration, field)


def _validate_video(video: DecodedVideo, field: str) -> None:
    _validate_ratio(video.width, video.height, field)
    _validate_duration(video.duration, field)
    if video.pixel_aspect_ratio != 1:
        raise UnsupportedInput(
            f"{field} must use square pixels, got {video.pixel_aspect_ratio}",
            code="reference_policy",
            fields=[field],
        )
    for index, duration in enumerate(video.frame_durations):
        if duration <= 0:
            raise InvalidRequest(
                f"{field} frame {index} has a non-positive presentation duration",
                code="reference_policy",
                fields=[field],
            )
        if (
            index
            and video.frame_pts[index - 1] + video.frame_durations[index - 1]
            != video.frame_pts[index]
        ):
            raise InvalidRequest(
                f"{field} has a presentation-clock gap or overlap before frame {index}",
                code="reference_policy",
                fields=[field],
            )
    if video.soundtrack is not None:
        _validate_audio_channels(video.soundtrack, field)
        offset_samples = (
            video.soundtrack.start_time - video.start_time
        ) * video.soundtrack.sample_rate
        target_samples = video.duration * video.soundtrack.sample_rate
        if offset_samples.denominator != 1 or target_samples.denominator != 1:
            raise InvalidRequest(
                f"{field} video and soundtrack clocks do not meet on exact samples",
                code="reference_policy",
                fields=[field],
            )
        offset = video.soundtrack.start_time - video.start_time
        if offset >= video.duration or offset + video.soundtrack.duration <= 0:
            raise InvalidRequest(
                f"{field} soundtrack does not overlap its video timeline",
                code="reference_policy",
                fields=[field],
            )


def _validate_audio_aggregate(duration: Fraction) -> None:
    """The standalone-audio modality cap. A video's soundtrack belongs to that video and
    rides the video aggregate; it never consumes this budget."""
    if duration > _MAX_REFERENCE_DURATION:
        raise InvalidRequest(
            "standalone audio references total more than 15 seconds",
            code="reference_policy",
            fields=["assets"],
        )


def _validate_audio_channels(audio: DecodedAudio, field: str) -> None:
    if audio.channels not in (1, 2):
        raise UnsupportedInput(
            f"{field} audio must be mono or stereo, got {audio.channels} channels",
            code="reference_policy",
            fields=[field],
        )


def _validate_duration(duration: Fraction, field: str) -> None:
    if not _MIN_REFERENCE_DURATION <= duration <= _MAX_REFERENCE_DURATION:
        raise InvalidRequest(
            f"{field} must be between 2 and 15 seconds, got {float(duration):.3f}",
            code="reference_policy",
            fields=[field],
        )


def _finish(
    model: H3Model,
    task: Task,
    state: Any,
    schedule: ScheduleFacts,
    *,
    duration_s: int,
    out: Outputs,
    tel: Telemetry,
    cancel: Any,
    checks: NumericalChecks | None = None,
) -> H3VideoOutput:

    cancel()
    with tel.stage("decode_audio", overall_range=(0.85, 0.90)):
        audio, sample_rate = model.decode_audio(task, state, checks=checks)
        release_rate = model.pipe.sample_rate
        if audio.ndim != 3 or int(audio.shape[0]) != 1 or sample_rate != release_rate:
            raise OutputError(
                f"official H3 audio decode returned shape {tuple(audio.shape)} at {sample_rate}Hz; "
                f"the release clock is {release_rate}Hz",
                code="output_integrity",
            )
        waveform = audio[0].to(torch.float32).contiguous().cpu()
    cancel()
    frames = frames_for(duration_s)
    clock = MediaFacts(frames=frames, fps=FPS, sample_rate=sample_rate)
    refuse_before_encode(
        waveform=waveform,
        audio_nonfinite_fraction=_nonfinite_fraction(torch, waveform),
        requested=clock,
        tel=tel,
    )

    # The encoder runs under the decode (h3a-017): each chunk is quantized on its device
    # and landed in the host buffer while a worker feeds the runtime's streaming MP4 sink.
    with tel.stage("decode_video", overall_range=(0.90, 0.98)):
        stream = _VideoStream(
            torch,
            out,
            frames=frames,
            waveform=waveform,
            sample_rate=sample_rate,
            cancel=cancel,
            progress=tel.step_callback(frames, stage="decode_video", overall_range=(0.90, 0.98)),
        )
        try:
            model.decode_video(task, state, on_chunk=stream.push, checks=checks)
            saved, video_pixel_digest = stream.finish()
        except BaseException:
            stream.abandon()
            raise
    tel.metric("video_nonfinite_fraction", 0.0)
    pixels = stream.pixels
    _, height, width, _ = (int(value) for value in pixels.shape)

    with tel.stage("check_output", overall_range=(0.98, 0.99)):
        warnings = report_after_encode(
            torch, pixels=pixels, waveform=waveform, requested=clock, tel=tel
        )

    cancel()
    frame_bytes = bytes(pixels[-1].numpy())
    with tel.stage("encode_outputs", overall_range=(0.99, 1.00)):
        continuation = out.save_image(ImageFrame(width, height, frame_bytes), format="png")

    tel.log(
        "h3 output geometry",
        width=width,
        height=height,
        frames=frames,
        fps=FPS,
        requested_duration_s=duration_s,
        duration_seconds=round(frames / FPS, 3),
        denoise_rows=denoise_rows(frames, height, width),
        sample_rate=sample_rate,
    )
    tel.log(
        "h3 schedule facts",
        timestep_plan_digest=schedule.timestep_plan_digest,
        video_sigma_digest=schedule.video_sigma_digest,
        audio_sigma_digest=schedule.audio_sigma_digest,
        video_timestep_digest=schedule.video_timestep_digest,
        audio_timestep_digest=schedule.audio_timestep_digest,
        sigma_grid_points=schedule.sigma_grid_points,
        transformer_evaluations=schedule.transformer_evaluations,
    )
    tel.log(
        "h3 source digests",
        video_pixel_digest=video_pixel_digest,
        audio_sample_digest=hashlib.sha256(waveform.numpy()).hexdigest(),
        continuation_pixel_digest=hashlib.sha256(frame_bytes).hexdigest(),
    )
    tel.log(
        "h3 container facts",
        video_codec=saved.video_codec,
        frame_count=saved.frame_count,
        frame_rate=str(saved.frame_rate),
        color_matrix=saved.color_matrix,
        color_range=saved.color_range,
        audio_codec=saved.audio.codec if saved.audio is not None else "",
        audio_decoded_samples=saved.audio.decoded_samples if saved.audio is not None else 0,
        video_bytes=saved.video.size_bytes,
    )
    return H3VideoOutput(video=saved.video, continuation_frame=continuation, warnings=warnings)


#: Frames per handoff slice: 8 frames of 1344x768 are 100 MB of fp32 source and 25 MB of
#: RGB8, so a slice's device temporaries stay small and its hash and encode hide under the
#: decode of the next chunk.
_RGB8_CHUNK_FRAMES = 8
#: Generated frames are square-pixel RGB at 24 fps, tagged BT.709 limited-range in the
#: container — the runtime's own HD rule, so a player converts them as the encoder did.
_VIDEO_COLOR = {"color_primaries": 1, "color_transfer": 1, "color_matrix": 1, "color_range": 1}
_AUDIO_LAYOUTS = {1: ("mono", ("FC",)), 2: ("stereo", ("FL", "FR"))}


def _land(torch: Any, chunk: Any, landed: Any) -> None:
    """Quantize one `(t, 3, H, W)` float slice on its device, straight into `landed`, its
    `(t, H, W, 3)` slice of the host buffer — no per-slice host temporary."""
    chunk.clamp_(0, 1).mul_(255).round_()
    landed.copy_(chunk.to(torch.uint8).permute(0, 2, 3, 1).contiguous())


def _rgb8(torch: Any, decoded: Any) -> tuple[Any, str]:
    """A whole `(1, T, 3, H, W)` float decode to ONE host RGB8 buffer and its digest: the
    slice-for-slice reference the streamed handoff below is held to."""
    source = decoded[0]
    frames, _, height, width = (int(value) for value in source.shape)
    pixels = torch.empty((frames, height, width, 3), dtype=torch.uint8, device="cpu")
    digest = hashlib.sha256()
    for start in range(0, frames, _RGB8_CHUNK_FRAMES):
        landed = pixels[start : start + _RGB8_CHUNK_FRAMES]
        _land(torch, source[start : start + len(landed)], landed)
        digest.update(landed.numpy())
    return pixels, digest.hexdigest()


class _VideoStream:
    """The decode-to-encode handoff (h3a-017).

    The decode thread refuses non-finite values, quantizes each chunk on its device and
    lands it in its slice of ONE host RGB8 buffer; a worker thread digests the slices in
    order and feeds the runtime's streaming MP4 sink while the next chunk decodes. The
    running sha256 is byte-for-byte the finished buffer's — the `h3 source digests`
    identity — while the container's bytes are a codec fact, not an output identity.
    The soundtrack, decoded first, is interleaved behind the frames it covers.
    """

    def __init__(
        self,
        torch: Any,
        out: Outputs,
        *,
        frames: int,
        waveform: Any,
        sample_rate: int,
        cancel: Any,
        progress: Callable[[int], None],
    ) -> None:
        self._torch = torch
        self._out = out
        self._frames = frames
        self._waveform = waveform
        self._sample_rate = sample_rate
        self._cancel = cancel
        self._progress = progress
        self._slices: queue.SimpleQueue[tuple[int, int] | None] = queue.SimpleQueue()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._encode: Future[SavedVideo] | None = None
        self._digest = hashlib.sha256()
        self._abandoned = False
        self.pixels: Any = None
        self.landed = 0

    def push(self, chunk: Any) -> None:
        """One `(t, 3, H, W)` float chunk on its device, in decode order."""
        self._cancel()
        if self._encode is not None and self._encode.done():
            self._encode.result()
        torch = self._torch
        frames, channels, height, width = (int(value) for value in chunk.shape)
        bad = _nonfinite_fraction(torch, chunk)
        if bad:
            raise OutputError(
                f"the official H3 video decode produced non-finite values ({bad:.6f} of frames "
                f"{self.landed}..{self.landed + frames})",
                code="output_integrity",
            )
        if self.pixels is None and channels == 3:
            self.pixels = torch.empty(
                (self._frames, height, width, 3), dtype=torch.uint8, device="cpu"
            )
            self._encode = self._pool.submit(self._out.save_video_stream, self._events())
        if (
            self.pixels is None
            or channels != 3
            or (height, width) != tuple(self.pixels.shape[1:3])
            or self.landed + frames > self._frames
        ):
            raise OutputError(
                f"official H3 decode produced a {channels}x{height}x{width} chunk of {frames} "
                f"frames at frame {self.landed}; the request is {self._frames} frames of "
                f"3x{height}x{width}",
                code="output_integrity",
            )
        for start in range(0, frames, _RGB8_CHUNK_FRAMES):
            stop = self.landed + min(_RGB8_CHUNK_FRAMES, frames - start)
            landed = self.pixels[self.landed : stop]
            _land(torch, chunk[start : start + len(landed)], landed)
            self._slices.put((self.landed, self.landed + len(landed)))
            self.landed += len(landed)
        self._progress(self.landed - 1)

    def finish(self) -> tuple[SavedVideo, str]:
        """Close the stream once every requested frame has landed; the sink's probed asset
        and the buffer's digest."""
        if self.landed != self._frames or self._encode is None:
            raise OutputError(
                f"official H3 decode returned {self.landed} frames, expected {self._frames}",
                code="output_integrity",
            )
        self._slices.put(None)
        try:
            return self._encode.result(), self._digest.hexdigest()
        finally:
            self._pool.shutdown(wait=False)

    def abandon(self) -> None:
        """A failed decode ends the stream so the sink aborts and unlinks its partial file."""
        self._abandoned = True
        self._slices.put(None)
        if self._encode is not None:
            with suppress(Exception):
                self._encode.result()
        self._pool.shutdown(wait=False)

    def _events(self) -> Iterator[DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk]:
        _, height, width, _ = (int(value) for value in self.pixels.shape)
        channels, samples = (int(value) for value in self._waveform.shape)
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=width,
                height=height,
                time_base=Fraction(1, FPS),
                pixel_aspect_ratio=Fraction(1),
                nominal_frame_rate=Fraction(FPS),
                **_VIDEO_COLOR,
            ),
            audio=DecodedAudioFormat(
                channels=channels,
                sample_rate=self._sample_rate,
                channel_layout=_AUDIO_LAYOUTS[channels][0],
                channel_names=_AUDIO_LAYOUTS[channels][1],
                time_base=Fraction(1, self._sample_rate),
            ),
        )
        submitted = 0
        while (item := self._slices.get()) is not None:
            start, stop = item
            landed = self.pixels[start:stop]
            self._digest.update(landed.numpy())
            for index in range(start, stop):
                self._check_encoding()
                yield DecodedVideoFrame(
                    width=width,
                    height=height,
                    rgb=bytes(landed[index - start].numpy()),
                    pts=index,
                    duration=1,
                    time_base=Fraction(1, FPS),
                    pixel_aspect_ratio=Fraction(1),
                    **_VIDEO_COLOR,
                )
            covered = min(samples, stop * self._sample_rate // FPS)
            yield from self._audio(submitted, covered)
            submitted = covered
        self._check_encoding()
        yield from self._audio(submitted, samples)

    def _check_encoding(self) -> None:
        if self._abandoned:
            raise OutputError("the decode abandoned the video stream", code="output_integrity")
        self._cancel()

    def _audio(self, start: int, stop: int) -> Iterator[DecodedAudioChunk]:
        if stop <= start:
            return
        channels = int(self._waveform.shape[0])
        yield DecodedAudioChunk(
            channels=channels,
            sample_count=stop - start,
            sample_rate=self._sample_rate,
            channel_layout=_AUDIO_LAYOUTS[channels][0],
            channel_names=_AUDIO_LAYOUTS[channels][1],
            pcm_f32le=tuple(
                self._waveform[channel, start:stop].numpy().tobytes() for channel in range(channels)
            ),
            pts=start,
            time_base=Fraction(1, self._sample_rate),
        )


def _nonfinite_fraction(torch: Any, value: Any) -> float:
    """Count non-finite values with at most roughly 32 MiB of temporary mask."""
    if value.ndim < 2:
        chunks = (value,)
    else:
        values_per_column = value.numel() // int(value.shape[1])
        columns = max(1, (32 * _MIB) // max(1, values_per_column))
        chunks = value.split(columns, dim=1)
    bad = sum(int(torch.count_nonzero(~torch.isfinite(chunk))) for chunk in chunks)
    return float(bad / int(value.numel()))


_DEFAULT_MODEL_LADDER = [
    {"gpu": "H100", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-pruned"},
    {"gpu": "B200", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-pruned"},
    {"gpu": "5090", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-pruned"},
]

_DEFAULT_TURBO_LORA_LADDER = [
    {"gpu": "*", "lane": "paul/minimax-h3-turbo-lora@1.0.0-audit.1/pdd8"},
]


@app.entrypoint(defaults={"model": _DEFAULT_MODEL_LADDER})
def fl2va(
    ctx: Context,
    payload: FirstLastFrameToVideoInput,
    assets: KeyframeAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    return _keyframes_to_video(ctx, "fl2va", payload, assets, model, out, tel, steps=payload.steps)


@app.entrypoint
def fl2va_turbo(
    ctx: Context,
    payload: FirstLastFrameToVideoTurboInput,
    assets: KeyframeAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    """`fl2va` under PDD-8: the same keyframes and prompt, eight transformer evaluations."""
    return _keyframes_to_video(
        ctx,
        "fl2va_turbo",
        payload,
        assets,
        base_model,
        out,
        tel,
        steps=TURBO_STEPS,
        turbo_lora=turbo_lora,
    )


@app.entrypoint(preflight=preflight_reference_media, defaults={"model": _DEFAULT_MODEL_LADDER})
def ref2va(
    ctx: Context,
    payload: ReferenceMediaToVideoInput,
    assets: ReferenceAssets,
    facts: Preflight[ReferencePolicyFacts],
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    del facts
    return _references_to_video(
        ctx, "ref2va", payload, assets, model, out, tel, steps=payload.steps
    )


@app.entrypoint(preflight=preflight_reference_media_turbo)
def ref2va_turbo(
    ctx: Context,
    payload: ReferenceMediaToVideoTurboInput,
    assets: ReferenceAssets,
    facts: Preflight[ReferencePolicyFacts],
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    """`ref2va` under PDD-8: the same references and prompt, eight transformer evaluations."""
    del facts
    return _references_to_video(
        ctx,
        "ref2va_turbo",
        payload,
        assets,
        base_model,
        out,
        tel,
        steps=TURBO_STEPS,
        turbo_lora=turbo_lora,
    )


def _keyframes_to_video(
    ctx: Context,
    task: Task,
    payload: FirstLastFrameToVideoInput | FirstLastFrameToVideoTurboInput,
    assets: KeyframeAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    steps: int,
    turbo_lora: H3TurboLoRA | None = None,
) -> H3VideoOutput:
    first_index, last_index = _keyframe_roles(assets)
    return _render_keyframes(
        ctx,
        task,
        model,
        out,
        tel,
        prompt=payload.prompt,
        seed=payload.seed,
        duration_s=payload.duration_s,
        steps=steps,
        first=_keyframe_image(assets, first_index, field="first"),
        last=_keyframe_image(assets, last_index, field="last"),
        turbo_lora=turbo_lora,
    )


def _render_keyframes(
    ctx: Context,
    task: Task,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    prompt: str,
    seed: int | None,
    duration_s: int,
    steps: int,
    first: Image | None,
    last: Image | None,
    turbo_lora: H3TurboLoRA | None = None,
) -> H3VideoOutput:
    """Shared FL2VA rendering for direct requests and chained asset handoffs."""
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=seed)
    checks = NumericalChecks(tel, model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        state = model.pipe.start_fl2va(
            prompt=prompt,
            first_frame=first,
            last_frame=last,
            generator=model.pipe.generator(view.generator),
            steps=steps,
            frames=frames_for(duration_s),
            task=task,
        )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        model.condition_text(task, state, checks=checks)
    if first is not None or last is not None:
        with tel.stage("condition_media", overall_range=(0.08, 0.15)):
            model.condition_fl2va_media(task, state, checks=checks)
    if turbo_lora is None:
        sample = model.sample_fl2va
    else:
        assert isinstance(model, H3TurboBase)
        sample = partial(model.sample_fl2va_turbo, turbo_lora=turbo_lora)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = sample(
            state,
            on_step=tel.step_callback(steps, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    return _finish(
        model,
        task,
        state,
        schedule,
        duration_s=duration_s,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        checks=checks,
    )


def _references_to_video(
    ctx: Context,
    task: Task,
    payload: ReferenceMediaToVideoInput | ReferenceMediaToVideoTurboInput,
    assets: ReferenceAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
    *,
    steps: int,
    turbo_lora: H3TurboLoRA | None = None,
) -> H3VideoOutput:
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        references, sizing = assets_to_h3_refs(assets, pipe=model.pipe)
        tel.log(
            "h3 reference sizing",
            images=sizing.summary,
            video_tokens=sizing.video_tokens,
            default=sizing.default,
            total=sizing.total,
            budget=MAX_CONDITIONER_VISION_TOKENS,
        )
        state = model.pipe.start_ref2va(
            prompt=payload.prompt,
            references=references,
            generator=model.pipe.generator(view.generator),
            steps=steps,
            frames=frames_for(payload.duration_s),
            reference_image_short_edges=sizing.edges,
            task=task,
        )
        for index, reference in enumerate(state.normalized_references):
            if reference.kind == "image":
                info = assets.info(index)
                width, height = reference.image.size
                tel.log(
                    "h3 reference resolution",
                    input_id=info.id,
                    label=info.label,
                    fidelity=info.fidelity,
                    width=width,
                    height=height,
                )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        model.condition_text(task, state, checks=checks)
    with tel.stage("condition_media", overall_range=(0.08, 0.15)):
        model.condition_ref2va_media(task, state, checks=checks)
    if turbo_lora is None:
        sample = model.sample_ref2va
    else:
        assert isinstance(model, H3TurboBase)
        sample = partial(model.sample_ref2va_turbo, turbo_lora=turbo_lora)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = sample(
            state,
            on_step=tel.step_callback(steps, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    return _finish(
        model,
        task,
        state,
        schedule,
        duration_s=payload.duration_s,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        checks=checks,
    )


# --- long-form composition -------------------------------------------------------------
# `long_form` is a CPU composition job. Runtime runs an ordinary serving child for each
# missing shot and retains its returned bytes. Explicit continuation reuses the native
# prefix after validating shot intents and renderer provenance; inference is not memoized.
# Only encoded asset handles cross the loop. Assembly decodes one bounded event at a time.

# A hand-off frame is exactly the generation canvas, which the conditioner already bounds
# at 16.7 M pixels; three bytes a pixel is its decoded ceiling.
_KEYFRAME_MAX_BYTES = 64 * _MIB
_KEYFRAME_MAX_DECODED_BYTES = 3 * 16_777_216


class Shot(msgspec.Struct, forbid_unknown_fields=True):
    """One segment's authored identity.

    `seed` is explicit and required: cl-021 forbids a hidden same-seed or seed+i policy, so
    the caller freezes every seed in the request or the chain is not reproducible.

    A shot defaults to the LONGEST served cell, not the package default, and it follows
    `MAX_DURATION_S` rather than naming a number — whatever the envelope serves is what a
    long-form shot takes. The two defaults answer different questions: se-047 makes a single
    clip default to the cheapest length so a caller who says nothing is not billed for the
    longest one, while a long-form piece has already committed to the spend and pays per
    SEAM. The longest cell halves the seam count against 5 s for ~1.8x the money, and
    conditioning rows are a fixed per-segment cost, so they amortise ~2.8x better across it.
    """

    prompt: Prompt
    seed: int
    duration_s: DurationSeconds = MAX_DURATION_S


class SegmentInput(msgspec.Struct, forbid_unknown_fields=True):
    """One shot, as an ordinary request.

    `first_frame` is the previous shot's `continuation_frame`, passed as a verified asset
    reference. An extending request also names the retained prefix's renderer provenance.
    """

    prompt: Prompt
    seed: int
    duration_s: DurationSeconds
    steps: Steps
    expected_provenance: RenderProvenance | None = None
    first_frame: Annotated[
        ImageAsset | None,
        AssetBound(
            media_types=("image/png",),
            max_bytes=_KEYFRAME_MAX_BYTES,
            max_decoded_bytes=_KEYFRAME_MAX_DECODED_BYTES,
        ),
    ] = None


class SegmentTurboInput(msgspec.Struct, forbid_unknown_fields=True):
    """One shot, as an ordinary request.

    `first_frame` is the previous shot's `continuation_frame`, passed as a verified asset
    reference. An extending request also names the retained prefix's renderer provenance.
    """

    prompt: Prompt
    seed: int
    duration_s: DurationSeconds
    expected_provenance: RenderProvenance | None = None
    first_frame: Annotated[
        ImageAsset | None,
        AssetBound(
            media_types=("image/png",),
            max_bytes=_KEYFRAME_MAX_BYTES,
            max_decoded_bytes=_KEYFRAME_MAX_DECODED_BYTES,
        ),
    ] = None


class SegmentOutput(msgspec.Struct):
    """`H3VideoOutput` without a default factory, which an invocable result may not carry."""

    video: Annotated[
        VideoAsset,
        AssetBound(
            max_bytes=MAX_PREFIX_BYTES,
            max_decoded_bytes=32 << 20,
            media_types=("video/mp4",),
        ),
    ]
    continuation_frame: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=_KEYFRAME_MAX_BYTES,
            max_decoded_bytes=_KEYFRAME_MAX_DECODED_BYTES,
            media_types=("image/png",),
        ),
    ]
    warnings: list[str]
    provenance: RenderProvenance


class SegmentReceipt(msgspec.Struct):
    """What a delivered shot contributes to the assembly and to a resume."""

    seed: int
    duration_s: int
    frames: int
    start_frame: int
    video_digest: str
    video_bytes: int
    continuation_frame_digest: str
    first_frame_digest: str
    child_request_id: str
    provenance: RenderProvenance


class LongFormInput(msgspec.Struct, forbid_unknown_fields=True):
    """A shot list. The identity and audio anchors are repeated verbatim in every segment."""

    shots: Annotated[list[Shot], msgspec.Meta(min_length=1, max_length=MAX_SHOTS)]
    subject_definitions: str = ""
    overall_soundscape: str = ""
    non_diegetic_music: str = ""
    mode: Literal["turbo", "standard"] = "turbo"
    steps: Steps | None = None
    resume_from: Annotated[Tree | None, AssetBound(max_bytes=MAX_PREFIX_BYTES)] = None
    opening_frame: Annotated[
        ImageAsset | None,
        AssetBound(
            media_types=("image/png",),
            max_bytes=_KEYFRAME_MAX_BYTES,
            max_decoded_bytes=_KEYFRAME_MAX_DECODED_BYTES,
        ),
    ] = None


class LongFormOutput(msgspec.Struct):
    """A playable full or partial delivery and exact native bytes for explicit continuation.

    A later-shot failure is a successful partial delivery with complete=False and failure
    facts. Pass prefix as resume_from in a new request to keep its shots without rendering
    them again. Cancellation and first-shot failure remain terminal failures/cancellation.
    """

    video: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]
    prefix: Annotated[Tree, AssetBound(max_bytes=MAX_PREFIX_BYTES)]
    complete: bool
    delivered: int
    reused: int
    segments: list[SegmentReceipt]
    requested: int
    delivered_frames: int
    fps: int
    failed_index: int
    failure_code: str
    failure_detail: str
    warnings: list[str]


_ANCHOR_BLOCKS = ("subject_definitions", "overall_soundscape", "non_diegetic_music")


def compose_shot_prompt(shot: Shot, payload: LongFormInput, *, index: int) -> str:
    """Upstream's own prompt schema, repeated verbatim in every segment.

    `subject_definitions` is the free textual identity anchor: it does not drift and it does
    not consume the reference budget. `overall_soundscape` / `non_diegetic_music` are the
    only cross-segment audio anchors that carry at all (h3a-024 §3.4, §4).
    """
    parts = [f"[Shot {index + 1}]", shot.prompt]
    for name in _ANCHOR_BLOCKS:
        value = getattr(payload, name).strip()
        if value:
            parts.append(f"{name}: {value}")
    composed = "\n\n".join(parts)
    if len(composed) > 4096:
        raise InvalidRequest(
            f"shot {index + 1}'s prompt and the repeated anchor blocks are {len(composed)} "
            "characters; H3 admits 4096. Shorten the anchors, which every shot repeats.",
            fields=["shots"],
        )
    return composed


def segment_clock(durations: Sequence[int]) -> tuple[list[int], int]:
    """Start frame of each shot and the delivered total, in exact integer frames.

    Exactly one replayed frame leaves the chain at every seam. This is integer arithmetic
    over frame counts and the caller turns it into seconds as `Fraction(frames, FPS)` —
    never a per-seam `round(sample_rate / fps)`, whose residue accumulated +8.33 ms per clip.
    """
    starts: list[int] = []
    total = 0
    for index, seconds in enumerate(durations):
        starts.append(total)
        total += frames_for(seconds) - (1 if index else 0)
    return starts, total


def _decoded_keyframe(decoder: MediaDecoder, asset: ImageAsset | None) -> Image | None:
    """Runtime decodes; the decoder's projection spells one union for every asset kind."""
    if asset is None:
        return None
    image = decoder.value(asset)
    if not isinstance(image, Image):
        raise InvalidRequest("first_frame is not an image", fields=["first_frame"])
    _validate_ratio(image.width, image.height, "first_frame")
    return image


@invocable(defaults={"model": _DEFAULT_MODEL_LADDER})
async def segment(
    ctx: Context,
    *,
    payload: SegmentInput,
    model: H3Model,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> SegmentOutput:
    """One shot of a chain, as an ordinary request.

    This is `fl2va` with the keyframe carried as an asset reference rather than a decoded
    upload, so a parent can bind it to the previous shot's `continuation_frame` by digest.
    It is an invocable serving entrypoint so Runtime constructs and loads H3Model.
    The CPU composition awaits one ordinary serving call per shot.
    """
    ctx.raise_if_cancelled()
    observed = provenance(model.checkpoint_ref)
    compatible(observed, payload.expected_provenance)
    shot = _render_keyframes(
        ctx,
        "fl2va",
        model,
        out,
        tel,
        prompt=payload.prompt,
        seed=payload.seed,
        duration_s=payload.duration_s,
        steps=payload.steps,
        first=_decoded_keyframe(decoder, payload.first_frame),
        last=None,
    )
    return SegmentOutput(shot.video, shot.continuation_frame, list(shot.warnings), observed)


@invocable(
    defaults={
        "base_model": _DEFAULT_MODEL_LADDER,
        "turbo_lora": _DEFAULT_TURBO_LORA_LADDER,
    }
)
async def segment_turbo(
    ctx: Context,
    *,
    payload: SegmentTurboInput,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> SegmentOutput:
    """One chained PDD-8 shot using independently bound base and adapter checkpoints."""
    ctx.raise_if_cancelled()
    observed = provenance(base_model.checkpoint_ref, turbo_lora.checkpoint_ref)
    compatible(observed, payload.expected_provenance)
    shot = _render_keyframes(
        ctx,
        "fl2va_turbo",
        base_model,
        out,
        tel,
        prompt=payload.prompt,
        seed=payload.seed,
        duration_s=payload.duration_s,
        steps=TURBO_STEPS,
        first=_decoded_keyframe(decoder, payload.first_frame),
        last=None,
        turbo_lora=turbo_lora,
    )
    return SegmentOutput(shot.video, shot.continuation_frame, list(shot.warnings), observed)


app.entrypoint(segment)
app.entrypoint(segment_turbo)


async def long_form(
    ctx: Context,
    payload: LongFormInput,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> LongFormOutput:
    """Render only missing shots, assemble the delivered prefix and retain its native bytes.

    Existing prefix intents must match the corresponding requested shots. The remaining
    prompts may change. Every new segment uses the prefix's recorded renderer/model cohort;
    completed shot records are preserved, never relabeled as new rendering.
    """
    ctx.raise_if_cancelled()
    if payload.mode == "turbo" and payload.steps is not None:
        raise InvalidRequest(
            "turbo fixes eight PDD evaluations; omit steps or select mode=standard",
            fields=["mode", "steps"],
        )
    steps = (
        TURBO_STEPS
        if payload.mode == "turbo"
        else (DEFAULT_STEPS if payload.steps is None else payload.steps)
    )
    prompts = [
        compose_shot_prompt(shot, payload, index=index) for index, shot in enumerate(payload.shots)
    ]
    records: list[StoredShot] = []
    videos: list[VideoAsset] = []
    frames: list[ImageAsset] = []
    warnings: list[str] = []
    frame = payload.opening_frame
    expected: RenderProvenance | None = None
    if payload.resume_from is not None:
        manifest, videos, frames = read_prefix(payload.resume_from, requested=len(payload.shots))
        records = list(manifest.shots)
        first_digest = records[0].intent.first_frame_digest
        if frame is not None and frame.digest != first_digest:
            raise InvalidRequest(
                "opening frame differs from the retained prefix", code="prefix_intent"
            )
        for index, record in enumerate(records):
            shot = payload.shots[index]
            incoming = first_digest if index == 0 else records[index - 1].continuation_frame_digest
            check_intent(
                record,
                ShotIntent(prompts[index], shot.seed, shot.duration_s, steps, incoming),
                frames_for(shot.duration_s),
            )
        expected = records[0].provenance
        frame = frames[-1]
    reused = len(records)
    failed_index, failure_code, failure_detail = -1, "", ""
    for index in range(reused, len(payload.shots)):
        ctx.raise_if_cancelled()
        shot = payload.shots[index]
        intent = ShotIntent(
            prompts[index],
            shot.seed,
            shot.duration_s,
            steps,
            "" if frame is None else frame.digest,
        )
        try:
            if payload.mode == "turbo":
                call = segment_turbo(  # type: ignore[call-arg]
                    payload=SegmentTurboInput(
                        prompt=intent.prompt,
                        seed=intent.seed,
                        duration_s=intent.duration_s,
                        expected_provenance=expected,
                        first_frame=frame,
                    )
                )
            else:
                call = segment(  # type: ignore[call-arg]
                    payload=SegmentInput(
                        prompt=intent.prompt,
                        seed=intent.seed,
                        duration_s=intent.duration_s,
                        steps=intent.steps,
                        expected_provenance=expected,
                        first_frame=frame,
                    )
                )
            shot_result = await call
        except ChildCallError as failure:
            # A concurrent caller cancellation must not become a successful partial result.
            ctx.raise_if_cancelled()
            failed_index, failure_code, failure_detail = index, failure.code, str(failure)[:512]
            break
        compatible(shot_result.provenance, expected)
        expected = shot_result.provenance
        records.append(
            StoredShot(
                intent=intent,
                provenance=shot_result.provenance,
                child_request_id=call.request_id,
                frames=frames_for(shot.duration_s),
                video_digest=shot_result.video.digest,
                video_bytes=shot_result.video.size_bytes,
                continuation_frame_digest=shot_result.continuation_frame.digest,
                continuation_frame_bytes=shot_result.continuation_frame.size_bytes,
            )
        )
        videos.append(shot_result.video)
        frames.append(shot_result.continuation_frame)
        warnings.extend(shot_result.warnings)
        frame = frames[-1]
    if not records:
        raise OutputError(
            f"shot 1 of {len(payload.shots)} failed ({failure_code}): {failure_detail}"
        )
    ctx.raise_if_cancelled()
    assembled = assemble(
        AssembleVideoRequest(videos=videos),
        decoder=decoder,
        out=out,
        tel=tel,
        check=ctx.raise_if_cancelled,
    )
    if [segment.source_frames for segment in assembled.segments] != [
        record.frames for record in records
    ]:
        raise OutputError(
            "retained shot video differs from its declared frame count", code="prefix_frames"
        )
    prefix = save_prefix(ctx, out, records, videos, frames)
    starts, delivered_frames = segment_clock([record.intent.duration_s for record in records])
    if assembled.output_frames != delivered_frames:
        raise OutputError("assembled output differs from the long-form clock", code="prefix_frames")
    complete = len(records) == len(payload.shots)
    if not complete:
        warnings.append(
            f"PARTIAL DELIVERY: {len(records)} of {len(payload.shots)} shots completed; "
            f"shot {failed_index + 1} failed ({failure_code}). Pass prefix as resume_from "
            "in a new request to continue without rerendering these shots."
        )
    receipts = [
        SegmentReceipt(
            seed=record.intent.seed,
            duration_s=record.intent.duration_s,
            frames=record.frames,
            start_frame=starts[index],
            video_digest=record.video_digest,
            video_bytes=record.video_bytes,
            continuation_frame_digest=record.continuation_frame_digest,
            first_frame_digest=record.intent.first_frame_digest,
            child_request_id=record.child_request_id,
            provenance=record.provenance,
        )
        for index, record in enumerate(records)
    ]
    tel.log(
        "h3 long-form delivery",
        mode=payload.mode,
        steps=steps,
        complete=complete,
        delivered=len(records),
        requested=len(payload.shots),
        reused=reused,
        delivered_frames=delivered_frames,
        failed_index=failed_index,
    )
    return LongFormOutput(
        video=assembled.video,
        prefix=prefix,
        complete=complete,
        delivered=len(records),
        reused=reused,
        segments=receipts,
        requested=len(payload.shots),
        delivered_frames=delivered_frames,
        fps=FPS,
        failed_index=failed_index,
        failure_code=failure_code,
        failure_detail=failure_detail,
        warnings=warnings,
    )


app.job(long_form, emits_media=True)
app.job(assemble_video, emits_media=True)
