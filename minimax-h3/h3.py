"""MiniMax-H3's two official serving actions over one dual-DiT construction.

Runtime decodes typed assets and encodes outputs. Official Diffusers owns H3 presentation,
conditioning, AdaLN, solver, and decode math. This module validates the product contract,
stages weighted roots, and joins those two boundaries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    AudioAsset,
    Context,
    DecodedAudio,
    DecodedVideo,
    ImageAsset,
    ImageFrame,
    InvalidRequest,
    Loader,
    MediaDecoder,
    Model,
    OutputError,
    Outputs,
    Preflight,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
    uses_components,
)
from msgspec.structs import replace

from gates import MediaFacts, pre_encode_gate
from official import (
    FPS,
    FRAMES,
    MAX_CONDITIONER_VISION_TOKENS,
    REFERENCE_IMAGE_SHORT_EDGE,
    NumericalChecks,
    OfficialH3Pipeline,
    ReferencePolicyFacts,
    ScheduleFacts,
    Task,
    build_h3_pipeline,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    validate_reference_policy,
)

app = App()

_MIB = 1 << 20
_GIB = 1 << 30
_IMAGE_BOUND = AssetBound(max_bytes=64 * _MIB, max_decoded_bytes=256 * _MIB)
_AUDIO_BOUND = AssetBound(max_bytes=128 * _MIB, max_decoded_bytes=128 * _MIB)
_VIDEO_BOUND = AssetBound(max_bytes=2 * _GIB, max_decoded_bytes=2 * _GIB)
_MIN_REFERENCE_DURATION = Fraction(2, 1)
_MAX_REFERENCE_DURATION = Fraction(15, 1)
DEFAULT_REFERENCE_IMAGE_SHORT_EDGE = 1024
#: Auto-sizing rungs, walked downward from the request default until the vision budget fits.
_SHORT_EDGE_LADDER = (2048, 1536, 1024, 768, 512, 256)
ShortEdge = Annotated[
    int,
    msgspec.Meta(
        ge=256,
        le=REFERENCE_IMAGE_SHORT_EDGE,
        description="Image-reference short edge in pixels; lower trades detail for speed.",
    ),
]


class ImageReference(msgspec.Struct, tag="image", tag_field="type", forbid_unknown_fields=True):
    image: Annotated[ImageAsset, _IMAGE_BOUND]
    #: Explicit fidelity. Absent, the request default applies and may auto-step down.
    short_edge: ShortEdge | None = None


class VideoReference(msgspec.Struct, tag="video", tag_field="type", forbid_unknown_fields=True):
    video: Annotated[VideoAsset, _VIDEO_BOUND]


class AudioReference(msgspec.Struct, tag="audio", tag_field="type", forbid_unknown_fields=True):
    audio: Annotated[AudioAsset, _AUDIO_BOUND]


Reference = ImageReference | VideoReference | AudioReference
Prompt = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
# Static wire declarations; h3-conform verifies the enum against both committed plans.
# The fastest supported schedule is the default.
DEFAULT_STEPS = 30
Steps = Annotated[
    Literal[30, 40, 50],
    msgspec.Meta(description="Denoise steps (transformer evaluations); fewer is faster."),
]


class FirstLastFrameToVideoInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    first_frame: Annotated[ImageAsset | None, _IMAGE_BOUND] = None
    last_frame: Annotated[ImageAsset | None, _IMAGE_BOUND] = None
    mute: bool = False
    seed: int | None = None
    steps: Steps = DEFAULT_STEPS


class ReferenceMediaToVideoInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    references: Annotated[list[Reference], msgspec.Meta(min_length=1, max_length=12)]
    mute: bool = False
    seed: int | None = None
    #: The short edge of every image without its own; stepped down until the budget fits.
    reference_image_short_edge: ShortEdge = DEFAULT_REFERENCE_IMAGE_SHORT_EDGE
    steps: Steps = DEFAULT_STEPS


class H3VideoOutput(msgspec.Struct):
    """The catalog wire shape. Checkpoint, plan, geometry, and digest facts remain
    attempt observations (se-012): they ride Telemetry, never the customer result."""

    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    continuation_frame: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    warnings: list[str] = msgspec.field(default_factory=list)


def preflight_reference_media(payload: ReferenceMediaToVideoInput) -> ReferencePolicyFacts:
    """Refuse cross-field count errors before Runtime hydrates a single asset."""
    kinds = [_reference_kind(reference) for reference in payload.references]
    try:
        return validate_reference_policy(kinds)
    except ValueError as exc:
        raise UnsupportedInput(str(exc), code="reference_policy", fields=["references"]) from exc


def _reference_kind(reference: Reference) -> str:
    if isinstance(reference, ImageReference):
        return "image"
    if isinstance(reference, VideoReference):
        return "video"
    return "audio"


class H3Model(Model[OfficialH3Pipeline], encoded_leaves="accept", fusion="accept"):
    pipe: OfficialH3Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3Pipeline, factory=build_h3_pipeline)

    def unload(self, loader: Loader) -> None:
        return None

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
    def decode_video(self, task: Task, state: Any, *, checks: NumericalChecks | None = None) -> Any:
        if checks is not None:
            checks.component("video_vae", self.pipe.components["video_vae"])
        video = self.pipe.decode_video(task, state)
        if checks is not None:
            checks.tensors("decode_video", [("video", video)])
        return video

    @uses_components("video_vae")
    def condition_fl2va_media(self, state: Any, *, checks: NumericalChecks) -> None:
        checks.component("video_vae", self.pipe.components["video_vae"])
        self.pipe.condition_media("fl2va", state, checks=checks)

    @uses_components("video_vae", "audio_vae")
    def condition_ref2va_media(self, state: Any, *, checks: NumericalChecks) -> None:
        for name in ("video_vae", "audio_vae"):
            checks.component(name, self.pipe.components[name])
        self.pipe.condition_media("ref2va", state, checks=checks)

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


def _decode_keyframe(
    asset: ImageAsset | None,
    *,
    field: str,
    decoder: MediaDecoder,
    pipe: OfficialH3Pipeline,
) -> Any | None:
    if asset is None:
        return None
    image = decoder.decode_image(asset)
    _validate_ratio(image.width, image.height, field)
    return pipe.keyframe(image)


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
    """An explicit short edge is never changed. The images without one share the request
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
        fields=["references"],
    )


def _decode_references(
    references: list[Reference],
    *,
    decoder: MediaDecoder,
    pipe: OfficialH3Pipeline,
    reference_image_short_edge: int = DEFAULT_REFERENCE_IMAGE_SHORT_EDGE,
) -> tuple[list[Any], ReferenceSizing]:
    prepared: list[Any] = []
    images: list[ImageSizing] = []
    video_duration = Fraction(0)
    audio_duration = Fraction(0)
    video_tokens = 0

    for index, reference in enumerate(references):
        field = f"references.{index}"
        if isinstance(reference, ImageReference):
            image = decoder.decode_image(reference.image)
            _validate_ratio(image.width, image.height, field)
            images.append(ImageSizing(field, image.width, image.height, reference.short_edge))
            prepared.append(pipe.image_reference(image))
        elif isinstance(reference, VideoReference):
            video = decoder.decode_video(reference.video)
            _validate_video(video, field)
            video_duration += video.duration
            if video_duration > _MAX_REFERENCE_DURATION:
                raise InvalidRequest(
                    "reference videos total more than 15 seconds",
                    code="reference_policy",
                    fields=["references"],
                )
            video_tokens += reference_video_vision_tokens(video.width, video.height, video.duration)
            prepared.append(pipe.video_reference(video))
        else:
            audio = decoder.decode_audio(reference.audio)
            _validate_audio(audio, field)
            audio_duration += audio.duration
            _validate_audio_aggregate(audio_duration)
            prepared.append(pipe.audio_reference(audio))
    sizing = resolve_reference_sizing(
        images, default=reference_image_short_edge, video_tokens=video_tokens
    )
    return prepared, sizing


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
            fields=["references"],
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
    mute: bool,
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
    with tel.stage("decode_video", overall_range=(0.90, 0.97)):
        decoded = model.decode_video(task, state, checks=checks)
        if decoded.ndim != 5 or int(decoded.shape[0]) != 1 or int(decoded.shape[2]) != 3:
            raise OutputError(
                f"official H3 video decode returned shape {tuple(decoded.shape)}",
                code="output_integrity",
            )

    cancel()
    video_nonfinite_fraction = _nonfinite_fraction(torch, decoded)
    audio_nonfinite_fraction = _nonfinite_fraction(torch, waveform)
    pixels, video_pixel_digest = _rgb8(torch, decoded)
    del decoded
    frames, height, width, channels = (int(value) for value in pixels.shape)
    facts = MediaFacts(
        width=width,
        height=height,
        frames=frames,
        fps=FPS,
        sample_rate=sample_rate,
        mute=mute,
    )
    if frames != FRAMES or channels != 3:
        raise OutputError(
            f"official H3 decode returned {frames} frames and {channels} channels",
            code="output_integrity",
        )

    with tel.stage("check_output", overall_range=(0.97, 0.98)):
        warnings = pre_encode_gate(
            torch,
            pixels=pixels,
            waveform=waveform,
            video_nonfinite_fraction=video_nonfinite_fraction,
            audio_nonfinite_fraction=audio_nonfinite_fraction,
            requested=facts,
            tel=tel,
        )

    cancel()
    pixel_array = pixels.numpy()
    audio_array = waveform.numpy()
    frame_bytes = bytes(pixel_array[-1])
    with tel.stage("encode_outputs", overall_range=(0.98, 1.00)):
        video = out.save_video(
            pixels,
            fps=FPS,
            audio=None if mute else waveform,
            sample_rate=sample_rate,
        )
        continuation = out.save_image(ImageFrame(width, height, frame_bytes), format="png")

    tel.log(
        "h3 output geometry",
        width=width,
        height=height,
        frames=frames,
        fps=FPS,
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
        audio_sample_digest=hashlib.sha256(audio_array).hexdigest(),
        continuation_pixel_digest=hashlib.sha256(frame_bytes).hexdigest(),
    )
    return H3VideoOutput(video=video, continuation_frame=continuation, warnings=warnings)


#: Frames per handoff chunk: 8 frames of 1344x768 are 100 MB of fp32 source and 25 MB of
#: RGB8, so one chunk's device temporaries stay small and the hash of one chunk (~25 ms)
#: hides under the quantize-and-copy of the next.
_RGB8_CHUNK_FRAMES = 8


def _rgb8(torch: Any, decoded: Any) -> tuple[Any, str]:
    """Bounded float-decode to ONE CPU RGB8 buffer, digested as it lands (h3a-017).

    Each chunk is quantized on the device, copied straight into its slice of the host
    buffer (no per-chunk host temporary), and handed to one hashing thread while the next
    chunk copies. The chunks are consecutive slices of a contiguous buffer, so the running
    sha256 is byte-for-byte the digest of the finished buffer — the `h3 source digests`
    identity — computed under the copies instead of after them.
    """
    source = decoded[0]
    frames, _, height, width = (int(value) for value in source.shape)
    pixels = torch.empty((frames, height, width, 3), dtype=torch.uint8, device="cpu")
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=1) as hasher:
        for start in range(0, frames, _RGB8_CHUNK_FRAMES):
            chunk = source[start : start + _RGB8_CHUNK_FRAMES]
            chunk.clamp_(0, 1).mul_(255).round_()
            landed = pixels[start : start + len(chunk)]
            landed.copy_(chunk.to(torch.uint8).permute(0, 2, 3, 1).contiguous())
            hasher.submit(digest.update, landed.numpy())
    return pixels, digest.hexdigest()


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


@app.entrypoint()
def first_last_frame_to_video(
    ctx: Context,
    payload: FirstLastFrameToVideoInput,
    model: H3Model,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        first = _decode_keyframe(
            payload.first_frame, field="first_frame", decoder=decoder, pipe=model.pipe
        )
        last = _decode_keyframe(
            payload.last_frame, field="last_frame", decoder=decoder, pipe=model.pipe
        )
        state = model.pipe.start_fl2va(
            prompt=payload.prompt,
            first_frame=first,
            last_frame=last,
            generator=model.pipe.generator(view.generator),
            steps=payload.steps,
        )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        model.condition_text("fl2va", state, checks=checks)
    if first is not None or last is not None:
        with tel.stage("condition_media", overall_range=(0.08, 0.15)):
            model.condition_fl2va_media(state, checks=checks)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = model.sample_fl2va(
            state,
            on_step=tel.step_callback(payload.steps, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    return _finish(
        model,
        "fl2va",
        state,
        schedule,
        mute=payload.mute,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        checks=checks,
    )


@app.entrypoint(preflight=preflight_reference_media)
def reference_media_to_video(
    ctx: Context,
    payload: ReferenceMediaToVideoInput,
    facts: Preflight[ReferencePolicyFacts],
    model: H3Model,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    ctx.raise_if_cancelled()
    del facts
    view = model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        references, sizing = _decode_references(
            payload.references,
            decoder=decoder,
            pipe=model.pipe,
            reference_image_short_edge=payload.reference_image_short_edge,
        )
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
            steps=payload.steps,
            reference_image_short_edges=sizing.edges,
        )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        model.condition_text("ref2va", state, checks=checks)
    with tel.stage("condition_media", overall_range=(0.08, 0.15)):
        model.condition_ref2va_media(state, checks=checks)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = model.sample_ref2va(
            state,
            on_step=tel.step_callback(payload.steps, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    return _finish(
        model,
        "ref2va",
        state,
        schedule,
        mute=payload.mute,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        checks=checks,
    )
