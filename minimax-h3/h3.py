"""MiniMax-H3's two official serving actions over one dual-DiT construction.

Runtime decodes typed assets and encodes outputs. Official Diffusers owns H3 presentation,
conditioning, AdaLN, solver, and decode math. This module validates the product contract,
stages weighted roots, and joins those two boundaries.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    Context,
    DecodedAudio,
    DecodedVideo,
    Image,
    ImageAsset,
    ImageFrame,
    InvalidRequest,
    Loader,
    Mixed,
    Model,
    OutputError,
    Outputs,
    Preflight,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
    uses_components,
)

from gates import MediaFacts, pre_encode_gate
from official import (
    FPS,
    FRAMES,
    MAX_AUDIO_REFERENCES,
    MAX_CONDITIONER_VISION_TOKENS,
    MAX_IMAGE_REFERENCES,
    MAX_REFERENCES,
    MAX_VIDEO_REFERENCES,
    REFERENCE_IMAGE_SHORT_EDGE,
    NumericalChecks,
    OfficialH3Pipeline,
    ReferencePolicyFacts,
    ScheduleFacts,
    Task,
    build_h3_pipeline,
    reference_image_vision_tokens,
    reference_video_vision_tokens,
    supported_steps,
    validate_reference_policy,
)

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
    msgspec.Meta(min_length=1),
]
KeyframeAssets = Annotated[Assets[Image], AssetLimits(images=2)]
_REFERENCE_FIDELITY_EDGES = {"low": 256, "medium": 1024, "high": REFERENCE_IMAGE_SHORT_EDGE}
Prompt = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
# The wire enum is the committed plans' step counts; a bound lane serves exactly these
# and the fastest is the default.
SUPPORTED_STEPS = supported_steps()
DEFAULT_STEPS = min(SUPPORTED_STEPS)
Steps = Annotated[
    Literal[SUPPORTED_STEPS],  # type: ignore[valid-type]
    msgspec.Meta(description="Denoise steps (transformer evaluations); fewer is faster."),
]


class FirstLastFrameToVideoInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    mute: bool = False
    seed: int | None = None
    steps: Steps = DEFAULT_STEPS


class ReferenceMediaToVideoInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Prompt
    mute: bool = False
    seed: int | None = None
    reference_image_short_edge: Annotated[
        int,
        msgspec.Meta(
            ge=256,
            le=REFERENCE_IMAGE_SHORT_EDGE,
            description="Image-reference short edge in pixels; lower trades detail for speed.",
        ),
    ] = REFERENCE_IMAGE_SHORT_EDGE
    steps: Steps = DEFAULT_STEPS


class H3VideoOutput(msgspec.Struct):
    """The catalog wire shape. Checkpoint, plan, geometry, and digest facts remain
    attempt observations (se-012): they ride Telemetry, never the customer result."""

    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    continuation_frame: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    warnings: list[str] = msgspec.field(default_factory=list)


def preflight_reference_media(
    payload: ReferenceMediaToVideoInput, assets: ReferenceAssets
) -> ReferencePolicyFacts:
    """Refuse cross-field count errors before Runtime hydrates a single asset."""
    del payload
    kinds = [assets.info(index).kind for index in range(len(assets))]
    try:
        return validate_reference_policy(kinds)
    except ValueError as exc:
        raise UnsupportedInput(str(exc), code="reference_policy", fields=["assets"]) from exc


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


def assets_to_h3_refs(
    references: ReferenceAssets,
    *,
    pipe: OfficialH3Pipeline,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
) -> list[Any]:
    prepared: list[Any] = []
    video_duration = Fraction(0)
    audio_duration = Fraction(0)
    vision_tokens = 0

    for index, reference in enumerate(references):
        info = references.info(index)
        field = info.id
        if isinstance(reference, Image):
            image = reference
            _validate_ratio(image.width, image.height, field)
            short_edge = (
                reference_image_short_edge
                if info.fidelity == "auto"
                else _REFERENCE_FIDELITY_EDGES[info.fidelity]
            )
            vision_tokens += reference_image_vision_tokens(image.width, image.height, short_edge)
            _validate_vision_budget(vision_tokens)
            prepared.append(pipe.image_reference(image, short_edge=short_edge))
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
            vision_tokens += reference_video_vision_tokens(
                video.width, video.height, video.duration
            )
            _validate_vision_budget(vision_tokens)
            prepared.append(pipe.video_reference(video))
        else:
            audio = reference
            _validate_audio(audio, field)
            audio_duration += audio.duration
            _validate_audio_aggregate(audio_duration)
            prepared.append(pipe.audio_reference(audio))
    return prepared


def _validate_vision_budget(tokens: int) -> None:
    if tokens > MAX_CONDITIONER_VISION_TOKENS:
        raise UnsupportedInput(
            f"reference vision presentation needs {tokens} tokens; this release admits at most "
            f"{MAX_CONDITIONER_VISION_TOKENS}",
            code="reference_policy",
            fields=["assets"],
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
def fl2va(
    ctx: Context,
    payload: FirstLastFrameToVideoInput,
    assets: KeyframeAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        first_index, last_index = _keyframe_roles(assets)
        first = _keyframe_image(assets, first_index, field="first")
        last = _keyframe_image(assets, last_index, field="last")
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
def ref2va(
    ctx: Context,
    payload: ReferenceMediaToVideoInput,
    assets: ReferenceAssets,
    facts: Preflight[ReferencePolicyFacts],
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> H3VideoOutput:
    ctx.raise_if_cancelled()
    del facts
    view = model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        references = assets_to_h3_refs(
            assets,
            pipe=model.pipe,
            reference_image_short_edge=payload.reference_image_short_edge,
        )
        state = model.pipe.start_ref2va(
            prompt=payload.prompt,
            references=references,
            generator=model.pipe.generator(view.generator),
            steps=payload.steps,
            reference_image_short_edge=payload.reference_image_short_edge,
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
