"""se-001 — the MiniMax H3 launch endpoint: joint video AND audio, two task routes.

The first launch family, and the N-ARY ARTIFACT REFERENCE. SDXL is one model class over
four components; H3 is TWO task-stamped model classes over FIVE, three of which they share
byte for byte. The shape that makes that work is not in this file's power to invent — it is
the runtime's resident-component registry — and this file's whole job is to be honest about
which components each operation may touch so the registry has something true to act on.

WHAT IS DECLARED HERE

  * TWO THIN ROLE CLASSES over one shared base. `Fl2VAModel(task="fl2va")` binds
    `transformer`; `Ref2VAModel(task="ref2va")` binds `transformer_ref`. They are separate
    instances always, even when both bind the SAME dual artifact, and the only method that
    differs between them is `denoise` — because the only thing that differs is which
    transformer it may touch. There is no partition selector, no `load_state_dict` over a
    shared graph and no first-non-null-partition pick anywhere in this file.
  * SIX COMPONENT-SCOPED OPERATIONS, none of them coarse: text condition (`text_encoder`),
    visual condition and video decode (`video_vae`), audio condition and audio decode
    (`audio_vae`), and the role's denoise (its own transformer). The coarse
    whole-pipeline declaration is legal and is not servable: the tuple is 72.9 GiB and the
    smallest coherent one measured 40.56 GiB (proto-001), so a method that declares
    everything leaves the residency ladder nothing to stage on any card we rent.
  * PURE ORCHESTRATION AS MODULE FUNCTIONS. `prepare`, `pack` and the run object are
    module-level; a Model method that touches no component is module code (§1.1), and the
    runtime refuses `@uses_components()` empty for exactly that reason.
  * REQUEST ISOLATION AS STRUCTURE. Every call builds one `H3Run` holding its plan, its
    layout, its generator and its conditioning. Component and cache leases are the
    runtime's capabilities inside the `@uses_components` wrapper and are never `H3Run`
    fields. Success, cancellation and failure all discard the run.

WHAT IS DELIBERATELY ABSENT: no engine class, no device call, no placement, no offload, no
pinning, no `.to()`, no compile marker, no runtime quantization, no source-format parsing,
no checkpoint path. There is no such surface on `cozy_runtime.author` for this file to
reach, and `scripts/fence.py` refuses the spellings.

NO GUIDANCE. H3-Base is guidance-distilled: there is no `guidance_scale`, no
`negative_prompt` and one forward pass per step. SDXL's two-layer clamp demo lives on that
family because guidance is real there; here the omission is the checkpoint's fact.

STATE OF PROOF (grades, honestly): this endpoint is BUILT. Its constructed graph is
key-exact against the pinned artifact header for all five components at zero cost
(`scripts/h3-keys.py`), which is a real falsifier and is not a serve. No number in this
file has been produced on a card.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal

import msgspec
from cozy_runtime.author import (
    App,
    AudioAsset,
    Context,
    ImageAsset,
    InvalidRequest,
    Loader,
    Model,
    ModelDefault,
    OutputError,
    Outputs,
    RequestView,
    Settings,
    Shape,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
    uses_components,
)

from h3_arch import H3Config, build_component
from h3_arch.layout import (
    Keyframe as PackedKeyframe,
)
from h3_arch.layout import (
    LatentGrid,
    PackedLayout,
    RefBlock,
    TimestepPlan,
    build_timestep_plan,
    latent_grid,
    modulation_segments,
    stream_rows,
)
from h3_arch.presentation import (
    Presentation,
    PresentedReference,
    ReferenceKind,
    Tokenizer,
)
from h3_arch.presentation import build as build_presentation

app = App()

#: The released checkpoint's native clock. 48/60 fps are delivery presets over it, not
#: model rates, and this endpoint does not offer them at launch.
FPS = 24

#: The VAE decodes on a 17n+5 frame grid, so a duration preset is a LABEL, not round
#: seconds: 5 means 124 frames = 5.167 s. Snapping to slightly more than the requested
#: duration is visible in the adjustments envelope, never silent.
DurationS = Literal[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def _frames_for(seconds: int) -> int:
    n = -(-(seconds * FPS - 5) // 17)
    return 17 * n + 5


_FRAMES = {d: _frames_for(d) for d in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)}

#: Finite step presets. Each publishes an exact `(preset, task, condition presence)` ->
#: TimestepPlan digest at conformance time, which is what an exact-baked artifact's
#: coverage is matched against. A step COUNT never selects a table (§1.1.1).
StepPreset = Literal[20, 30, 50]


class AspectPreset(Enum):
    """The offered canvases. A ratio absent here does not round — it refuses."""

    W21_9 = "21:9"
    W16_9 = "16:9"
    W4_3 = "4:3"
    SQUARE = "1:1"
    T3_4 = "3:4"
    T9_16 = "9:16"
    T9_21 = "9:21"


#: THE ONE CANONICAL FAMILY TABLE — request schema, demand features, geometry and the
#: layout all read it. A copy of these numbers anywhere else is never a second authority.
#: Values are the released checkpoint's own resolutions at its 768 short edge under the
#: 1344x768 = 1,032,192-pixel budget, both axes on the 32 grid.
_PIXELS: dict[AspectPreset, tuple[int, int]] = {
    AspectPreset.W21_9: (1536, 672),
    AspectPreset.W16_9: (1344, 768),
    AspectPreset.W4_3: (1024, 768),
    AspectPreset.SQUARE: (768, 768),
    AspectPreset.T3_4: (768, 1024),
    AspectPreset.T9_16: (768, 1344),
    AspectPreset.T9_21: (672, 1536),
}

#: The released reference caps, per type and in total. Refused typed at decode.
_MAX_IMAGES = 9
_MAX_VIDEOS = 3
_MAX_AUDIO = 3
_MAX_REFERENCES = 12


# ------------------------------------------------------------------ the wire


class ImageReference(msgspec.Struct, tag="image", tag_field="kind"):
    image: ImageAsset


class VideoReference(msgspec.Struct, tag="video", tag_field="kind"):
    video: VideoAsset
    fps: float | None = None
    """Override container metadata that is missing or wrong. A video's REAL rate must
    survive decoding: it sets both the 2 fps presentation sampling and the VAE's clock."""
    sample_rate: int | None = None


class AudioReference(msgspec.Struct, tag="audio", tag_field="kind"):
    audio: AudioAsset
    sample_rate: int | None = None


#: ONE ORDERED DISCRIMINATED UNION, never three modality-grouped lists. Order controls the
#: `<Picture i>` / `<Video i>` / `<Audio i>` labels AND the shared rotary clock, so a list
#: per modality would silently destroy request semantics.
Reference = ImageReference | VideoReference | AudioReference


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    first_frame: ImageAsset | None = None
    """A TARGET-CLOCK anchor, not a reference. When present it fixes the output canvas."""
    last_frame: ImageAsset | None = None
    aspect_ratio: Annotated[AspectPreset, Shape(pixels=_PIXELS)] = AspectPreset.W16_9
    duration_s: Annotated[DurationS, Shape(frames=_FRAMES)] = 5
    num_inference_steps: ModelDefault[StepPreset] = 30
    mute: bool = False
    """An OUTPUT toggle: the audio track is omitted at finalization. Never a speed win —
    the audio stream is denoised jointly and cannot be skipped."""
    seed: int | None = None


class RefGenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    references: Annotated[list[Reference], msgspec.Meta(min_length=1, max_length=_MAX_REFERENCES)]
    first_frame: ImageAsset | None = None
    last_frame: ImageAsset | None = None
    aspect_ratio: Annotated[AspectPreset, Shape(pixels=_PIXELS)] = AspectPreset.W16_9
    duration_s: Annotated[DurationS, Shape(frames=_FRAMES)] = 5
    num_inference_steps: ModelDefault[StepPreset] = 30
    mute: bool = False
    seed: int | None = None


class GenerateOutput(msgspec.Struct):
    video: VideoAsset
    """ONE muxed mp4. The soundtrack rides INSIDE it; there is no sibling audio asset, and
    `mute` omits the track at finalization rather than producing a second file."""
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    plan_digest: str
    """The request's complete `TimestepPlan` digest — the term exact-baked coverage is
    matched against, and the determinism fence's identity half."""
    video_digest: str
    """sha256 of the decoded video pixel bytes. The fence over the WHOLE loop."""
    audio_digest: str
    checkpoint: str


class RefServeSettings(msgspec.Struct):
    """Deployment-owned toggles. Both defaults are OFF because both doors are
    output-UNVERIFIED, and a door whose LAYOUT is proven is not a door that works."""

    combined_keyframes: bool = False
    """Keyframes alongside references. v1 built the path and nobody judged the result."""


# ------------------------------------------------------------------ the pipeline


class H3Pipeline:
    """The constructed component roots of ONE task slot.

    The `components` mapping is what the runtime censuses, so its keys ARE the artifact's
    component roles. A slot exposes its own transformer role and the three shared ones; it
    never exposes the twin's, which is what makes an undeclared access spellable.
    """

    def __init__(self, config: Any, *, transformer_role: str) -> None:
        whole = H3Config.from_mapping(config.mapping())
        self.config = whole
        self.transformer_role = transformer_role
        self.components: dict[str, Any] = {
            transformer_role: build_component(transformer_role, whole),
            "text_encoder": build_component("text_encoder", whole),
            "video_vae": build_component("video_vae", whole),
            "audio_vae": build_component("audio_vae", whole),
        }

    @property
    def transformer(self) -> Any:
        return self.components[self.transformer_role]


def build_fl2va_pipeline(config: Any) -> H3Pipeline:
    return H3Pipeline(config, transformer_role="transformer")


def build_ref2va_pipeline(config: Any) -> H3Pipeline:
    return H3Pipeline(config, transformer_role="transformer_ref")


# ------------------------------------------------------------------ the run


@dataclass(frozen=True, slots=True)
class H3Plan:
    """One request's resolved facts, before any component is touched."""

    prompt: str
    grid: LatentGrid
    keyframes: tuple[PackedKeyframe, ...]
    references: tuple[PresentedReference, ...]
    ref_blocks: tuple[RefBlock, ...]
    steps: int
    mute: bool


@dataclass(slots=True)
class H3Run:
    """REQUEST SEMANTICS ONLY. Fresh per call, discarded on success, cancellation and
    failure alike. It holds no component lease and no cache handle: those are runtime
    capabilities of the `@uses_components` wrapper, and a field here would outlive them."""

    plan: H3Plan
    layout: PackedLayout
    timestep_plan: TimestepPlan
    view: RequestView
    presentation: Presentation
    conditioning: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return self.timestep_plan.digest()


def prepare(
    payload: GenerateInput | RefGenerateInput,
    *,
    tokenizer: Tokenizer,
    references: tuple[PresentedReference, ...],
    ref_blocks: tuple[RefBlock, ...],
) -> tuple[H3Plan, PackedLayout, Presentation]:
    """CPU media preparation and the exact request plan. No component is touched."""
    width, height = _PIXELS[payload.aspect_ratio]
    frames = _FRAMES[payload.duration_s]
    grid = latent_grid(frames, width, height)
    keyframes: list[PackedKeyframe] = []
    if payload.first_frame is not None:
        keyframes.append(PackedKeyframe(0))
    if payload.last_frame is not None:
        keyframes.append(PackedKeyframe(frames - 1))
    presentation = build_presentation(
        tokenizer,
        payload.prompt,
        keyframes=tuple(
            f for f in (payload.first_frame, payload.last_frame) if f is not None
        ),
        references=references,
    )
    plan = H3Plan(
        prompt=payload.prompt,
        grid=grid,
        keyframes=tuple(keyframes),
        references=references,
        ref_blocks=ref_blocks,
        steps=payload.num_inference_steps,
        mute=payload.mute,
    )
    layout = PackedLayout(
        len(presentation.rows), grid, keyframes=plan.keyframes, refs=ref_blocks
    )
    return plan, layout, presentation


def decode_references(payload: RefGenerateInput) -> tuple[PresentedReference, ...]:
    """The typed caps, refused per TYPE and in total before any decoding cost is paid."""
    counts = {"image": 0, "video": 0, "audio": 0}
    for ref in payload.references:
        counts[_kind_of(ref)] += 1
    for kind, cap in (("image", _MAX_IMAGES), ("video", _MAX_VIDEOS), ("audio", _MAX_AUDIO)):
        if counts[kind] > cap:
            raise InvalidRequest(
                f"{counts[kind]} {kind} references, and this checkpoint accepts {cap}",
                code="reference_cap",
                fields=["references"],
            )
    return tuple(_present(ref) for ref in payload.references)


def _kind_of(ref: Reference) -> ReferenceKind:
    if isinstance(ref, ImageReference):
        return "image"
    return "video" if isinstance(ref, VideoReference) else "audio"


def _present(ref: Reference) -> PresentedReference:
    kind = _kind_of(ref)
    if kind == "audio":
        # A WAVEFORM NEVER ENTERS QWEN. The presentation gets the label; the audio VAE
        # gets the samples. Upstream refused audio as the only modality; that guard was
        # measured to protect nothing and this endpoint carries the Cozy extension
        # instead, WITHOUT claiming viseme or beat synchronization.
        return PresentedReference(kind="audio")
    return PresentedReference(kind=kind)  # pixels are attached by the condition operation


# ------------------------------------------------------------------ the models


class _H3Base(Model[H3Pipeline]):
    """Everything both task slots share, including all three shared components.

    Subclasses add exactly one method — `denoise` — because exactly one component differs.
    """

    pipe: H3Pipeline
    tokenizer: Tokenizer

    def load(self, loader: Loader) -> None:
        raise NotImplementedError  # a role class states its factory; the base has none

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("text_encoder")
    def condition_text(self, rows: Any, vision: Any) -> tuple[Any, Any]:
        """Qwen3-VL over the multimodal presentation: the UNNORMALIZED hidden state after
        layer 50 plus one modality tag per embedding row."""
        import torch

        with torch.inference_mode():
            states, tags = self.pipe.components["text_encoder"](rows, vision=vision)
            return states, tags

    @uses_components("video_vae")
    def condition_visual(self, pixels: Any) -> Any:
        """Keyframe and visual-reference latents. Separate from the presentation: the same
        image reaches Qwen as a vision block and the DiT as VAE rows."""
        import torch

        with torch.inference_mode():
            return self.pipe.components["video_vae"].encode(pixels)

    @uses_components("audio_vae")
    def condition_audio(self, waveform: Any) -> Any:
        import torch

        with torch.inference_mode():
            return self.pipe.components["audio_vae"].encode(waveform)

    @uses_components("audio_vae")
    def decode_audio(self, latents: Any) -> Any:
        """The SMALL decode runs first: it is 0.6 GiB against the video VAE's 5.2, so
        taking it before the video decode is what keeps peak residency where it is."""
        import torch

        with torch.inference_mode():
            return self.pipe.components["audio_vae"].decode(latents)

    @uses_components("video_vae")
    def decode_video(self, latents: Any) -> Any:
        import torch

        with torch.inference_mode():
            return self.pipe.components["video_vae"].decode(latents)


class Fl2VAModel(_H3Base, task="fl2va"):
    """Text and first/last keyframes. Binds the `transformer` role."""

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(H3Pipeline, factory=build_fl2va_pipeline)
        self.tokenizer = Tokenizer()

    @uses_components("transformer")
    def denoise(
        self, run: H3Run, packed: Any, t_values: Any, segments: Any, on_step: Any = None
    ) -> tuple[Any, Any]:
        import torch

        with torch.inference_mode():
            return _evaluate(self.pipe.transformer, run, packed, t_values, segments, on_step)


class Ref2VAModel(_H3Base, task="ref2va"):
    """Ordered image/video/audio references, optionally with keyframes when the deployment
    opens that door. Binds the `transformer_ref` role."""

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(H3Pipeline, factory=build_ref2va_pipeline)
        self.tokenizer = Tokenizer()

    @uses_components("transformer_ref")
    def denoise(
        self, run: H3Run, packed: Any, t_values: Any, segments: Any, on_step: Any = None
    ) -> tuple[Any, Any]:
        import torch

        with torch.inference_mode():
            return _evaluate(self.pipe.transformer, run, packed, t_values, segments, on_step)


#: What the module functions accept. NOT `_H3Base`: the base deliberately has no `denoise`,
#: because a method on it would have to declare a component set, and the whole point is
#: that the set is the one thing the two roles do not share. An undecorated `denoise` on
#: the base would declare ALL components by omission — the coarse contract this endpoint
#: exists to avoid — so the union is the honest type.
H3Model = Fl2VAModel | Ref2VAModel


def _evaluate(
    transformer: Any, run: H3Run, packed: Any, t_values: Any, segments: Any, on_step: Any
) -> tuple[Any, Any]:
    """The one denoise evaluation body, shared because it is the same arithmetic. It takes
    the transformer as an ARGUMENT: which one it is was decided by the caller's declared
    component set, and this function has no opinion about it."""
    import torch

    video_seg, audio_seg = stream_rows(
        run.layout,
        list(t_values.tolist()),
        t_video=run.conditioning["t_video"],
        t_audio=run.conditioning["t_audio"],
    )
    video, audio = transformer(
        packed,
        torch.as_tensor(t_values),
        segments,
        run.conditioning["position_ids"],
        video_seg,
        audio_seg,
        on_step=on_step,
    )
    return video, audio


# ------------------------------------------------------------------ the output floor


#: THE OUTPUT-INTEGRITY FLOOR, and the #411 lesson carried across families: NaN is read off
#: the FLOAT decode, before quantization. A `clamp(0,1).to(uint8)` erases exactly the
#: evidence a diverged decode leaves, so a fully-NaN generation checked on pixels reports
#: clean. Spread is read off the pixels a caller will actually open.
#:
#: H3 has a THIRD branch SDXL does not need, and it is the one the fp8-under-staging
#: history argues for: a silent audio track. A video whose picture is fine and whose
#: soundtrack is digital silence is a joint generation that half-failed, and it encodes,
#: plays, and looks correct until someone turns the volume up.
_MIN_VIDEO_SPREAD = 4.0
_MIN_AUDIO_RMS = 1e-4


def _integrity(torch: Any, decoded: Any, pixels: Any, waveform: Any, tel: Telemetry) -> None:
    video_nan = float(torch.isnan(decoded).float().mean())
    audio_nan = float(torch.isnan(waveform).float().mean())
    tel.metric("video_nan_fraction", round(video_nan, 6))
    tel.metric("audio_nan_fraction", round(audio_nan, 6))
    if video_nan > 0.0 or audio_nan > 0.0:
        raise OutputError(
            f"the decode produced NaN over {video_nan:.4%} of the video and "
            f"{audio_nan:.4%} of the audio, and this endpoint does not publish it: a "
            "non-finite decode is a failed generation, not a clip with artefacts",
            code="output_integrity_nan",
        )
    spread = float(pixels.to(torch.float32).std())
    tel.metric("video_spread", round(spread, 4))
    if spread <= _MIN_VIDEO_SPREAD:
        raise OutputError(
            f"the decoded video is a flat field (std {spread:.3f} <= {_MIN_VIDEO_SPREAD}): "
            "the generation produced no picture, and an encodable rectangle is not a result",
            code="output_integrity_flat",
        )
    rms = float(waveform.to(torch.float32).pow(2).mean().sqrt())
    tel.metric("audio_rms", round(rms, 8))
    if rms <= _MIN_AUDIO_RMS:
        raise OutputError(
            f"the decoded soundtrack is silence (rms {rms:.3e} <= {_MIN_AUDIO_RMS}): H3 "
            "generates audio jointly, so a silent track is a half-failed generation and "
            "not a muted preference — `mute` omits the track at finalization instead",
            code="output_integrity_silent",
        )


# ------------------------------------------------------------------ the handlers


def _run(
    model: H3Model,
    ctx: Context,
    payload: GenerateInput | RefGenerateInput,
    out: Outputs,
    tel: Telemetry,
    *,
    task: str,
    references: tuple[PresentedReference, ...],
    ref_blocks: tuple[RefBlock, ...],
) -> GenerateOutput:
    """The one generation body. Both routes are the same six operations in the same order;
    what differs is which transformer the role class declared and what the plan contains."""
    import torch

    view = model.for_request(ctx, seed=payload.seed)
    with tel.stage("prepare"):
        plan, layout, presentation = prepare(
            payload, tokenizer=model.tokenizer, references=references, ref_blocks=ref_blocks
        )
    config = model.pipe.config.dit
    timestep_plan = build_timestep_plan(
        task=task,
        structure=config.structure.value,
        steps=plan.steps,
        layout=layout,
        sigma_shift_video=config.sigma_shift_video,
        sigma_shift_audio=config.sigma_shift_audio,
        visual_cond_timestep=0.999 if plan.keyframes or plan.references else None,
        audio_cond_timestep=1.0 if any(b.ref_audio_t for b in ref_blocks) else None,
        adapters=tuple(str(a) for a in view.adapters),
    )
    run = H3Run(plan, layout, timestep_plan, view, presentation)
    tel.metric("packed_rows", float(layout.seq_len))
    tel.metric("steps", float(plan.steps))
    ctx.raise_if_cancelled()

    with tel.stage("condition_text"):
        text_states, tags = model.condition_text(presentation.rows, presentation.tags)
    if plan.keyframes or plan.references:
        with tel.stage("condition_visual"):
            run.conditioning["visual"] = model.condition_visual(run.conditioning.get("pixels"))
    if any(b.ref_audio_t for b in ref_blocks):
        with tel.stage("condition_audio"):
            run.conditioning["audio"] = model.condition_audio(run.conditioning.get("waveform"))

    on_step = tel.step_callback(plan.steps, stage="denoise")
    with tel.stage("denoise"):
        latents = _denoise_loop(torch, model, run, text_states, tags, on_step, ctx)

    with tel.stage("decode_audio"):
        waveform = model.decode_audio(latents["audio"])
    with tel.stage("decode_video"):
        decoded = model.decode_video(latents["video"])

    pixels = ((decoded / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8)
    _integrity(torch, decoded, pixels, waveform, tel)
    frames = pixels.permute(1, 2, 3, 0).contiguous()
    rgb = bytes(frames.cpu().numpy().tobytes())
    samples = bytes(waveform.to(torch.float32).cpu().numpy().tobytes())

    with tel.stage("encode_mp4"):
        asset = out.save_video(
            frames,
            fps=FPS,
            audio=None if plan.mute else waveform,
            sample_rate=model.pipe.config.audio_vae.sample_rate,
        )
    return GenerateOutput(
        video=asset,
        width=plan.grid.width,
        height=plan.grid.height,
        frames=plan.grid.frames,
        fps=FPS,
        steps=plan.steps,
        plan_digest=run.digest,
        video_digest=hashlib.sha256(rgb).hexdigest(),
        audio_digest=hashlib.sha256(samples).hexdigest(),
        checkpoint=model.checkpoint_ref,
    )


def _denoise_loop(
    torch: Any,
    model: H3Model,
    run: H3Run,
    text_states: Any,
    tags: Any,
    on_step: Any,
    ctx: Context,
) -> dict[str, Any]:
    """The joint video+audio sampling loop. ONE forward per step — no negative branch."""
    plan = run.timestep_plan
    generator = torch.Generator(device="cpu").manual_seed(run.view._seed)
    grid = run.plan.grid
    video = torch.randn(
        1, 24, grid.latent_t, grid.latent_h, grid.latent_w, generator=generator
    )
    audio = torch.randn(1, 32, 2, grid.audio_t, generator=generator)
    for index in range(plan.steps):
        ctx.raise_if_cancelled()
        run.conditioning["t_video"] = 1.0 - plan.video_sigmas[index]
        run.conditioning["t_audio"] = 1.0 - plan.audio_sigmas[index]
        segments, unique = modulation_segments(
            run.layout,
            t_video=run.conditioning["t_video"],
            t_audio=run.conditioning["t_audio"],
            visual_cond_t=plan.visual_cond_timestep or 0.0,
            audio_cond_t=plan.audio_cond_timestep or 0.0,
            text_token_tags=tuple(tags),
        )
        run.conditioning["position_ids"] = torch.tensor(
            run.layout.position_ids, dtype=torch.float64
        )
        packed = _pack(torch, run, text_states, video, audio)
        v_out, a_out = model.denoise(
            run, packed, torch.tensor(unique, dtype=torch.float32), segments, on_step
        )
        dt_v = plan.video_sigmas[index + 1] - plan.video_sigmas[index]
        dt_a = plan.audio_sigmas[index + 1] - plan.audio_sigmas[index]
        video = video + dt_v * v_out
        audio = audio + dt_a * a_out
        on_step(index)
    return {"video": video, "audio": audio}


def _pack(torch: Any, run: H3Run, text_states: Any, video: Any, audio: Any) -> Any:
    """Assemble the packed sequence by SEGMENT SLICE. Segments are contiguous and uniform,
    which is why this is a handful of copies rather than a gather over 100k rows."""
    from h3_arch.dit import pack_audio, patchify_video

    rows: list[Any] = []
    video_rows = patchify_video(video, (1, 2, 2))
    audio_rows = pack_audio(audio)
    for _, _, kind in run.layout.segments:
        if kind == "text":
            rows.append(text_states)
        elif kind in ("cond", "ref_img"):
            rows.append(run.conditioning["visual"])
        elif kind == "ref_audio":
            rows.append(run.conditioning["audio"])
        elif kind == "video":
            rows.append(video_rows)
        else:
            rows.append(audio_rows)
    return torch.cat(rows, dim=0)


@app.entrypoint
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: Fl2VAModel,
    out: Outputs,
    tel: Telemetry,
) -> GenerateOutput:
    """Text, optionally anchored by a first and/or last frame, to one muxed mp4."""
    return _run(model, ctx, payload, out, tel, task="fl2va", references=(), ref_blocks=())


@app.entrypoint
def reference_to_video(
    ctx: Context,
    payload: RefGenerateInput,
    model: Ref2VAModel,
    out: Outputs,
    tel: Telemetry,
    settings: Settings[RefServeSettings],
) -> GenerateOutput:
    """Ordered image/video/audio references, in request order, to one muxed mp4."""
    if (payload.first_frame or payload.last_frame) and not settings.value.combined_keyframes:
        raise UnsupportedInput(
            "keyframes alongside references are not enabled on this deployment: the "
            "combined layout is proven and its OUTPUT has never been judged, so the door "
            "opens per deployment rather than by default",
            code="combined_keyframes_disabled",
            fields=["first_frame", "last_frame"],
        )
    references = decode_references(payload)
    blocks = tuple(RefBlock(kind="image") for _ in references)
    return _run(
        model,
        ctx,
        payload,
        out,
        tel,
        task="ref2va",
        references=references,
        ref_blocks=blocks,
    )
