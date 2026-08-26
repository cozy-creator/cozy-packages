"""se-001 — the MiniMax H3 launch endpoint: joint video AND audio, two task routes.

The first launch family, and the N-ARY ARTIFACT REFERENCE. SDXL is one model class over
four components; H3 is TWO task-stamped model classes over FIVE, three of which they share
byte for byte. The shape that makes that work is not in this file's power to invent — it is
the runtime's resident-component registry — and this file's whole job is to be honest about
which components each operation may touch so the registry has something true to act on.

WHAT IS DECLARED HERE

  * TWO CONSTRUCTION LAYERS, chosen by the ARTIFACT and not by this file. `H3Config.graph`
    names one: `h3-native` builds the hand port (`h3_arch/`, the community curve carrier's
    3,445 destinations) and `h3-diffusers` builds upstream's official classes through
    `h3_ref/` (the official tree's 3,486). The term selects a KEY SET, so it is structural
    (§1.1.1) and defaults to the dialect the one ingested H3 artifact actually carries.
  * TWO THIN ROLE CLASSES over one shared base. `Fl2VAModel(task="fl2va")` binds
    `transformer`; `Ref2VAModel(task="ref2va")` binds `transformer_ref`. They are separate
    instances always, even when both bind the SAME dual artifact, and the only method that
    differs between them is `predict_data_velocity` — because the only thing that differs
    is which transformer it may touch. There is no partition selector, no
    `load_state_dict` over a shared graph and no first-non-null-partition pick here.
  * SIX COMPONENT-SCOPED OPERATIONS, none of them coarse: text condition (`text_encoder`),
    visual condition and video decode (`video_vae`), audio condition and audio decode
    (`audio_vae`), and the role's velocity prediction (its own transformer). The coarse
    whole-pipeline declaration is legal and is not servable: the tuple is 72.9 GiB and the
    smallest coherent one measured 40.56 GiB (proto-001), so a method that declares
    everything leaves the residency ladder nothing to stage on any card we rent.
  * ONE MODEL BOUNDARY, one SOLVER. `predict_data_velocity(run, text, video_latents,
    audio_latents, modulation)` takes latents and returns LATENT-SHAPED, DATA-WARD
    velocities; it owns projection, packing, the forward and unpacking, so nothing
    row-shaped is visible to orchestration. `layout.H3Solver` owns sign and schedule and
    touches no component. Those are the two halves se-002's first render got wrong at
    once: the sign was inverted and the row-shaped middle leaked out of every scope.
  * PURE ORCHESTRATION AS MODULE FUNCTIONS. `prepare`, `_sample` and the run object are
    module-level; a Model method that touches no component is module code (§1.1), and the
    runtime refuses `@uses_components()` empty for exactly that reason.
  * TWO OUTPUT GATES ON EVERY REQUEST, over cozy-eval's own instruments (`gates.py`): the
    tensors before encoding, and the MP4 after. #520's standing law — a generation that
    fails the gate is a failed generation regardless of exit status.
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

STATE OF PROOF (grades, honestly). The constructed graph is KEY-EXACT against the pinned
artifact header for all five components at zero cost (`scripts/h3-keys.py`). The reference
semantics this file's sampler depends on — the data-ward sign, the 17k+5 temporal
geometry, the sigma grid and its evaluation count, the pixel conversion, the
requested-vs-observed media agreement — are CONFORMANCE-PROVEN ON CPU against
independently written upstream expressions (`scripts/h3-conform.py`, which runs in CI).
The output gates are proven against digest-pinned real media, including the failed
`daf50552…` render as a permanent red arm (`scripts/h3-live.py gates`).

NONE OF THAT IS AN OUTPUT VERIFICATION. The whole-seam oracle (#523.3) and a full-length
viewed render (#521) are a later lane, and the first serve of this file produced garbage
with every component key-exact — which is exactly why "the components match" is not a
grade anyone should quote as working.
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
    Outputs,
    RequestView,
    Settings,
    Shape,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
    uses_components,
)

from gates import post_encode_gate, pre_encode_gate
from h3_arch import GraphDialect, H3Config, build_component
from h3_arch.layout import (
    FPS,
    FRAMES_PER_CLIP,
    HEAD_FRAMES,
    H3Solver,
    LatentGrid,
    MediaFacts,
    Modulation,
    PackedLayout,
    RefBlock,
    TimestepPlan,
    build_modulation,
    build_timestep_plan,
    latent_grid,
)
from h3_arch.layout import (
    Keyframe as PackedKeyframe,
)
from h3_arch.pixels import pixel_bytes
from h3_arch.presentation import (
    Presentation,
    PresentedReference,
    ReferenceKind,
    Tokenizer,
)
from h3_arch.presentation import build as build_presentation

app = App()

#: The VAE decodes on a 17k+5 frame grid, so a duration preset is a LABEL, not round
#: seconds: 5 means 124 frames = 5.167 s. Snapping to slightly more than the requested
#: duration is visible in the adjustments envelope, never silent.
DurationS = Literal[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def _frames_for(seconds: int) -> int:
    """The smallest on-grid frame count that covers `seconds`. The grid constants are
    `layout`'s — a second copy of `17` and `5` here is a second authority for the number
    that was already wrong once (#522b)."""
    clips = -(-(seconds * FPS - HEAD_FRAMES) // FRAMES_PER_CLIP)
    return FRAMES_PER_CLIP * clips + HEAD_FRAMES


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

    TWO CONSTRUCTION LAYERS, SELECTED BY THE ARTIFACT (#531/#540). `h3_arch` is the hand
    port and answers to the community curve carrier's key set; `h3_ref` is a thin layer over
    upstream's official classes and answers to the official diffusers-format tree's. The
    artifact's own config says which, through `H3Config.graph`, and nothing else in this file
    knows the difference — `build_component(role, ...)` is the same signature on both sides,
    which is the whole reason the rebase can be a construction decision rather than a fork.
    """

    def __init__(self, config: Any, *, transformer_role: str) -> None:
        whole = H3Config.from_mapping(config.mapping())
        self.config = whole
        self.transformer_role = transformer_role
        roles = (transformer_role, "text_encoder", "video_vae", "audio_vae")
        self.components: dict[str, Any]
        if whole.graph is GraphDialect.DIFFUSERS:
            import h3_ref

            mapping = h3_ref.config_mapping(whole)
            self.components = {r: h3_ref.build_component(r, mapping) for r in roles}
        else:
            self.components = {r: build_component(r, whole) for r in roles}

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
    def condition_text(self, presentation: Presentation) -> Any:
        """Qwen3-VL over the presentation: the UNNORMALIZED hidden state after layer 50,
        `[1, L, text_dim]`.

        THE VISION SEAM IS NOT BUILT and this refuses rather than pretending. The
        conditioner takes `[batch, seq]` token ids plus already-patchified
        `VisionBlock(patches, grid_thw, index)` splices, and nothing in this endpoint turns
        a keyframe's pixels into that triple. se-002 hit the gap on the pod and patched
        around it with the same refusal; the refusal belongs in the landed source."""
        import torch

        if any(not isinstance(row, int) for row in presentation.rows):
            raise UnsupportedInput(
                "vision-block conditioning is not built in this port: the presentation to "
                "conditioner seam (pixels to patches, grid and splice index) has no "
                "implementation, so a keyframe or visual reference would reach Qwen3-VL as "
                "nothing at all",
                code="vision_seam_unbuilt",
                fields=["first_frame", "last_frame", "references"],
            )
        encoder = self.pipe.components["text_encoder"]
        with torch.inference_mode():
            tokens = torch.tensor(
                [presentation.text_ids()],
                dtype=torch.long,
                device=encoder.model.embed_tokens.weight.device,
            )
            states: Any = encoder(tokens)
            return states

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
    def predict_data_velocity(
        self,
        run: H3Run,
        text_states: Any,
        video_latents: Any,
        audio_latents: Any,
        modulation: Modulation,
    ) -> tuple[Any, Any]:
        import torch

        with torch.inference_mode():
            return _predict_data_velocity(
                self.pipe.transformer,
                run=run,
                text_states=text_states,
                video_latents=video_latents,
                audio_latents=audio_latents,
                modulation=modulation,
                graph=self.pipe.config.graph,
            )


class Ref2VAModel(_H3Base, task="ref2va"):
    """Ordered image/video/audio references, optionally with keyframes when the deployment
    opens that door. Binds the `transformer_ref` role."""

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(H3Pipeline, factory=build_ref2va_pipeline)
        self.tokenizer = Tokenizer()

    @uses_components("transformer_ref")
    def predict_data_velocity(
        self,
        run: H3Run,
        text_states: Any,
        video_latents: Any,
        audio_latents: Any,
        modulation: Modulation,
    ) -> tuple[Any, Any]:
        import torch

        with torch.inference_mode():
            return _predict_data_velocity(
                self.pipe.transformer,
                run=run,
                text_states=text_states,
                video_latents=video_latents,
                audio_latents=audio_latents,
                modulation=modulation,
                graph=self.pipe.config.graph,
            )


#: What the module functions accept. NOT `_H3Base`: the base deliberately has no
#: `predict_data_velocity`, because a method on it would have to declare a component set,
#: and the whole point is that the set is the one thing the two roles do not share. An
#: undecorated one on the base would declare ALL components by omission — the coarse
#: contract this endpoint exists to avoid — so the union is the honest type.
H3Model = Fl2VAModel | Ref2VAModel


def _predict_data_velocity(
    transformer: Any,
    *,
    run: H3Run,
    text_states: Any,
    video_latents: Any,
    audio_latents: Any,
    modulation: Modulation,
    graph: GraphDialect,
) -> tuple[Any, Any]:
    """THE MODEL BOUNDARY (#524): latents in, LATENT-SHAPED DATA-WARD VELOCITY out.

    It owns the whole row-shaped middle — the in-projections, the pack, the forward and
    the unpack — so nothing row-shaped crosses back into endpoint orchestration. The
    previous shape leaked all of it: a module-level `_pack` assembled rows OUTSIDE any
    declared component scope, could not have worked (text is 5120 wide, a video patch row
    96 and an audio row 32, and `torch.cat` refuses that), and would have had to reach the
    transformer's projections from outside its lease to fix. The solver then received ROWS
    and multiplied them by a sigma delta as if they were latents.

    Input latents are `[1, 24, t, h, w]` and `[1, 32, 2, audio_t]`; the returned
    velocities have exactly those shapes. It takes the transformer as an ARGUMENT: which
    one it is was decided by the caller's declared component set, and this function has no
    opinion about it.

    DATA-WARD, and the heads are RAW. See `dit.MiniMaxH3Dit.forward` and
    `layout.H3Solver` — both state the same convention, and this is the seam between them.
    """
    import torch

    from h3_arch.dit import pack_audio, patchify_video, unpack_audio, unpatchify_video

    if graph is GraphDialect.DIFFUSERS:
        return _predict_data_velocity_ref(
            transformer,
            run=run,
            text_states=text_states,
            video_latents=video_latents,
            audio_latents=audio_latents,
            modulation=modulation,
        )

    grid = run.plan.grid
    patch = transformer.config.patch_size
    weight = transformer.condition_proj.weight
    device, hidden_dtype = weight.device, weight.dtype

    text_rows = transformer.refine_text(text_states[0].to(device=device, dtype=hidden_dtype))
    video_rows = transformer.video_patch_proj(
        patchify_video(video_latents, patch).to(device=device, dtype=torch.float32)
    ).to(hidden_dtype)
    audio_rows = transformer.audio_patch_proj(
        pack_audio(audio_latents).to(device=device, dtype=torch.float32)
    ).to(hidden_dtype)

    rows: list[Any] = []
    for _, _, kind in run.layout.segments:
        if kind == "text":
            rows.append(text_rows)
        elif kind in ("cond", "ref_img"):
            rows.append(run.conditioning["visual"])
        elif kind == "ref_audio":
            rows.append(run.conditioning["audio"])
        elif kind == "video":
            rows.append(video_rows)
        else:
            rows.append(audio_rows)

    video_out, audio_out = transformer(
        torch.cat(rows, dim=0),
        torch.tensor(modulation.timesteps, dtype=torch.float32, device=device),
        list(modulation.segments),
        torch.tensor(modulation.position_ids, dtype=torch.float64, device=device),
        modulation.video_stream,
        modulation.audio_stream,
    )
    return (
        unpatchify_video(
            video_out,
            grid.latent_t // patch[0],
            grid.latent_h // patch[1],
            grid.latent_w // patch[2],
            transformer.config.latents_dim,
            patch,
        ).to(video_latents.dtype),
        unpack_audio(audio_out).to(audio_latents.dtype),
    )


def _predict_data_velocity_ref(
    transformer: Any,
    *,
    run: H3Run,
    text_states: Any,
    video_latents: Any,
    audio_latents: Any,
    modulation: Modulation,
) -> tuple[Any, Any]:
    """THE SAME BOUNDARY, over upstream's `MiniMaxH3Transformer3DModel`.

    The two transformers take the packed sequence apart differently and this is the whole of
    that difference. The port takes ONE already-concatenated `[S, hidden]` buffer plus a
    per-SEGMENT modulation table; upstream takes the three modalities SEPARATELY plus
    per-ROW index tensors, and scatters them into the buffer itself. Both describe the same
    layout — `PackedLayout` — so the translation is arithmetic on the segment table and
    invents nothing:

        row -> modulation row       (the port's `Modulation.segments`)
        row -> timestep index       row // MODALITY_NUM
        row -> modality tag         row %  MODALITY_NUM

    which is exactly the relation upstream builds in the other direction
    (`adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags`).

    CONDITIONING ROWS ARE NOT HANDLED HERE and do not silently vanish: upstream's
    `hidden_states` must carry the conditioning video rows interleaved in `video_indices`
    order, and building that is the same unbuilt pixels-to-rows seam `condition_text`
    refuses on (`vision_seam_unbuilt`). A layout that carries one reaches an explicit
    refusal rather than a forward that quietly drops it.

    The heads are RAW and DATA-WARD on this side too — upstream's own
    `MiniMaxH3Scheduler` states the convention — so `H3Solver` consumes them unchanged.
    """
    import torch

    from h3_arch.dit import pack_audio, patchify_video, unpack_audio, unpatchify_video

    modality_num = 3
    grid = run.plan.grid
    patch = transformer.config.patch_size
    weight = transformer.context_embedder.weight
    device = weight.device

    conditioning = [k for _, _, k in run.layout.segments if k in ("cond", "ref_img", "ref_audio")]
    if conditioning:
        raise UnsupportedInput(
            "the reference construction layer packs conditioning rows into the modality "
            f"streams itself, and this layout carries {', '.join(sorted(set(conditioning)))} "
            "rows the endpoint has no builder for: the pixels-to-rows seam is the same one "
            "`condition_text` refuses on",
            code="vision_seam_unbuilt",
            fields=["first_frame", "last_frame", "references"],
        )

    seq_len = run.layout.seq_len
    timestep_indices = torch.zeros(seq_len, dtype=torch.long)
    token_tags = torch.zeros(seq_len, dtype=torch.long)
    for start, end, row in modulation.segments:
        timestep_indices[start:end] = row // modality_num
        token_tags[start:end] = row % modality_num

    def _span(kind: str) -> Any:
        a, b = run.layout.stream(kind)
        return torch.arange(a, b, dtype=torch.long, device=device)

    video_out, audio_out = transformer(
        patchify_video(video_latents, patch).unsqueeze(0).to(device=device),
        pack_audio(audio_latents).unsqueeze(0).to(device=device),
        text_states.to(device=device),
        torch.tensor(modulation.timesteps, dtype=torch.float32, device=device),
        timestep_indices.to(device),
        token_tags.to(device),
        torch.tensor(modulation.position_ids, dtype=torch.float32, device=device),
        _span("video"),
        _span("audio"),
        _span("text"),
        return_dict=False,
    )
    return (
        unpatchify_video(
            video_out[0],
            grid.latent_t // patch[0],
            grid.latent_h // patch[1],
            grid.latent_w // patch[2],
            transformer.config.in_channels,
            patch,
        ).to(video_latents.dtype),
        unpack_audio(audio_out[0]).to(audio_latents.dtype),
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
        evaluations=plan.steps,
        layout=layout,
        sigma_shift_video=config.sigma_shift_video,
        sigma_shift_audio=config.sigma_shift_audio,
        visual_cond_timestep=0.999 if plan.keyframes or plan.references else None,
        audio_cond_timestep=1.0 if any(b.ref_audio_t for b in ref_blocks) else None,
        adapters=tuple(str(a) for a in view.adapters),
    )
    solver = H3Solver(timestep_plan)
    run = H3Run(plan, layout, timestep_plan, view, presentation)
    tel.metric("packed_rows", float(layout.seq_len))
    # BOTH numbers, named apart. One request's plan holds `grid_points` sigmas and costs
    # `evaluations` forward passes, and they are never the same integer (#522c).
    tel.metric("evaluations", float(solver.evaluations))
    tel.metric("sigma_grid_points", float(timestep_plan.grid_points))
    ctx.raise_if_cancelled()

    with tel.stage("condition_text"):
        text_states = model.condition_text(presentation)
    if plan.keyframes or plan.references:
        with tel.stage("condition_visual"):
            run.conditioning["visual"] = model.condition_visual(run.conditioning.get("pixels"))
    if any(b.ref_audio_t for b in ref_blocks):
        with tel.stage("condition_audio"):
            run.conditioning["audio"] = model.condition_audio(run.conditioning.get("waveform"))

    on_step = tel.step_callback(solver.evaluations, stage="denoise")
    with tel.stage("denoise"):
        latents = _sample(torch, model, run, solver, text_states, on_step, ctx)

    with tel.stage("decode_audio"):
        waveform = model.decode_audio(latents["audio"])
        # The codec plane takes [S], [1, S] or [2, S] and refuses a leading batch dim by
        # design (cr-017); the audio VAE returns [1, C, S]. Dropping the batch dim is the
        # caller's job, and this is the caller.
        if waveform.ndim == 3:
            waveform = waveform[0]
        waveform = waveform.to(torch.float32)
    with tel.stage("decode_video"):
        decoded = model.decode_video(latents["video"])

    # ONE conversion. The VAE already returned [0, 1]; see `h3_arch.pixels`.
    frames = pixel_bytes(decoded[0]).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
    facts = MediaFacts(
        width=plan.grid.width,
        height=plan.grid.height,
        frames=plan.grid.frames,
        fps=FPS,
        sample_rate=model.pipe.config.audio_vae.sample_rate,
        mute=plan.mute,
    )
    with tel.stage("gate_pre_encode"):
        pre_encode_gate(
            torch, decoded=decoded, pixels=frames, waveform=waveform, requested=facts, tel=tel
        )
    rgb = bytes(frames.cpu().numpy().tobytes())
    samples = bytes(waveform.cpu().numpy().tobytes())

    with tel.stage("encode_mp4"):
        asset = out.save_video(
            frames,
            fps=FPS,
            audio=None if plan.mute else waveform,
            sample_rate=facts.sample_rate,
        )
    with tel.stage("gate_post_encode"):
        # THE CONTAINER IS WHAT A CALLER OPENS, so the container is what gets decoded. The
        # bytes come back through the asset's own reader; the spool file is the only way to
        # hand a path to ffprobe and dies with the attempt.
        probe = out.temporary_file(".mp4")
        probe.write_bytes(asset.read_bytes())
        post_encode_gate(probe, requested=facts, tel=tel)

    return GenerateOutput(
        video=asset,
        width=facts.width,
        height=facts.height,
        frames=facts.frames,
        fps=facts.fps,
        steps=solver.evaluations,
        plan_digest=run.digest,
        video_digest=hashlib.sha256(rgb).hexdigest(),
        audio_digest=hashlib.sha256(samples).hexdigest(),
        checkpoint=model.checkpoint_ref,
    )


def _sample(
    torch: Any,
    model: H3Model,
    run: H3Run,
    solver: H3Solver,
    text_states: Any,
    on_step: Any,
    ctx: Context,
) -> dict[str, Any]:
    """The joint video+audio sampling loop. ONE model evaluation per step — no negative
    branch, no zero-delta tail, and the sign is the solver's.

    ORCHESTRATION ONLY: it holds latents, asks the boundary for their data-ward velocity,
    and lets `SolverStep` apply it. Nothing row-shaped appears in this function, which is
    the whole point of the boundary (#524) — the previous shape assembled transformer rows
    here, outside every declared component scope, and then multiplied ROWS by a sigma
    delta as if they were latents."""
    plan = run.timestep_plan
    dit = model.pipe.config.dit
    grid = run.plan.grid
    generator = torch.Generator(device="cpu").manual_seed(run.view._seed)
    # SEEDED ON THE HOST, always: a device-side draw is not reproducible across cards, and
    # the seed is a request fact. The latents then follow the conditioning to wherever the
    # runtime placed the weights.
    video = torch.randn(
        1, dit.latents_dim, grid.latent_t, grid.latent_h, grid.latent_w, generator=generator
    ).to(text_states.device)
    audio = torch.randn(
        1, dit.audio_latents_dim, 2, grid.audio_t, generator=generator
    ).to(text_states.device)

    for step in solver.steps():
        ctx.raise_if_cancelled()
        modulation = build_modulation(
            run.layout,
            t_video=step.t_video,
            t_audio=step.t_audio,
            visual_cond_t=plan.visual_cond_timestep or 0.0,
            audio_cond_t=plan.audio_cond_timestep or 0.0,
            text_token_tags=run.presentation.tags,
        )
        v_video, v_audio = model.predict_data_velocity(
            run, text_states, video, audio, modulation
        )
        video = step.advance_video(video, v_video)
        audio = step.advance_audio(audio, v_audio)
        # ONE progress advance per model evaluation. It used to fire per transformer BLOCK
        # as well, so a 30-step request reported 1530 advances (#522e).
        on_step(step.index)
    return {"video": video, "audio": audio}


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
