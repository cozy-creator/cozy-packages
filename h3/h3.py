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
safetensors header for all five components at zero cost (`scripts/h3-keys.py`). The
reference semantics this file's sampler depends on — the data-ward sign, the 17k+5 temporal
geometry, the sigma grid and its evaluation count, the pixel conversion, the
requested-vs-observed media agreement — are CONFORMANCE-PROVEN ON CPU against
independently written upstream expressions (`scripts/h3-conform.py`, which runs in CI).
The output gates are proven against digest-pinned real media, including the failed
`daf50552…` render as a permanent red arm (`scripts/h3-live.py gates`).

THE WHOLE-SEAM ORACLE HAS NOW RUN (2026-08-26, one H200; `scripts/h3-seam-oracle.py`
carries the table). Against ComfyUI v0.33.0 on the same bf16 carriers, the same prompt and
the same seed, every seam agrees at cosine >= 0.9993 — token ids identical, the text
encoder EXACT at float32 — and a full-length 15.083 s / 362-frame / 30-evaluation render was
produced and VIEWED: coherent motion matching the prompt, an audio track that follows the
picture, adjacent-frame correlation 0.9882 against cozy-eval's 0.6 floor. The endpoint's
model path is OUTPUT-VERIFIED.

WHAT THAT DOES NOT COVER, said plainly. This file's SERVING path — the fill plane, the
stamp plane, the residency ladder — is a separate subject with its own open defects (#529's
seven), and the oracle drove the model path directly. Two artifacts remain in the picture,
a 32-pixel spatial lattice and a 17-frame temporal seam, and both are reproduced identically
by upstream on the same weights: they are the released carrier's, and they are the
optimization lane's quality question (#531).
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
from h3_arch.vision import ExpandedPresentation, PatchedVision

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
    components. A slot exposes its own transformer component and the three shared ones; it
    never exposes the twin's, which is what makes an undeclared access spellable.

    TWO CONSTRUCTION LAYERS, SELECTED BY THE ARTIFACT (#531/#540). `h3_arch` is the hand
    port and answers to the community curve carrier's key set; `h3_ref` is a thin layer over
    upstream's official classes and answers to the official diffusers-format tree's. The
    artifact's own config says which, through `H3Config.graph`, and nothing else in this file
    knows the difference — `build_component(name, ...)` is the same signature on both sides,
    which is the whole reason the rebase can be a construction decision rather than a fork.
    """

    def __init__(self, config: Any, *, transformer_component: str) -> None:
        whole = H3Config.from_mapping(config.mapping())
        self.config = whole
        self.transformer_component = transformer_component
        components = (transformer_component, "text_encoder", "video_vae", "audio_vae")
        self.components: dict[str, Any]
        if whole.graph is GraphDialect.DIFFUSERS:
            import h3_ref

            mapping = h3_ref.config_mapping(whole)
            self.components = {c: h3_ref.build_component(c, mapping) for c in components}
        else:
            self.components = {c: build_component(c, whole) for c in components}

    @property
    def transformer(self) -> Any:
        return self.components[self.transformer_component]


def build_fl2va_pipeline(config: Any) -> H3Pipeline:
    return H3Pipeline(config, transformer_component="transformer")


def build_ref2va_pipeline(config: Any) -> H3Pipeline:
    return H3Pipeline(config, transformer_component="transformer_ref")


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
    condition_images: tuple[Any, ...] = ()
    """Every visually-conditioning image IN PACKED ORDER — the keyframes first, then the
    references — which is the order `PackedLayout` appends their `cond` and `ref_img`
    segments in. One list, because the DiT does not distinguish them when it reads rows."""
    condition_geometry: tuple[tuple[int, int, int], ...] = ()
    """The `(latent_t, latent_h, latent_w)` the layout reserved rows for, per condition."""


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
    expanded: ExpandedPresentation
    """The presentation with its vision blocks expanded into real pad runs. THIS is the
    sequence the text encoder ran on and the one `PackedLayout`'s text span was sized from;
    `presentation.tags` carries one tag per PRESENTATION row and is one row per block."""
    patched: PatchedVision
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
) -> tuple[H3Plan, PackedLayout, Presentation, ExpandedPresentation, PatchedVision]:
    """CPU media preparation and the exact request plan. No component is touched.

    THE PATCHIFICATION HAPPENS HERE, not at conditioning time, and that placement is the
    whole reason the layout can be right. A vision block stands in the presentation as ONE
    row and becomes thousands of them, and `PackedLayout`'s text length plus the DiT's
    per-row modality tags both have to be the EXPANDED count. Resolving it later would
    build the packed sequence against a text span that does not exist. It is pure CPU work
    with no weights in it, so it belongs in the stage that has no component lease.
    """
    from h3_arch import vision

    width, height = _PIXELS[payload.aspect_ratio]
    frames = _FRAMES[payload.duration_s]
    grid = latent_grid(frames, width, height)
    keyframes: list[PackedKeyframe] = []
    if payload.first_frame is not None:
        keyframes.append(PackedKeyframe(0))
    if payload.last_frame is not None:
        keyframes.append(PackedKeyframe(frames - 1))
    keyframe_pixels = _present_keyframes(payload, width=width, height=height)
    presentation = build_presentation(
        tokenizer,
        payload.prompt,
        keyframes=keyframe_pixels,
        references=references,
    )
    patched = vision.patchify(presentation)
    expanded = vision.expand(presentation, patched.token_counts)
    # PACKED ORDER, and it is the layout's order rather than the request's: `PackedLayout`
    # appends every keyframe `cond` segment before the first `ref_img` one, so the
    # conditioning rows have to be produced in that order too or each one lands on
    # another's segment.
    condition_images = (*keyframe_pixels, *(r.pixels for r in references if r.kind == "image"))
    condition_geometry = tuple(
        vision.reference_block_geometry(image) for image in condition_images
    )
    plan = H3Plan(
        prompt=payload.prompt,
        grid=grid,
        keyframes=tuple(keyframes),
        references=references,
        ref_blocks=ref_blocks,
        steps=payload.num_inference_steps,
        mute=payload.mute,
        condition_images=condition_images,
        condition_geometry=condition_geometry,
    )
    layout = PackedLayout(
        len(expanded.token_ids), grid, keyframes=plan.keyframes, refs=ref_blocks
    )
    return plan, layout, presentation, expanded, patched


def _present_keyframes(
    payload: GenerateInput | RefGenerateInput, *, width: int, height: int
) -> tuple[Any, ...]:
    """The first/last keyframes as pixels on the TARGET canvas.

    A keyframe is not a reference and does not take the reference rule: it is an anchor on
    the generated clip's own clock, so it is put on the request's canvas rather than on a
    2048 short edge of its own. Upstream stretches the geometry anchor onto the canvas and
    cover-crops the follower; that asymmetry is reproduced, with the rounding upstream
    documents `VaeImageProcessor` does NOT do.
    """
    from PIL import Image

    named = (("first_frame", payload.first_frame), ("last_frame", payload.last_frame))
    assets = [(name, a) for name, a in named if a is not None]
    out: list[Any] = []
    for position, (name, asset) in enumerate(assets):
        image = _decode_image(asset, where=name)
        if position == 0:
            out.append(image.resize((width, height), Image.Resampling.LANCZOS))
            continue
        scale = max(width / image.size[0], height / image.size[1])
        sized = (round(image.size[0] * scale), round(image.size[1] * scale))
        left = max(0, (sized[0] - width) // 2)
        top = max(0, (sized[1] - height) // 2)
        resized = image.resize(sized, Image.Resampling.LANCZOS)
        out.append(resized.crop((left, top, left + width, top + height)))
    return tuple(out)


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
    return tuple(
        _present(ref, where=f"references.{i}") for i, ref in enumerate(payload.references)
    )


def _reference_blocks(references: tuple[PresentedReference, ...]) -> tuple[RefBlock, ...]:
    """Each presented reference's PACKED contribution, with its real latent geometry.

    This used to be `RefBlock(kind="image")` for every reference — zero latent extents, so
    `_frame_grid(0, 0)` gave every reference ZERO rows and the packed layout reserved
    nothing for any of them. A ref2va request built a layout identical to a t2va one and
    nothing said so.

    The geometry is DERIVED from the normalized canvas rather than measured off the encode,
    because the layout is built before any component is leased. `condition_visual` checks
    the encode against it.
    """
    from h3_arch.vision import reference_block_geometry

    blocks: list[RefBlock] = []
    for ref in references:
        if ref.kind != "image":
            # Audio references need their waveform decoded and encoded by the audio VAE
            # before `ref_audio_t` can be stated, and that decode is not built. Refused
            # here rather than packed as a zero-row block that silently conditions nothing.
            raise UnsupportedInput(
                f"{ref.kind} references are not packed by this endpoint yet: image "
                "references are the built path, and an audio reference needs its waveform "
                "decoded and encoded before its packed rows can be reserved",
                code="audio_reference_unpacked",
                fields=["references"],
            )
        latent_t, latent_h, latent_w = reference_block_geometry(ref.pixels)
        blocks.append(
            RefBlock(kind="image", latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
        )
    return tuple(blocks)


def _kind_of(ref: Reference) -> ReferenceKind:
    if isinstance(ref, ImageReference):
        return "image"
    return "video" if isinstance(ref, VideoReference) else "audio"


def _decode_image(asset: Any, *, where: str) -> Any:
    """A hydrated image asset -> an RGB `PIL.Image`. An asset that is not a decodable image
    is a typed REQUEST refusal, not a backend fault: the caller sent it."""
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(asset.read_bytes()))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidRequest(
            f"{where} is not a decodable image ({type(exc).__name__}): {str(exc)[:120]}",
            code="undecodable_image",
            fields=[where],
        ) from exc
    return image.convert("RGB")


def _present(ref: Reference, *, where: str) -> PresentedReference:
    """One wire reference -> its presentation form, with the pixels ATTACHED.

    The pixels used to be dropped here with a note saying the condition operation would
    attach them, and nothing ever did — which is one half of what the old seam-wide refusal
    was standing in front of.
    """
    from h3_arch.vision import normalize_reference_image

    kind = _kind_of(ref)
    if kind == "audio":
        # A WAVEFORM NEVER ENTERS QWEN. The presentation gets the label; the audio VAE
        # gets the samples. Upstream refused audio as the only modality; that guard was
        # measured to protect nothing and this endpoint carries the Cozy extension
        # instead, WITHOUT claiming viseme or beat synchronization.
        return PresentedReference(kind="audio", has_audio=True)
    if kind == "video":
        # The frames would have to be decoded, resampled onto the 24 fps clock and then
        # sampled again onto the text encoder's 2 fps grid, and the soundtrack encoded
        # beside them. The presentation layer builds video blocks and the packed layout
        # lays them out; what has no implementation is the DECODE. Refused typed and
        # narrowly, rather than under the seam-wide code that no longer applies.
        raise UnsupportedInput(
            "video references are not decoded by this endpoint yet: image references are "
            "the built path, and a video reference needs its own frame decode, its 24 fps "
            "resample and its soundtrack before it can be presented",
            code="video_reference_undecoded",
            fields=[where],
        )
    try:
        pixels = normalize_reference_image(_decode_image(ref.image, where=where))
    except ValueError as exc:
        raise InvalidRequest(str(exc), code="reference_geometry", fields=[where]) from exc
    return PresentedReference(kind="image", pixels=pixels)


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
    def condition_text(
        self, expanded: ExpandedPresentation, patched: PatchedVision
    ) -> Any:
        """Qwen3-VL over the expanded presentation: the UNNORMALIZED hidden state after
        layer 50, `[1, L, text_dim]`.

        THE VISION SEAM IS BUILT HERE. The text encoder takes `[batch, seq]` token ids plus
        already-patchified `VisionBlock(patches, grid_thw, index)` splices; `prepare` has
        produced both, from upstream's own processor. Every vision block's pad run is
        already in `expanded.token_ids` at `expanded.splices[i]`, so the splice is an
        overwrite of rows that exist rather than an insertion that would move the layout
        out from under the packed sequence.
        """
        import torch

        from h3_arch import vision

        encoder = self.pipe.components["text_encoder"]
        device = encoder.model.embed_tokens.weight.device
        blocks = vision.text_encoder_blocks(patched, expanded)
        with torch.inference_mode():
            tokens = torch.tensor([expanded.token_ids], dtype=torch.long, device=device)
            placed = [
                type(b)(
                    patches=b.patches.to(device=device),
                    grid_thw=b.grid_thw.to(device=device),
                    index=b.index,
                )
                for b in blocks
            ]
            states: Any = encoder(tokens, vision=placed)
            return states

    @uses_components("video_vae")
    def condition_visual(self, run: H3Run, *, noise_level: float) -> list[Any]:
        """Keyframe and visual-reference CONDITIONING latents, one per packed block.

        Separate from the presentation: the same image reaches Qwen as a vision block and
        the DiT as VAE rows, and both are needed — the vision tokens carry semantics and
        these rows carry the pixels.

        A LIST, one entry per block, never one tensor for all of them. Each block is its own
        geometry and lands at its own segment; the previous shape handed a single tensor to
        every `cond`/`ref_img` slot at once, which cannot be right for more than one
        reference and was never populated anyway.

        The anchors are noised ONCE, here, to `noise_level` and held there for the whole
        loop — they are not on the sampler's schedule. `layout.build_modulation` pins their
        rows to the same level, so this and the modulation state the same fact.
        """
        import torch

        from h3_arch.vision import condition_pixels

        vae = self.pipe.components["video_vae"]
        device = vae.post_quant_conv.weight.device
        out: list[Any] = []
        with torch.inference_mode():
            for position, image in enumerate(run.plan.condition_images):
                latents = vae.encode_condition(condition_pixels(image).to(device))
                expected = run.plan.condition_geometry[position]
                got = tuple(int(n) for n in latents.shape[2:5])
                if got != expected:
                    raise RuntimeError(
                        f"conditioning block {position} encoded to {got} latents and the "
                        f"packed layout reserved rows for {expected}: the derived geometry "
                        "and the encode disagree, so the packed sequence would misalign"
                    )
                # x_t = t*x_0 + (1-t)*noise, in H3's `t` convention — `t = 1` is clean. The
                # noise is drawn on the HOST so two cards produce the same anchor.
                generator = torch.Generator(device="cpu").manual_seed(run.view._seed + position)
                noise = torch.randn(
                    latents.shape, generator=generator, dtype=torch.float32
                ).to(device=latents.device, dtype=latents.dtype)
                out.append(noise_level * latents + (1.0 - noise_level) * noise)
        return out

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
    """Text and first/last keyframes. Binds the `transformer` component."""

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
    opens that door. Binds the `transformer_ref` component."""

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

    # THE CONDITIONING ROWS, one condition per `cond`/`ref_img` segment IN ORDER. The
    # previous shape appended `run.conditioning["visual"]` — a single tensor — to every such
    # slot at once, which cannot be right for more than one condition and was never
    # populated at all. `condition_visual` produces the list; it is consumed as an iterator
    # so a count mismatch is a refusal rather than a silently reused tensor.
    conditions = iter(run.conditioning.get("visual") or ())

    def _condition_rows(kind: str, count: int) -> Any:
        try:
            latents = next(conditions)
        except StopIteration:
            raise RuntimeError(
                f"the packed layout carries a {kind!r} segment with no conditioning latents "
                "left to fill it: the plan's condition list and the layout's segments "
                "disagree, and the sequence would be built from a reused tensor"
            ) from None
        projected = transformer.video_patch_proj(
            patchify_video(latents, patch).to(device=device, dtype=torch.float32)
        ).to(hidden_dtype)
        if projected.shape[0] != count:
            raise RuntimeError(
                f"a {kind!r} condition projected to {projected.shape[0]} rows and the "
                f"layout reserved {count}"
            )
        return projected

    rows: list[Any] = []
    for start, end, kind in run.layout.segments:
        if kind == "text":
            rows.append(text_rows)
        elif kind in ("cond", "ref_img"):
            rows.append(_condition_rows(kind, end - start))
        elif kind == "ref_audio":
            rows.append(run.conditioning["audio"])
        elif kind == "video":
            rows.append(video_rows)
        else:
            rows.append(audio_rows)
    if next(conditions, None) is not None:
        raise RuntimeError(
            "more conditioning latents were encoded than the packed layout reserved "
            "segments for — the plan and the layout disagree about what this request is"
        )

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
    order, and this endpoint has no builder for them. That is a DIFFERENT seam from the one
    `condition_text` used to refuse on, and it is now the only one left: the vision seam
    (pixels to patches, grid and splice index) is built, so a reference reaches Qwen3-VL
    correctly and reaches the DiT's LATENT rows not at all. A layout that carries one
    reaches an explicit refusal rather than a forward that quietly drops it.

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
            "rows this DIALECT has no builder for. The port dialect builds them; upstream's "
            "transformer wants them interleaved into `hidden_states` in `video_indices` "
            "order instead, which is a different assembly and is unbuilt — and no "
            "diffusers-format artifact is bound yet for it to run against",
            code="conditioning_rows_unbuilt_diffusers",
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
        plan, layout, presentation, expanded, patched = prepare(
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
    run = H3Run(plan, layout, timestep_plan, view, presentation, expanded, patched)
    tel.metric("packed_rows", float(layout.seq_len))
    tel.metric("text_rows", float(len(expanded.token_ids)))
    tel.metric("vision_blocks", float(len(patched.token_counts)))
    # BOTH numbers, named apart. One request's plan holds `grid_points` sigmas and costs
    # `evaluations` forward passes, and they are never the same integer (#522c).
    tel.metric("evaluations", float(solver.evaluations))
    tel.metric("sigma_grid_points", float(timestep_plan.grid_points))
    ctx.raise_if_cancelled()

    with tel.stage("condition_text"):
        text_states = model.condition_text(expanded, patched)
    if plan.condition_images:
        with tel.stage("condition_visual"):
            run.conditioning["visual"] = model.condition_visual(
                run, noise_level=timestep_plan.visual_cond_timestep or 1.0
            )
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
            # THE EXPANDED TAGS, one per text-encoder row. `presentation.tags` carries one
            # tag per PRESENTATION row, where a whole vision block is a single row, so
            # using it would tag the text span by a length the sequence does not have.
            text_token_tags=run.expanded.tags,
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


# HIDDEN, and now structurally so (#572d). Ref2VA's vision-conditioning seam is UNBUILT
# (#529/#539f): this surface has never produced a frame, and until #558e it was "hidden" only
# in a tracker row. A deployment therefore staged its binding like any other, its construction
# refused, and — before prepare became per-binding — it took the working T2VA sibling down
# with it on a rented H200 with 92.6 GiB already resident.
#
# `hidden=True` keeps the surface in the descriptor, because it is real code with a real
# signature and the descriptor must not lie about the release, while excluding it from the
# serving set: no binding is staged and no request lands. It comes OFF the day the vision
# seam renders something a person has looked at.
@app.entrypoint(hidden=True)
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
    blocks = _reference_blocks(references)
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
