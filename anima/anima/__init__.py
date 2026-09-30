"""Anima text-to-image through Diffusers' maintained modular pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import Enum, IntEnum
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, NamedTuple

import cozy_runtime.author as cozy_author
import cozy_runtime.derive as derive
import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    Loader,
    Model,
    ModelArtifact,
    ModelDefault,
    Outputs,
    Shape,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    invocable,
    uses_components,
)
from diffusers import (
    AnimaAutoBlocks,
    AnimaModularPipeline,
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.first_block_cache import (
    _FBC_BLOCK_HOOK,
    _FBC_LEADER_BLOCK_HOOK,
    FirstBlockCacheConfig,
    apply_first_block_cache,
)
from diffusers.hooks.hooks import HookRegistry
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock
from diffusers.modular_pipelines import ModularPipelineBlocks
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
from transformers import initialization as transformer_init

app = App()
_MODULE_ROOT = Path(__file__).resolve().parent
_ROOT = _MODULE_ROOT / "anima_assets" if (_MODULE_ROOT / "anima_assets").is_dir() else _MODULE_ROOT


class AspectRatio(Enum):
    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    WIDE = "16:9"
    PORTRAIT = "3:4"
    TALL = "9:16"


class Megapixels(IntEnum):
    """Anima's native resolution classes, in nominal megapixels (se-024).

    Tier 2 is the 1536-class (~2.3 MP) the model is meant to be run at and this
    package's default; tier 1 trades resolution for speed. Denoise attention is full
    (no windowing), so time and VRAM scale roughly linearly with area — tier 1 costs
    about half of tier 2. The VAE always tiles. Every bucket is a multiple of the
    pipeline's 16-px stride.
    """

    MP1 = 1
    MP2 = 2


#: (aspect, tier) -> (width, height): tier 1 is the ~1 MP training set, tier 2 scales it
#: by exactly 1.5 to the 1536 class. A pair absent here does not round — it refuses at
#: decode (today the grid is complete, so only an out-of-enum value can refuse).
_BUCKETS: dict[tuple[AspectRatio, Megapixels], tuple[int, int]] = {
    (AspectRatio.SQUARE, Megapixels.MP1): (1024, 1024),
    (AspectRatio.LANDSCAPE, Megapixels.MP1): (1152, 896),
    (AspectRatio.WIDE, Megapixels.MP1): (1344, 768),
    (AspectRatio.PORTRAIT, Megapixels.MP1): (896, 1152),
    (AspectRatio.TALL, Megapixels.MP1): (768, 1344),
    (AspectRatio.SQUARE, Megapixels.MP2): (1536, 1536),
    (AspectRatio.LANDSCAPE, Megapixels.MP2): (1728, 1344),
    (AspectRatio.WIDE, Megapixels.MP2): (2016, 1152),
    (AspectRatio.PORTRAIT, Megapixels.MP2): (1344, 1728),
    (AspectRatio.TALL, Megapixels.MP2): (1152, 2016),
}

#: The demand table `Shape` reads: the runtime derives (width, height, pixels) from ONE
#: field, so the tier carries its largest bucket as an upper bound over the tier.
_TIER_DEMAND: dict[Megapixels, tuple[int, int]] = {
    tier: max(
        (size for (_, t), size in _BUCKETS.items() if t is tier),
        key=lambda size: size[0] * size[1],
    )
    for tier in Megapixels
}
_WEBP_OUTPUT = AssetBound(max_bytes=64 << 20, media_types=("image/webp",))


class _Phase(NamedTuple):
    """One render phase: the stage name a person reads, and its overall-progress span."""

    stage: str
    start: float
    stop: float

    @property
    def bounds(self) -> tuple[float, float]:
        return self.start, self.stop


#: The render ladder. `cozy run list` renders "<stage> <percent>" verbatim, so these names
#: are read by a person; the spans are contiguous and cover the whole request. Only
#: `denoise` is measured per step — it is the only phase long enough to need it, and at the
#: default 1536 class it is 150 of the roughly 160 seconds.
_ENCODE_PROMPT = _Phase("encoding prompt", 0.00, 0.06)
_CONDITION = _Phase("conditioning", 0.06, 0.10)
_DENOISE = _Phase("denoise", 0.10, 0.90)
_DECODE = _Phase("decoding", 0.90, 0.98)
_SAVE = _Phase("saving image", 0.98, 1.00)

#: The official Anima model card's negative prompt, verbatim (se-026). Owner-labelled bank
#: 2026-09-02: with seed, steps, guidance and geometry fixed, THIS string is the quality
#: knob — Anima's score-bucket conditioning acts through the negative, so pushing away
#: score_1/score_2/score_3 (plus jpeg artifacts, chromatic aberration, artist name — a
#: literal tag that suppresses signature text) moves output off the model's flat dated
#: aesthetic. The negative is the ISOLATED cause: the labelled loser carried `score_7, safe`
#: in its positive prompt and still lost, and one labelled winner carried no positive quality
#: prefix at all — so `quality_prefix` below reproduces the judged configuration but is not
#: what carries the quality. Changing this string changes every render that omits the field.
CARD_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, "
    "artist name, blurry, jpeg artifacts, chromatic aberration"
)

#: The card's human-scored quality prefix, prepended to the caller's prompt (se-026). Every
#: owner-approved image in the 2026-09-02 bank carried this text, so it is what a bare prompt
#: has to reproduce to land in the configuration that was actually judged. It is a DEFAULTED
#: FIELD, not a hidden injection: `quality_prefix=""` turns it off, and a caller whose intent
#: it fights ("crayon drawing, childlike, naive") can say so. No rating tag rides along —
#: se-026 proposed `safe` and the owner declined it (2026-09-03), so a neutral prompt can
#: return NSFW output by design.
CARD_QUALITY_PREFIX = "masterpiece, best quality, "


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    quality_prefix: str = CARD_QUALITY_PREFIX
    negative_prompt: str = CARD_NEGATIVE
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP2
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=8, le=50)] = 30
    guidance: Annotated[ModelDefault[float], msgspec.Meta(ge=1.0, le=10.0)] = 4.5
    #: CFG interval (cr-086 arm 1, Kynkäänniemi et al., NeurIPS 2024): guidance helps only
    #: in a middle band of the noise schedule, so the uncond forward is skipped outside
    #: [start, stop) of the step fraction. (0, 1) is full-range CFG. The banked default is
    #: ON (se-026, 2026-09-03): at the default 1536 class it cut the denoise loop from 205 s
    #: to 156 s on an sm89 4070, and the same-seed image was at least as good — the paper's
    #: own claim is that the omitted band costs nothing. Subtractive and bounded: every block
    #: still runs on every step, so its failure mode is uniform rather than structural.
    cfg_interval_start: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.15
    cfg_interval_stop: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.7
    #: First-block cache (cr-086 arm 2, FBCache): when the first transformer block's
    #: residual moves less than this threshold between steps, the remaining 27 blocks are
    #: skipped and the cached tail residual is reused. Cond and uncond passes keep separate
    #: cache states under this package's sequential batch-1 CFG.
    #:
    #: DEFAULT OFF, deliberately (se-026, 2026-09-03). Measured at 0.075 rather than assumed:
    #: it extrapolates 44% of forwards, and against the same seed it consistently smooths
    #: faces and coarsens fine texture — the two places the first block is a bad proxy for
    #: the other 27. It also costs VRAM, because those cached residuals stay live: ALONE at
    #: the default 1536 class it refuses `device_shortfall` on an 8 GiB card, reproducibly.
    #: It survives only when `cfg_interval` happens to be narrowing the live cache states,
    #: and a default that works only because another default masks it is not a default.
    first_block_cache: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0
    seed: int = 1005


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, _WEBP_OUTPUT]
    width: int
    height: int
    steps: int
    guidance: float
    digest: str


def _tokenizer(path: Path) -> Any:
    config = json.loads((path / "tokenizer_config.json").read_text())
    config.pop("tokenizer_class", None)
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), **config
    )


def _declare_vae_retry_state(vae: Any) -> None:
    """Restore AutoencoderKLQwenImage's encode/decode cache lists by identity.

    Whole calls and individual tiles clear caches before reading them. Causal layers
    only replace entries with new tensors; previous tensor contents are read-only.
    Failed-attempt tensors therefore need no copies: restore the original bindings.
    """
    declare = cozy_author.retry_state
    names = ("_conv_num", "_conv_idx", "_feat_map", "_enc_conv_num", "_enc_conv_idx", "_enc_feat_map")

    def save() -> dict[str, tuple[Any, list[Any] | None]]:
        return {
            name: (value, list(value) if isinstance(value, list) else None)
            for name in names if name in vars(vae)
            for value in (vars(vae)[name],)
        }

    def restore(state: dict[str, tuple[Any, list[Any] | None]]) -> None:
        for name in names:
            vars(vae).pop(name, None)
        for name, (value, items) in state.items():
            if items is not None:
                value[:] = items
            vars(vae)[name] = value

    declare(vae, save, restore)


class AnimaPipeline:
    def __init__(self, config: Any) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            transformer = CosmosTransformer3DModel.from_config(mapping["transformer"]).to(
                torch.bfloat16
            )
            text_encoder: Any = Qwen3Model(Qwen3Config(**mapping["text_encoder"]))
            text_encoder.to(dtype=torch.bfloat16)
            text_conditioner = AnimaTextConditioner.from_config(
                mapping["text_conditioner"]
            ).to(torch.bfloat16)
            vae = AutoencoderKLQwenImage.from_config(mapping["vae"]).to(torch.bfloat16)

        for component in (transformer, text_encoder, text_conditioner, vae):
            component.eval()
        # Cosmos' uncached forward (including rotary embeddings) only reads module
        # state and arguments. First-block caching gets a separate request contract
        # for its stateful hooks and forward replacement below.
        cozy_author.pure(transformer)
        cozy_author.pure(text_conditioner)
        # The modular encoder passes no past_key_values: any KV cache is local
        # to this call. Dynamic RoPE rewrites buffers and has no purity contract.
        if getattr(text_encoder.rotary_emb, "rope_type", None) == "default":
            cozy_author.pure(text_encoder)
        _declare_vae_retry_state(vae)
        vae.enable_tiling()
        self.scheduler_config = mapping["scheduler"]
        self.tokenizer = _tokenizer(_ROOT / "tokenizer")
        self.t5_tokenizer = _tokenizer(_ROOT / "t5_tokenizer")
        self.components: dict[str, Any] = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "text_conditioner": text_conditioner,
            "vae": vae,
        }


def warmup() -> None:
    """Runtime's off-lock import hook (cr-104). Every import is at module scope (se-041), so
    `import anima` already paid the cost and nothing is left to defer."""


def build_pipeline(config: Any) -> AnimaPipeline:
    return AnimaPipeline(config)


class AnimaModel(Model[AnimaPipeline], encoded_leaves="accept"):
    """Touches its components through their forward passes only, so an encoded lane may
    replace DiT linears with native leaves (the fp8 lane's whole point on sm89)."""

    pipe: AnimaPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(AnimaPipeline, factory=build_pipeline)

    def warm(self, ctx: Context) -> None:
        """One 512px, one-step render at the card's defaults, so no request pays a
        first-call cost. The runtime calls it once per fill, before the placement serves;
        there is no attempt to meter, so the phases stay silent."""
        card = GenerateInput(prompt="")
        ctx.raise_if_cancelled()
        render_request(
            self,
            card.quality_prefix,
            card.negative_prompt,
            512,
            512,
            1,
            card.guidance,
            (card.cfg_interval_start, card.cfg_interval_stop),
            card.first_block_cache,
            card.seed,
            _Phases(),
        )

    @uses_components("text_encoder", "text_conditioner", "transformer", "vae")
    def prepare_request(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        cfg_interval: tuple[float, float],
        first_block_cache: float,
        seed: int,
        phases: _Phases,
    ) -> Any:
        device = next(self.pipe.components["transformer"].parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        pipeline: Any = _text2image_pipeline(device, phases, self)
        pipeline.register_components(
            **self.pipe.components,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler_config),
            tokenizer=self.pipe.tokenizer,
            t5_tokenizer=self.pipe.t5_tokenizer,
        )
        pipeline.guider.guidance_scale = guidance
        # Diffusers 0.40 stores the interval on private attrs set by BaseGuidance.__init__;
        # outside [start, stop) num_conditions == 1 and the uncond forward never runs.
        pipeline.guider._start, pipeline.guider._stop = cfg_interval
        transformer = self.pipe.components["transformer"]
        restore_cache = _apply_first_block_cache(transformer, pipeline.guider, first_block_cache)
        return pipeline, generator, restore_cache

    @uses_components("text_encoder")
    def encode_stage(self, block: Any, components: Any, state: Any) -> Any:
        return block(components, state)

    @uses_components("text_conditioner", "transformer")
    def condition_stage(self, block: Any, components: Any, state: Any) -> Any:
        return block(components, state)

    @uses_components("transformer")
    def denoise_stage(self, block: Any, components: Any, state: Any) -> Any:
        return block(components, state)

    @uses_components("vae")
    def decode_stage(self, block: Any, components: Any, state: Any) -> Any:
        return block(components, state)


def render_request(
    model: AnimaModel, prompt: str, negative_prompt: str, width: int, height: int,
    steps: int, guidance: float, cfg_interval: tuple[float, float],
    first_block_cache: float, seed: int, phases: _Phases,
) -> Any:
    pipeline, generator, restore_cache = model.prepare_request(
        prompt, negative_prompt, width, height, steps, guidance, cfg_interval,
        first_block_cache, seed, phases,
    )
    phases.enter(_ENCODE_PROMPT)
    try:
        return pipeline(
            prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
            num_inference_steps=steps, generator=generator, output="images", output_type="pt",
        )
    finally:
        # FBC cleanup only restores Python hook/forward/cache bindings. It must run
        # even when a failed device cannot admit another component scope.
        restore_cache()


def _text2image_pipeline(device: Any, phases: _Phases, scope_owner: AnimaModel | None = None) -> Any:
    """The text2image pipeline, instrumented on the blocks it will actually run.

    `ModularPipeline.blocks` is a property returning a DEEPCOPY, so a hook installed through
    it is thrown away and the meter never moves — the whole of the frozen `conditioning 0%`
    defect (se-026). `blocks=` hands the pipeline the object it keeps, and the identity is
    the fix: `scripts/anima-conform.py` executes Diffusers' own loop driver against it.
    """
    class Announce(ModularPipelineBlocks):
        """A weightless block whose only effect is to advance the meter."""

        model_name = "anima"

        @property
        def description(self) -> str:
            return "Announces Anima's conditioning phase to Runtime telemetry."

        def __call__(self, components: Any, state: Any) -> tuple[Any, Any]:
            phases.enter(_CONDITION)
            return components, state

    class RuntimeAnimaPipeline(AnimaModularPipeline):
        @property
        def _execution_device(self) -> Any:
            return device

    blocks = AnimaAutoBlocks().get_workflow("text2image")
    order = list(blocks.sub_blocks)
    if not {"denoise.denoise", "denoise.text_conditioning"} <= set(order):
        raise RuntimeError(f"Diffusers Anima text2image blocks changed: {order}")
    blocks.sub_blocks["denoise.denoise"].progress_bar = _progress_bar(phases)
    blocks.sub_blocks.insert("conditioning", Announce(), order.index("denoise.text_conditioning"))
    if scope_owner is not None:
        from .stage_scopes import install_scopes

        install_scopes(blocks, scope_owner)
    return RuntimeAnimaPipeline(blocks=blocks)


class _Phases:
    """The one open `tel.stage` bracket, advanced from inside Diffusers' opaque call.

    `pipeline(...)` runs text encoding, conditioning, denoise and decode behind a single
    call, so the brackets are opened and closed by the blocks themselves rather than by
    `with` statements around them. Without a `tel` — `warm` has no attempt — it is silent.
    """

    def __init__(self, tel: Telemetry | None = None) -> None:
        self.tel = tel
        self.open: AbstractContextManager[None] | None = None

    def enter(self, phase: _Phase) -> None:
        self.close()
        if self.tel is None:
            return
        self.open = self.tel.stage(phase.stage, overall_range=phase.bounds)
        self.open.__enter__()

    def steps(self, total: int) -> Callable[[int], None]:
        if self.tel is None:
            return lambda position: None
        return self.tel.step_callback(  # type: ignore[no-any-return]
            total, stage=_DENOISE.stage, overall_range=_DENOISE.bounds
        )

    def close(self) -> None:
        self.__exit__(None, None, None)

    def __enter__(self) -> _Phases:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        bracket, self.open = self.open, None
        if bracket is not None:
            bracket.__exit__(kind, exc, traceback)


class _DenoiseProgress:
    """Diffusers' denoise-loop progress bar projected onto Runtime telemetry."""

    def __init__(self, total: int, phases: _Phases) -> None:
        self.total = total
        self.phases = phases
        self.position = 0
        self.step: Callable[[int], None] | None = None

    def __enter__(self) -> _DenoiseProgress:
        self.phases.enter(_DENOISE)
        self.step = self.phases.steps(self.total)
        return self

    def update(self, count: int = 1) -> None:
        if self.step is None:
            raise RuntimeError("Anima denoise progress updated outside its loop")
        for _ in range(count):
            self.step(self.position)
            self.position += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.phases.enter(_DECODE)


def _progress_bar(phases: _Phases) -> Callable[..., _DenoiseProgress]:
    def progress_bar(iterable: object = None, total: int | None = None) -> _DenoiseProgress:
        if iterable is not None or total is None or total < 1:
            raise RuntimeError("Anima denoise progress requires one positive total")
        return _DenoiseProgress(total, phases)

    return progress_bar


@app.entrypoint
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: AnimaModel,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    """Generate one native-resolution Anima image."""
    width, height = _BUCKETS[(payload.aspect_ratio, payload.megapixels)]
    steps = payload.steps
    if payload.cfg_interval_start > payload.cfg_interval_stop:
        raise UnsupportedInput(
            "cfg_interval_start must not exceed cfg_interval_stop", code="cfg_interval"
        )
    with _Phases(tel) as phases:
        images = render_request(
            model,
            payload.quality_prefix + payload.prompt,
            payload.negative_prompt,
            width,
            height,
            steps,
            payload.guidance,
            (payload.cfg_interval_start, payload.cfg_interval_stop),
            payload.first_block_cache,
            payload.seed,
            phases,
        )
        # Still `decoding`: the pipeline left that bracket open and the host copy below is
        # the tail of the same work.
        image = images[0]
        pixels = (image.clamp(0, 1) * 255).to("cpu", dtype=torch.uint8)
        if pixels.ndim == 3 and pixels.shape[0] == 3:
            pixels = pixels.permute(1, 2, 0)
        pixels = pixels.contiguous()
        rgb = bytes(pixels.numpy().tobytes())
        phases.enter(_SAVE)
        asset = out.save_image(ImageFrame(width, height, rgb), format="webp")
    return ImageOutput(
        asset, width, height, steps, payload.guidance, hashlib.sha256(rgb).hexdigest()
    )


def _declare_first_block_retry_state(transformer: Any, forward: Any) -> Callable[[], None]:
    """Own Diffusers FBC's named state, without copying residual tensor storage.

    Cosmos blocks and the FBC hooks only read cached tensors; new residuals replace
    bindings. The shared manager owns separate cond/uncond entries and the current
    context. Preserve all of those identities, plus lazy registry traversal caches.
    """
    declare = cozy_author.retry_state
    registries = [
        module._diffusers_hook for module in transformer.modules()
        if hasattr(module, "_diffusers_hook")
    ]
    allowed = {_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK}
    if any(set(registry.hooks) - allowed for registry in registries):
        return lambda: None  # No contract for unrelated third-party cache hooks.
    leader = transformer.transformer_blocks[0]._diffusers_hook.hooks[_FBC_LEADER_BLOCK_HOOK]
    manager = leader.state_manager
    metadata = {
        id(hook._metadata): hook._metadata
        for registry in registries for hook in registry.hooks.values()
    }.values()
    fields = ("head_block_output", "head_block_residual", "tail_block_residuals", "should_compute")
    names = (
        "_cozy_retry_state", "_cozy_retry_tensors", "_cozy_retry_generators",
        "_cozy_retry_hooks", "_cozy_retry_forwards",
    )
    previous = {name: vars(transformer)[name] for name in names if name in vars(transformer)}
    base = getattr(transformer, "_cozy_retry_state", None)

    def save() -> Any:
        cache = manager._state_cache
        return (
            None if base is None else base[0](),
            manager._current_context, cache, dict(cache),
            [(state, tuple(getattr(state, name) for name in fields)) for state in cache.values()],
            [
                (registry, "_child_registries_cache" in vars(registry),
                 getattr(registry, "_child_registries_cache", None))
                for registry in registries
            ],
            [(item, item._cached_parameter_indices) for item in metadata],
        )

    def restore(state: Any) -> None:
        underlying, context, cache, entries, states, traversals, parameters = state
        if base is not None:
            base[1](underlying)
        cache.clear()
        cache.update(entries)
        manager._state_cache = cache
        manager._current_context = context
        for value, items in states:
            for name, item in zip(fields, items, strict=True):
                setattr(value, name, item)
        for registry, existed, value in traversals:
            if existed:
                registry._child_registries_cache = value
            else:
                vars(registry).pop("_child_registries_cache", None)
        for item, indices in parameters:
            item._cached_parameter_indices = indices

    # Claim only the FBC hooks just installed and their rewritten block forwards,
    # plus our condition wrapper. Never sweep unrelated module hooks/replacements.
    hooks = tuple(hook for registry in registries for hook in registry.hooks.values())
    forwards = (forward, *(block.forward for block in transformer.transformer_blocks))
    declare(
        transformer, save, restore,
        tensors=getattr(transformer, "_cozy_retry_tensors", lambda: ()),
        generators=getattr(transformer, "_cozy_retry_generators", lambda: ()),
        hooks=(*getattr(transformer, "_cozy_retry_hooks", ()), *hooks),
        forwards=(*getattr(transformer, "_cozy_retry_forwards", ()), *forwards),
    )

    def detach() -> None:
        for name in names:
            vars(transformer).pop(name, None)
        vars(transformer).update(previous)

    return detach


def _apply_first_block_cache(transformer: Any, guider: Any, threshold: float) -> Callable[[], None]:
    """Attach FBCache hooks for one request; returns the restore. 0.0 attaches nothing.

    Diffusers ships `apply_first_block_cache` but its registry does not know
    `CosmosTransformerBlock` — the one-line registration below is the whole shim. Under
    this package's sequential batch-1 CFG the cond and uncond forwards would otherwise
    share one residual cache and compare cond against uncond; wrapping the transformer
    forward in a per-condition `cache_context` keeps the two streams separate. Hooks are
    request-scoped (the HiDiffusion precedent in sdxl): removed in the restore so the
    resident module leaves exactly as it entered.

    Its request contract restores residual-cache bindings and the registry's condition
    context together. It requires the current explicit Runtime retry contract.
    """
    if threshold <= 0.0:
        return lambda: None

    # Verify originals before installing wrappers. An unknown forward must not be
    # hidden inside the closure that this request subsequently claims as its own.
    originals_known = True
    for module in (transformer, *transformer.transformer_blocks):
        original = cozy_author.original_forward(module)
        if not (
            cozy_author.is_declared_pure(original)
            or (
                getattr(original, "__self__", None) is module
                and getattr(original, "__func__", None) is getattr(type(module), "forward", None)
            )
        ):
            originals_known = False
            break

    try:
        TransformerBlockRegistry.get(CosmosTransformerBlock)
    except ValueError:
        TransformerBlockRegistry.register(
            CosmosTransformerBlock,
            TransformerBlockMetadata(return_hidden_states_index=0),
        )
    apply_first_block_cache(transformer, FirstBlockCacheConfig(threshold=threshold))
    original_forward = transformer.forward
    registry = HookRegistry.check_if_exists_or_initialize(transformer)

    def forward(*args: Any, **kwargs: Any) -> Any:
        # CosmosTransformer3DModel predates diffusers' CacheMixin, so the context is set
        # on the hook registry directly — the same two calls `cache_context` makes.
        registry._set_context("cond" if guider.is_conditional else "uncond")
        try:
            return original_forward(*args, **kwargs)
        finally:
            registry._set_context(None)

    transformer.forward = forward
    detach_retry = (
        _declare_first_block_retry_state(transformer, forward)
        if originals_known else lambda: None
    )

    def restore() -> None:
        detach_retry()
        transformer.forward = original_forward
        for block in transformer.transformer_blocks:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            for name in (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK):
                if name in registry.hooks:
                    registry.remove_hook(name, recurse=False)

    return restore

# ------------------------------------------------------------------ the quantize job


#: The family's derived DiT encodings by lane (se-009/cr-086, same shape as sdxl's
#: se-023 job). `bf16` is not an output: the SOURCE is the canonical BF16 cut. Only the
#: transformer quantizes — text encoder, conditioner and VAE stay at source precision.
_LANE_ENCODINGS: dict[str, str] = {"fp8": "fp8-rowwise/1", "mxfp8": "mxfp8/1"}
_LANE_BYTES = 16 << 30

Lane = Literal["fp8", "mxfp8"]


def _quantize(
    ctx: Context, tel: Telemetry, source: AnimaModel, lane: Lane, max_relative_frobenius: float | None
) -> ModelArtifact:
    return derive.quantize_artifact(
        source,
        derive.plan(("transformer",), _LANE_ENCODINGS[lane], max_relative_frobenius=max_relative_frobenius),
        ctx=ctx, tel=tel, output=lane,
    )


# One function per lane: Runtime requires a receipt for EVERY declared weights output. The
# quantizer checkpoints every encoded tensor, so a re-issued run adopts completed tensors.
@invocable(memoize=True)
async def fp8(
    ctx: Context,
    *,
    source: AnimaModel,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """Row-wise FP8 DiT weights from one BF16/F16 source; everything else inherits."""
    return _quantize(ctx, tel, source, "fp8", max_relative_frobenius)


@invocable(memoize=True)
async def mxfp8(
    ctx: Context,
    *,
    source: AnimaModel,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """MXFP8 DiT weights from one BF16/F16 source; everything else inherits."""
    return _quantize(ctx, tel, source, "mxfp8", max_relative_frobenius)


app.job(fp8, name="fp8", weights=(WeightsOutput("fp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
app.job(mxfp8, name="mxfp8", weights=(WeightsOutput("mxfp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
