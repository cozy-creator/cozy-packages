"""se-008 — the SDXL launch package: `generate`, text to image, four components.

The second launch family, and the ENGINELESS-PATH REFERENCE (`cozy-runtime-sdxl.md`): one
`Model` class with author-written `load`, three public methods each declaring the
heavyweight components it may touch, one entrypoint. There is no engine class, no device
call, no placement, no offload, no pinning and no backend API — there is no such surface on
`cozy_runtime.author` for this file to reach. Whether the UNet was resident when `denoise`
was entered is the runtime's question, decided before entry, and this file cannot observe
the answer.

What it adds over cr-008b's corpus fixture, which it is otherwise faithful to (its banked
1024px pixel digest reproduces here, byte for byte):

  * ASPECT x MEGAPIXEL BUCKETS (se-024). SDXL fine-tunes are trained on fixed
    resolutions, not free geometry, so the request names a ratio and a pixel-count tier
    and `_BUCKETS` maps the pair to one native bucket; `Shape(pixels=...)` on the tier is
    how the demand axes are derived from a preset the runtime cannot otherwise measure
    (§1.2). A pair this package does not offer is unspellable, not rounded. Tiers above
    1 ride HiDiffusion — base SDXL duplicates subjects past its training resolution.
  * `ModelDefault` steps/guidance, and the TWO-LAYER CLAMP. Layer 1 is the field's own
    `Meta` bound, which REJECTS what the caller actually sent. Layer 2 is a deployment's
    visible `Clamp`, which lowers it and says so in the adjustments envelope. A silent
    clamp is the failure both layers exist to prevent.
  * VALUE-PLANE CFG GATING. `guidance <= 1.0` skips the negative branch entirely — the
    turbo/lightning execution shape follows from the RESOLVED VALUE alone, with zero
    adapter awareness in this file. When the adapter runtime (cr-010) lands, a turbo
    recipe supplies that value as an omitted-field default and not one line here changes.
  * AN OUTPUT-INTEGRITY FLOOR. A decode that produced NaNs, or a flat field with no
    picture in it, is a failure this package reports rather than a PNG it publishes.

Since cr-073 the family's quantization job (`quantize`, se-023) ships here too — a
package legally mixes entrypoints and jobs, the job's slot is this file's own SdxlModel,
and `cozy_runtime.derive` owns the quantization math and the tier-1 tripwire. The request
names the derived lanes it wants ({fp8, mxfp8}); `bf16` is not a lane the job produces —
the SOURCE is the canonical BF16 cut.

Code states CAPABILITY; bindings state SELECTION. Nothing here names a repo, release,
checkpoint or revision — `package.toml` and the deploy binding do.
"""

from __future__ import annotations

import hashlib
import json
import random
from enum import Enum, IntEnum
from typing import Annotated, Any, Literal

import cozy_runtime.derive as derive
import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    ConformanceError,
    Context,
    ImageAsset,
    ImageFrame,
    Loader,
    Model,
    ModelDefault,
    OutputError,
    Outputs,
    Shape,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    uses_components,
)
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from hidiffusion import (  # type: ignore[import-not-found,import-untyped]
    apply_hidiffusion,
    remove_hidiffusion,
)
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from transformers import initialization as transformer_init

app = App()


class AspectRatio(Enum):
    """The offered buckets. A ratio absent here does not round — it refuses at decode."""

    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    WIDE = "16:9"
    PORTRAIT = "3:4"
    TALL = "9:16"


class Megapixels(IntEnum):
    """The offered pixel-count tiers, in nominal megapixels.

    Tier 1 is SDXL's native training resolution; every higher tier rides HiDiffusion
    (RAU-Net + windowed attention), whose published recipes cover the 2048-class through
    ~8MP and the 4096² flagship. VRAM, fp16, CFG on (weights 6.5 GiB total, staged per
    method — denoise holds the 4.8 GiB UNet, decode holds the 0.16 GiB VAE):

        1   ~1.0 MP  the launch envelope — serves on 8 GiB
        2   ~2.1 MP  denoise ~6 GiB, whole-frame fp32 decode ~4 GiB — a 12 GiB card
        4   ~4.2 MP  denoise ~8 GiB, whole-frame fp32 decode ~8 GiB — 12 GiB at its edge
        8   ~8.3 MP  denoise ~12 GiB, decode tiled — a 24 GiB card
        16  ~16.8 MP denoise ~15-19 GiB, decode tiled, 1:1 only — the top of 24 GiB
    """

    MP1 = 1
    MP2 = 2
    MP4 = 4
    MP8 = 8
    MP16 = 16


#: (aspect, tier) -> (width, height). Tier 1 is SDXL's own training buckets; a higher
#: tier scales them by sqrt(tier) and snaps to the 64-px stride the latent grid requires.
#: Tier 16 offers only 1:1: HiDiffusion's 4096-class recipe engages when BOTH latent
#: sides reach 512, which no non-square 16 MP bucket satisfies — those pairs are absent,
#: not rounded.
_BUCKETS: dict[tuple[AspectRatio, Megapixels], tuple[int, int]] = {
    (AspectRatio.SQUARE, Megapixels.MP1): (1024, 1024),
    (AspectRatio.LANDSCAPE, Megapixels.MP1): (1152, 896),
    (AspectRatio.WIDE, Megapixels.MP1): (1344, 768),
    (AspectRatio.PORTRAIT, Megapixels.MP1): (896, 1152),
    (AspectRatio.TALL, Megapixels.MP1): (768, 1344),
    (AspectRatio.SQUARE, Megapixels.MP2): (1472, 1472),
    (AspectRatio.LANDSCAPE, Megapixels.MP2): (1600, 1280),
    (AspectRatio.WIDE, Megapixels.MP2): (1920, 1088),
    (AspectRatio.PORTRAIT, Megapixels.MP2): (1280, 1600),
    (AspectRatio.TALL, Megapixels.MP2): (1088, 1920),
    (AspectRatio.SQUARE, Megapixels.MP4): (2048, 2048),
    (AspectRatio.LANDSCAPE, Megapixels.MP4): (2304, 1792),
    (AspectRatio.WIDE, Megapixels.MP4): (2688, 1536),
    (AspectRatio.PORTRAIT, Megapixels.MP4): (1792, 2304),
    (AspectRatio.TALL, Megapixels.MP4): (1536, 2688),
    (AspectRatio.SQUARE, Megapixels.MP8): (2880, 2880),
    (AspectRatio.LANDSCAPE, Megapixels.MP8): (3264, 2560),
    (AspectRatio.WIDE, Megapixels.MP8): (3776, 2176),
    (AspectRatio.PORTRAIT, Megapixels.MP8): (2560, 3264),
    (AspectRatio.TALL, Megapixels.MP8): (2176, 3776),
    (AspectRatio.SQUARE, Megapixels.MP16): (4096, 4096),
}

#: The demand table `Shape` reads. The runtime derives (width, height, pixels) from ONE
#: field, so the tier carries its largest bucket: pixel count is the axis that moves VRAM
#: (attention and decode scale with area; aspect at equal area does not), and the largest
#: bucket keeps the derived demand an upper bound over the tier.
_TIER_DEMAND: dict[Megapixels, tuple[int, int]] = {
    tier: max(
        (size for (_, t), size in _BUCKETS.items() if t is tier),
        key=lambda size: size[0] * size[1],
    )
    for tier in Megapixels
}

#: Whole-frame decode stops here. Above the 4 MP class one fp32 frame's decoder features
#: alone outgrow a 24 GiB card, so the VAE tiles; at or below it the decode stays
#: whole-frame and the banked 1024px digest is untouched.
_UNTILED_DECODE_PIXELS = 2048 * 2048
_WEBP_OUTPUT = AssetBound(max_bytes=64 << 20, media_types=("image/webp",))

#: The warm dry step's geometry, measured rather than the request's (cl-003, decisions
#: #392): a pass that decodes 1024px with every component resident OOMs an 8 GiB card at
#: every boot, so the binding came up DEGRADED for nothing. 512px pays the same first-call
#: costs at a shape every card fits.
_WARM_SIDE = 512

class Txt2ImgInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    negative_prompt: str = ""
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    #: A caller may opt out of HiDiffusion without changing the selected model or package.
    #: At tier 1, non-square buckets always use baseline SDXL because those live
    #: comparisons regressed. Above tier 1 there is no baseline to fall back to.
    hidiffusion: bool = True
    #: LAYER 1 of the two-layer clamp. These bounds REJECT; they never quietly clip. The
    #: `ModelDefault` marker is what makes the field omittable on the wire and concrete
    #: before the handler runs — the handler sees `int`, never `int | None`.
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=1, le=50)] = 30
    guidance: Annotated[ModelDefault[float], msgspec.Meta(ge=0.0, le=20.0)] = 7.0
    seed: int = 1005

    def __post_init__(self) -> None:
        if (self.aspect_ratio, self.megapixels) not in _BUCKETS:
            offered = ", ".join(
                aspect.value for aspect, tier in _BUCKETS if tier is self.megapixels
            )
            raise ValueError(
                f"no {self.megapixels.value}-megapixel bucket for aspect "
                f"{self.aspect_ratio.value}; this tier offers: {offered}"
            )
        if self.megapixels is not Megapixels.MP1 and not self.hidiffusion:
            raise ValueError(
                "megapixels above 1 requires HiDiffusion: base SDXL is trained at ~1MP "
                "and duplicates subjects beyond it — omit hidiffusion or send true"
            )


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, _WEBP_OUTPUT]
    width: int
    height: int
    steps: int
    guidance: float
    classifier_free: bool
    """Whether the negative branch ran. Value-plane gating, made observable: a caller can
    see that `guidance <= 1.0` bought it a one-pass step rather than having to trust it."""
    hidiffusion_applied: bool
    """Whether this request actually used HiDiffusion after the tier and geometry gates."""
    digest: str
    """sha256 of the decoded RGB pixel bytes — the determinism fence over the WHOLE loop,
    not over one step."""


# ------------------------------------------------------------------ the model


def _hidiffusion_unet_type() -> type[Any]:
    """The patched UNet owner, imported only when Runtime constructs the model."""
    class HiDiffusionUNet(UNet2DConditionModel):
        """Select and reset the request's denoising implementation before step zero."""

        _cozy_base_num_upsamplers: int
        _cozy_hidiffusion_active: bool

        def begin_denoise_request(self, steps: int, hidiffusion: bool) -> None:
            if steps <= 0:
                raise ValueError("HiDiffusion requires a positive timestep count")
            active = bool(getattr(self, "_cozy_hidiffusion_active", False))
            if hidiffusion != active:
                if hidiffusion:
                    self.num_upsamplers = self._cozy_base_num_upsamplers
                    apply_hidiffusion(self)
                else:
                    remove_hidiffusion(self)
                    self.num_upsamplers = self._cozy_base_num_upsamplers
                self._cozy_hidiffusion_active = hidiffusion
            if not hidiffusion:
                return
            self._num_timesteps = steps
            info = getattr(self, "info", None)
            if isinstance(info, dict):
                info["size"] = None
                info["upsample_size"] = None
            for module in self.modules():  # type: ignore[attr-defined]
                if hasattr(module, "timestep"):
                    module.timestep = 0
                if hasattr(module, "max_timestep"):
                    module.max_timestep = steps

    return HiDiffusionUNet


class SdxlPipeline:
    """The four constructed component roots, built from the artifact's immutable config.

        text_encoder     196 destinations   0.229 GiB   CLIPTextModel (ViT-L text tower)
        text_encoder_2   517 destinations   1.294 GiB   CLIPTextModelWithProjection (bigG)
        unet            1680 destinations   4.782 GiB   UNet2DConditionModel
        vae              248 destinations   0.156 GiB   AutoencoderKL
                        ────────────────────────────
                        2641 destinations   6.461 GiB

    Construction declares fp16 logical destinations. Plain weights fill those
    destinations; the runtime may replace supported UNet linear leaves with native
    encoded implementations when the checkpoint carries FP8 weights.
    """

    def __init__(self, config: Any) -> None:
        mapping = config.mapping()
        text_encoder = dict(mapping["text_encoder"])
        text_encoder_2 = dict(mapping["text_encoder_2"])
        for clip_config in (text_encoder, text_encoder_2):
            clip_config["initializer_factor"] = float(clip_config["initializer_factor"])
        unet_type = _hidiffusion_unet_type()
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "text_encoder": CLIPTextModel(CLIPTextConfig(**text_encoder)).to(
                    torch.float16  # type: ignore[arg-type]
                ),
                "text_encoder_2": CLIPTextModelWithProjection(
                    CLIPTextConfig(**text_encoder_2)
                ).to(torch.float16),  # type: ignore[arg-type]
                "unet": unet_type.from_config(mapping["unet"]).to(torch.float16),
                "vae": AutoencoderKL.from_config(mapping["vae"]).to(  # type: ignore[no-untyped-call]
                    torch.float16
                ),
            }
        unet = self.components["unet"]
        # HiDiffusion recognizes this constructed SDXL UNet by its module keys. Keep the
        # package free of a catalog/model identifier and let that structural check decide.
        unet.name_or_path = ""
        unet._cozy_base_num_upsamplers = unet.num_upsamplers
        unet._cozy_hidiffusion_active = False
        self.scheduler_config: dict[str, Any] = dict(mapping["scheduler"])
        self.vae_scale: float = float(mapping["vae"]["scaling_factor"])


def warmup() -> None:
    """Runtime's off-lock import hook (cr-104). Every import is at module scope (se-041), so
    `import sdxl` already paid the cost; the HiDiffusion subclass body left here is free."""
    _hidiffusion_unet_type()


def build_pipeline(config: Any) -> SdxlPipeline:
    return SdxlPipeline(config)


class SdxlModel(Model[SdxlPipeline], encoded_leaves="accept"):
    """Admits by pure topology satisfaction: no structural twins, so no stamp keyword.

    Runtime may replace supported linear leaves with encoded implementations. The
    model invokes those leaves through their forward methods and leaves storage and
    execution routing to Runtime. Plain checkpoints remain supported.

    THREE component sets rather than one coarse whole-pipeline set. The coarse form is
    legal and simpler, and on this card it is not servable: 6.461 GiB of weights declared
    resident at once beside a 1024px VAE decode does not fit 8 GiB, and the answer to that
    is the runtime's staging ladder — which has nothing to stage if every method declares
    everything.
    """

    pipe: SdxlPipeline
    tokenizers: tuple[Any, Any]

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(SdxlPipeline, factory=build_pipeline)
        # Tokenizers are model semantics, not package code: their vocab and merges are
        # CozyTensors assets, read as verified bytes. Loading writes no file.
        self.tokenizers = (
            _tokenizer(loader.assets, "tokenizer"),
            _tokenizer(loader.assets, "tokenizer_2"),
        )

    def warm(self, ctx: Context) -> None:
        """One dry step — tokenize, encode, denoise, decode — at the default request's
        shape (classifier-free batch, HiDiffusion on), so no request pays a first-call
        cost. The runtime calls it once per fill, before the placement serves. Outputs
        are dropped, so no schedule: the tensors carry their own dtype and device."""
        ctx.raise_if_cancelled()
        prompt, pooled = self.encode(*_tokenize(self.tokenizers, ""))
        side = _WARM_SIDE // 8
        latents = prompt.new_empty((2, 4, side, side)).normal_()
        time_ids = prompt.new_tensor([[_WARM_SIDE, _WARM_SIDE, 0, 0, _WARM_SIDE, _WARM_SIDE]] * 2)
        noise = self.denoise(
            latents, 999, prompt.repeat(2, 1, 1), pooled.repeat(2, 1), time_ids, 0, 1, True
        )
        self.decode(noise.chunk(2)[1])

    @uses_components("text_encoder", "text_encoder_2")
    def encode(self, ids: Any, ids_2: Any) -> tuple[Any, Any]:
        """SDXL's two-tower conditioning. ONE method, BOTH encoders — a composite operation
        declares its components together rather than opening two scopes.

        IT PLACES ITS OWN INPUTS (#534c). Token ids arrive on the host, and the only code
        entitled to say where they go is code that holds a lease on the component they are
        going to: it follows the encoder's own weights to wherever the runtime already put
        them. The caller used to do this with a hard-coded `cuda:0`, which was an author
        naming a device — the one live instance se-001's record left open — and it made this
        package unservable on any envelope the runtime did not place at ordinal zero.
        """
        first_encoder = self.pipe.components["text_encoder"]
        second_encoder = self.pipe.components["text_encoder_2"]
        with torch.inference_mode():
            ids = ids.to(first_encoder.get_input_embeddings().weight.device)
            ids_2 = ids_2.to(second_encoder.get_input_embeddings().weight.device)
            first = first_encoder(ids, output_hidden_states=True)
            second = second_encoder(ids_2, output_hidden_states=True)
            prompt = torch.cat([first.hidden_states[-2], second.hidden_states[-2]], dim=-1)
            return prompt, second.text_embeds

    @uses_components("unet")
    def denoise(
        self,
        latents: Any,
        timestep: Any,
        prompt: Any,
        text_embeds: Any,
        time_ids: Any,
        step: int,
        total_steps: int,
        hidiffusion: bool,
    ) -> Any:
        unet = self.pipe.components["unet"]
        if step == 0:
            unet.begin_denoise_request(total_steps, hidiffusion)
        with torch.inference_mode():
            return unet(
                latents,
                timestep,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample

    @uses_components("vae")
    def decode(self, latents: Any) -> Any:
        vae = self.pipe.components["vae"]
        tiled = int(latents.shape[-2]) * int(latents.shape[-1]) * 64 > _UNTILED_DECODE_PIXELS
        if tiled:
            vae.enable_tiling()
        original_dtype = next(vae.parameters()).dtype
        upcast = bool(getattr(vae.config, "force_upcast", False))
        if upcast:
            vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)
        try:
            with torch.inference_mode():
                return vae.decode(latents / self.pipe.vae_scale).sample
        finally:
            if upcast:
                vae.to(dtype=original_dtype)
            if tiled:
                vae.disable_tiling()


# ------------------------------------------------------------------ the handler


def _finite(torch: Any, value: Any) -> float:
    """The tensor's largest magnitude, with NaN neutralized so the metric is spellable.

    A metric that cannot be serialized is a metric that is not there: the observation emit
    boundary refuses a non-finite value, so the NaN FRACTION is its own number and this one
    stays a real magnitude.
    """
    return round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)


def _tokenizer(assets: Any, name: str) -> Any:
    """Build one CLIP tokenizer from the checkpoint's vocabulary/merge asset bytes.

    Deliberately NOT ``from_pretrained``: that spelling can resolve against the Hub and
    would make model execution depend on mutable network state. The in-memory constructor
    cannot reach anywhere and touches no filesystem. Missing model assets are a Runtime
    admission error, never a source-tree or network fallback.
    """
    pad = {"tokenizer": "<|endoftext|>", "tokenizer_2": "!"}[name]
    vocab = json.loads(assets.read(f"{name}/vocab.json"))
    merges = [
        tuple(line.split(" "))
        for line in assets.read(f"{name}/merges.txt").decode().splitlines()
        if line and not line.startswith("#version")
    ]
    return CLIPTokenizer(
        vocab=vocab, merges=merges, errors="replace", pad_token=pad, model_max_length=77
    )


def _tokenize(tokenizers: tuple[Any, Any], prompt: str) -> tuple[Any, Any]:
    first, second = tokenizers
    return _ids(first, prompt), _ids(second, prompt)


def _ids(tok: Any, prompt: str) -> Any:
    return tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


#: THE NaN CHECK reads the FLOAT decode, before quantization, and that is the whole of
#: what a red arm taught here: `torch.isnan` on a `clamp(0,1).to(uint8)` tensor is
#: constant False, because a uint8 cannot be NaN — the very quantization that makes an
#: image encodable is what erases the evidence of a diverged decode.
#:
#: Pixel SPREAD is reported as telemetry, not refused. A flat field is a real failure
#: mode (a wrong VAE scaling factor, an unfilled decoder) but it is also a real REQUEST:
#: "solid white background", "minimalist pure black poster", "flat pastel swatch". A
#: refusal here destroyed a correct, completed, billed generation.
def _integrity(torch: Any, image: Any, pixels: Any, tel: Telemetry) -> None:
    nan_fraction = float(torch.isnan(image).float().mean())
    tel.metric("image_nan_fraction", round(nan_fraction, 6))
    if nan_fraction > 0.0:
        raise OutputError(
            f"the decode produced NaN over {nan_fraction:.4%} of the image and this "
            "package does not publish it: a non-finite decode is a failed generation, "
            "not a picture with artefacts",
            code="output_integrity",
        )
    tel.metric("image_spread", round(float(pixels.to(torch.float32).std()), 4))


def _request_generator(torch: Any, source: object, *, device: Any) -> Any:
    """Adapt Runtime's public request generator to a device-placed torch generator.

    `view.generator` is a `random.Random` while the runtime is weightless and a
    `torch.Generator` once real fills exist; both spellings resolve here.
    """
    if isinstance(source, torch.Generator):
        return source
    if not isinstance(source, random.Random):
        raise ConformanceError(
            f"request generator has unsupported type {type(source).__name__}",
            code="artifact_config",
        )
    return torch.Generator(device=device).manual_seed(source.getrandbits(63))


@app.entrypoint
def generate(
    ctx: Context,
    payload: Txt2ImgInput,
    model: SdxlModel,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    """One text-to-image generation: tokenize, encode, denoise, decode, encode a PNG."""
    view = model.for_request(ctx, seed=payload.seed)
    width, height = _BUCKETS[(payload.aspect_ratio, payload.megapixels)]
    steps = payload.steps
    # Above tier 1 HiDiffusion always runs — decode already refused the contradiction —
    # and any aspect is legal: past its training resolution base SDXL is not an
    # alternative. At tier 1 the measured geometry gate stands: square only.
    hidiffusion_applied = payload.hidiffusion and (
        payload.megapixels is not Megapixels.MP1 or width == height
    )
    # The value plane, and the only branch in this file that reads a request number: above
    # 1.0 the negative branch is worth its second forward pass, at or below it is not.
    classifier_free = payload.guidance > 1.0

    with tel.stage("tokenize", overall_range=(0.00, 0.02)):
        ids, ids_2 = _tokenize(model.tokenizers, payload.prompt)
        if classifier_free:
            neg, neg_2 = _tokenize(model.tokenizers, payload.negative_prompt)

    with tel.stage("encode", overall_range=(0.02, 0.10)):
        prompt, pooled = model.encode(ids, ids_2)
        if classifier_free:
            negative, neg_pooled = model.encode(neg, neg_2)
    # THE DEVICE ENVELOPE, READ RATHER THAN NAMED (#534c). The conditioning came back from
    # the encoders the runtime placed, so it already carries where this request runs; the
    # latents and the schedule follow it. This is data flow across the boundary an author
    # owns, and it is the whole of what the deleted `torch.device("cuda", 0)` was doing.
    device = prompt.device
    tel.metric("prompt_absmax", _finite(torch, prompt))
    tel.metric("pooled_absmax", _finite(torch, pooled))

    scheduler = EulerDiscreteScheduler.from_config(  # type: ignore[no-untyped-call]
        model.pipe.scheduler_config
    )
    scheduler.set_timesteps(steps, device=device)
    generator = _request_generator(torch, view.generator, device=device)
    latents = (
        torch.randn(
            1, 4, height // 8, width // 8, generator=generator, device=device,
            dtype=torch.float16,
        )
        * scheduler.init_noise_sigma
    )
    # SDXL's micro-conditioning: (original_h, original_w, crop_top, crop_left, target_h,
    # target_w). The bucket IS the original size — these fine-tunes were trained on it.
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], device=device, dtype=torch.float16
    )
    if classifier_free:
        batch_prompt = torch.cat([negative, prompt])
        batch_pooled = torch.cat([neg_pooled, pooled])
        batch_ids = torch.cat([time_ids, time_ids])
    else:
        batch_prompt, batch_pooled, batch_ids = prompt, pooled, time_ids

    on_step = tel.step_callback(steps, stage="denoise", overall_range=(0.10, 0.90))
    # HiDiffusion's window-attention shift uses torch's CPU RNG. Isolate and seed it from
    # the request so a canceled or concurrent history cannot change this request's output.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(payload.seed)
        with tel.stage("denoise", overall_range=(0.10, 0.90)):
            for index, timestep in enumerate(scheduler.timesteps):
                ctx.raise_if_cancelled()
                batch = torch.cat([latents] * 2) if classifier_free else latents
                model_input = scheduler.scale_model_input(batch, timestep)
                noise = model.denoise(
                    model_input,
                    timestep,
                    batch_prompt,
                    batch_pooled,
                    batch_ids,
                    index,
                    steps,
                    hidiffusion_applied,
                )
                if classifier_free:
                    uncond, cond = noise.chunk(2)
                    noise = uncond + payload.guidance * (cond - uncond)
                if index == 0:
                    tel.metric("noise_absmax_step0", _finite(torch, noise))
                latents = scheduler.step(noise, timestep, latents).prev_sample
                on_step(index)
    tel.metric("latent_absmax", _finite(torch, latents))

    with tel.stage("decode", overall_range=(0.90, 0.98)):
        image = model.decode(latents)
    tel.metric("image_absmax", _finite(torch, image))
    pixels = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    _integrity(torch, image, pixels, tel)

    rgb = bytes(pixels.cpu().numpy().tobytes())
    decoded_h, decoded_w = int(pixels.shape[0]), int(pixels.shape[1])
    tel.metric("decoded_pixels", float(decoded_w * decoded_h))
    with tel.stage("encode_webp", overall_range=(0.98, 1.00)):
        asset = out.save_image(ImageFrame(decoded_w, decoded_h, rgb), format="webp")
    return ImageOutput(
        image=asset,
        width=decoded_w,
        height=decoded_h,
        steps=steps,
        guidance=payload.guidance,
        classifier_free=classifier_free,
        hidiffusion_applied=hidiffusion_applied,
        digest=hashlib.sha256(rgb).hexdigest(),
    )


# ------------------------------------------------------------------ the quantize job


#: The family's derived UNet encodings by lane (cr-073, se-023). `bf16` is not an output
#: here: the SOURCE is the canonical BF16 cut, and the lane name stays in the catalog as
#: that cut. Quantization lives with its serving package — the slot is this file's own
#: SdxlModel, and `cozy_runtime.derive` owns the math and the tier-1 tripwire.
_LANE_ENCODINGS: dict[str, str] = {"fp8": "fp8-rowwise/1", "mxfp8": "mxfp8/1"}
_LANE_BYTES = 16 << 30

Lane = Literal["fp8", "mxfp8"]


class QuantizeInput(msgspec.Struct, forbid_unknown_fields=True):
    lanes: Annotated[tuple[Lane, ...], msgspec.Meta(min_length=1)] = ("fp8", "mxfp8")
    max_relative_frobenius: float | None = None


class QuantizedLanes(msgspec.Struct):
    """Each requested lane's full `derive.QuantizeResult`; an unrequested lane is null."""

    fp8: derive.QuantizeResult | None = None
    mxfp8: derive.QuantizeResult | None = None


@app.job(
    name="quantize",
    weights=tuple(WeightsOutput(lane, max_new_bytes=_LANE_BYTES) for lane in _LANE_ENCODINGS),
)
def quantize(
    ctx: Context,
    payload: QuantizeInput,
    source: SdxlModel,
    tel: Telemetry,
) -> QuantizedLanes:
    """Derive the requested row-wise UNet lanes from one reviewed BF16 source.

    Outputs stay declared for the full supported set — the descriptor is static — and an
    unrequested lane is simply never opened; the sink permits unwritten declared outputs.
    """
    if len(set(payload.lanes)) != len(payload.lanes):
        raise UnsupportedInput("quantize lanes must be unique", code="quantization_lanes")
    results = {
        lane: derive.quantize(
            source,
            derive.plan(
                ("unet",),
                _LANE_ENCODINGS[lane],
                max_relative_frobenius=payload.max_relative_frobenius,
            ),
            ctx=ctx, tel=tel, output=lane,
        )
        for lane in payload.lanes
    }
    return QuantizedLanes(fp8=results.get("fp8"), mxfp8=results.get("mxfp8"))


# Normalization is a managed source operation; quantization policy is plain composition.
from . import normalization  # noqa: E402

app.job(
    normalization.normalize_component, name="normalize-component",
    weights=(WeightsOutput("model", max_new_bytes=normalization.MAX_NEW_BYTES),),
)
app.job(
    normalization.assemble_normalized, name="assemble-normalized",
    weights=(WeightsOutput("model", max_new_bytes=0),),
)
