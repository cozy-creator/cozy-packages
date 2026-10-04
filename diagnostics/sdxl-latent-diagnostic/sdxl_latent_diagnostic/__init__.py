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
    Tier 1 is base SDXL unless the caller opts in.
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
import importlib.metadata
import json
import json
import random
from enum import Enum, IntEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import cozy_runtime.derive as derive
import msgspec
import torch
import cozy_runtime.author as cozy_author
from cozy_runtime.author import (
    App,
    AssetBound,
    ConformanceError,
    Context,
    FileAsset,
    ImageAsset,
    ImageFrame,
    Loader,
    MemoDistribution,
    Model,
    ModelArtifact,
    ModelDefault,
    OutputError,
    Outputs,
    Shape,
    Telemetry,
    WeightsOutput,
    invocable,
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

class Txt2ImgInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = ""
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP1
    #: None runs HiDiffusion where base SDXL cannot go: above tier 1. An explicit choice
    #: always runs, and the result warns where it is known to hurt.
    hidiffusion: bool | None = None
    #: LAYER 1 of the two-layer clamp. These bounds REJECT; they never quietly clip. The
    #: `ModelDefault` marker is what makes the field omittable on the wire and concrete
    #: before the handler runs — the handler sees `int`, never `int | None`.
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=1, le=50)] = 20
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


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, _WEBP_OUTPUT]
    latents: Annotated[FileAsset, AssetBound(max_bytes=131072, media_types=("application/octet-stream",))]
    latent_sha256: str
    diagnostic_json: str
    width: int
    height: int
    steps: int
    guidance: float
    classifier_free: bool
    """Whether the negative branch ran. Value-plane gating, made observable: a caller can
    see that `guidance <= 1.0` bought it a one-pass step rather than having to trust it."""
    hidiffusion_applied: bool
    """Whether this request used HiDiffusion."""
    digest: str
    """sha256 of the decoded RGB pixel bytes — the determinism fence over the WHOLE loop,
    not over one step."""
    warnings: list[str]
    """Explicit choices that ran although they are known to hurt the picture."""


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
                    # Its size hook writes the image size from the args and keeps nothing else,
                    # so a step that ran out of device memory may run again. Runtimes before
                    # memory v3 have no `pure`: nothing to mark there.
                    mark = getattr(cozy_author, "pure", None)
                    for handle in self.info["hooks"] if mark is not None else ():
                        hook = self._forward_pre_hooks.get(handle.id)
                        if hook is not None:
                            mark(hook)
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
        # Checkpoint vocabularies override the standard SDXL CLIP vocabulary. Both
        # sources are read-only and the parsed tokenizers survive between requests.
        self.tokenizers = (
            _tokenizer(loader.assets, "tokenizer"),
            _tokenizer(loader.assets, "tokenizer_2"),
        )

    def warm(self, ctx: Context) -> None:
        """One dry step — tokenize, encode, denoise, decode — at the default request's
        shape (classifier-free batch, base SDXL), so no request pays a first-call
        cost. The runtime calls it once per fill, before the placement serves. Outputs
        are dropped, so no schedule: the tensors carry their own dtype and device."""
        ctx.raise_if_cancelled()
        prompt, pooled = self.encode(*_tokenize(self.tokenizers, ""))
        side = _WARM_SIDE // 8
        latents = prompt.new_empty((2, 4, side, side)).normal_()
        time_ids = prompt.new_tensor([[_WARM_SIDE, _WARM_SIDE, 0, 0, _WARM_SIDE, _WARM_SIDE]] * 2)
        noise = self.denoise(
            latents, 999, prompt.repeat(2, 1, 1), pooled.repeat(2, 1), time_ids, 0, 1, False
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
            # The SDXL VAE overflows fp16. bf16 has fp32's range: 48 dB PSNR to fp32 on the
            # 1024px fox (run 1642), ~0.5 s faster on an RTX 4070 Laptop.
            native_bf16 = latents.is_cuda and torch.cuda.is_bf16_supported(including_emulation=False)
            wide = torch.bfloat16 if native_bf16 else torch.float32
            vae.to(dtype=wide)
            latents = latents.to(dtype=wide)
        try:
            with torch.inference_mode():
                return vae.decode(latents / self.pipe.vae_scale).sample
        finally:
            if upcast:
                vae.to(dtype=original_dtype)
            if tiled:
                vae.disable_tiling()

    @uses_components("vae")
    def decode_saved_latent(self, latents: Any, mode: str) -> Any:
        """One independent stock decode of an immutable saved latent, never a retry."""
        vae = self.pipe.components["vae"]
        latents = latents.to(next(vae.parameters()).device)
        original_dtype = next(vae.parameters()).dtype
        original = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
        upcast = bool(getattr(vae.config, "force_upcast", False))
        if upcast:
            native_bf16 = latents.is_cuda and torch.cuda.is_bf16_supported(including_emulation=False)
            wide = torch.bfloat16 if native_bf16 else torch.float32
            vae.to(dtype=wide)
            latents = latents.to(dtype=wide)
        try:
            with torch.inference_mode():
                if mode == "tiled512":
                    vae.enable_tiling()
                    vae.tile_sample_min_size = 512
                    vae.tile_latent_min_size = 64
                    return vae.tiled_decode(latents / self.pipe.vae_scale).sample
                vae.disable_tiling()
                # Adequate-memory stock decode; the native receipt and decoder input
                # shape must independently verify that no runtime tile was selected.
                return vae.decode(latents / self.pipe.vae_scale).sample
        finally:
            vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size = original
            if upcast:
                vae.to(dtype=original_dtype)


# ------------------------------------------------------------------ the handler


def _finite(torch: Any, value: Any) -> float:
    """The tensor's largest magnitude, with NaN neutralized so the metric is spellable.

    A metric that cannot be serialized is a metric that is not there: the observation emit
    boundary refuses a non-finite value, so the NaN FRACTION is its own number and this one
    stays a real magnitude.
    """
    return round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)


def _tokenizer(assets: Any, name: str) -> Any:
    """Use checkpoint overrides, or bundled CLIP data when the whole pair is absent.

    A partial override still fails through ModelAssets; mixing vocabularies and merges
    from different sources would silently change the model's tokenization. No download
    or temporary file is needed, including during read-only admission.
    """
    pad = {"tokenizer": "<|endoftext|>", "tokenizer_2": "!"}[name]
    vocab_name, merges_name = f"{name}/vocab.json", f"{name}/merges.txt"
    names = assets.names()
    if vocab_name in names or merges_name in names:
        vocab_bytes = assets.read(vocab_name)
        merges_bytes = assets.read(merges_name)
    else:
        root = Path(__file__).parent
        vocab_bytes = (root / "clip_vocab.json").read_bytes()
        merges_bytes = (root / "clip_merges.txt").read_bytes()
    vocab = json.loads(vocab_bytes)
    merges = [
        tuple(line.split(" "))
        for line in merges_bytes.decode().splitlines()
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
    above_native = payload.megapixels is not Megapixels.MP1
    hidiffusion_applied = above_native if payload.hidiffusion is None else payload.hidiffusion
    warnings: list[str] = []
    if above_native and not hidiffusion_applied:
        warnings.append("HiDiffusion off above 1 MP: expect duplicated or tiled subjects")
    if hidiffusion_applied and not above_native:
        warnings.append("HiDiffusion at 1 MP composes at half resolution: expect simpler scenes")
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

    if tuple(latents.shape) != (1, 4, 128, 128) or latents.dtype is not torch.float16:
        raise RuntimeError("diagnostic requires the original full1024-square FP16 latent")
    latent_bytes = latents.detach().cpu().contiguous().numpy().tobytes()
    if len(latent_bytes) != 131072:
        raise RuntimeError("latent artifact exceeds its declared exact bound")
    latent_asset = out.save_bytes(latent_bytes, media_type="application/octet-stream")
    out.publish("latents", latent_asset)
    decoder_inputs: list[list[int]] = []
    vae = model.pipe.components["vae"]

    def observe_decoder(_module: Any, args: Any) -> None:
        decoder_inputs.append([int(n) for n in args[0].shape])

    observer = vae.decoder.register_forward_pre_hook(observe_decoder)

    try:
        with tel.stage("decode", overall_range=(0.90, 0.98)):
            image = model.decode(latents)
    finally:
        observer.remove()
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
        latents=latent_asset,
        latent_sha256=hashlib.sha256(latent_bytes).hexdigest(),
        diagnostic_json=json.dumps({
            "source_release": "installed sdxl2.4.0",
            "original_source_sha256": "f4c6f63303df72913ca49ab3a33a08b841dafb3c5993fcebf7ed35a7b801e8b8",
            "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sdk": importlib.metadata.version("cozy-runtime"),
            "latent_dtype": "float16", "latent_shape": [1, 4, 128, 128],
            "latent_layout": "little-endian C-order NCHW", "latent_bytes": len(latent_bytes),
            "vae_scale": model.pipe.vae_scale, "scheduler": model.pipe.scheduler_config,
            "decoder_input_shapes": decoder_inputs,
            "mode_evidence": "actual decoder input shapes plus separate native invoke modes",
        }, sort_keys=True),
        width=decoded_w,
        height=decoded_h,
        steps=steps,
        guidance=payload.guidance,
        classifier_free=classifier_free,
        hidiffusion_applied=hidiffusion_applied,
        digest=hashlib.sha256(rgb).hexdigest(),
        warnings=warnings,
    )


class DecodeInput(msgspec.Struct, forbid_unknown_fields=True):
    latents: Annotated[FileAsset, AssetBound(max_bytes=131072)]
    latent_sha256: str
    mode: Literal["untiled", "tiled512"]


class DecodeOutput(msgspec.Struct):
    image: Annotated[ImageAsset, _WEBP_OUTPUT]
    latent_sha256: str
    diagnostic_json: str


@app.entrypoint
def decode_latents(ctx: Context, payload: DecodeInput, model: SdxlModel,
                   out: Outputs, tel: Telemetry) -> DecodeOutput:
    raw = payload.latents.read_bytes()
    if len(raw) != 131072 or hashlib.sha256(raw).hexdigest() != payload.latent_sha256:
        raise RuntimeError("decode requires the exact immutable128KiB saved latent")
    model.for_request(ctx, seed=0)
    latents = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(1, 4, 128, 128)
    vae = model.pipe.components["vae"]
    decoder_inputs: list[list[int]] = []

    def observe_decoder(_module: Any, args: Any) -> None:
        decoder_inputs.append([int(n) for n in args[0].shape])

    observer = vae.decoder.register_forward_pre_hook(observe_decoder)
    try:
        image = model.decode_saved_latent(latents, payload.mode)
    finally:
        observer.remove()
    pixels = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    rgb = bytes(pixels.cpu().numpy().tobytes())
    asset = out.save_image(ImageFrame(1024, 1024, rgb), format="webp")
    return DecodeOutput(asset, payload.latent_sha256, json.dumps({
        "requested_mode": payload.mode, "decoder_input_shapes": decoder_inputs,
        "sdk": importlib.metadata.version("cozy-runtime"),
        "vae_scale": model.pipe.vae_scale, "width": 1024, "height": 1024,
        "limit": "independent stock decode; source/native receipts required, not a whole-forward retry",
    }, sort_keys=True))


# ------------------------------------------------------------------ the quantize job


#: The family's derived UNet encodings by lane (cr-073, se-023). `bf16` is not an output
#: here: the SOURCE is the canonical BF16 cut, and the lane name stays in the catalog as
#: that cut. Quantization lives with its serving package — the slot is this file's own
#: SdxlModel, and `cozy_runtime.derive` owns the math and the tier-1 tripwire.
_LANE_ENCODINGS: dict[str, str] = {"fp8": "fp8-rowwise/1", "mxfp8": "mxfp8/1"}
_LANE_BYTES = 16 << 30

Lane = Literal["fp8", "mxfp8"]


def _quantize(
    ctx: Context, tel: Telemetry, source: SdxlModel, lane: Lane, max_relative_frobenius: float | None
) -> ModelArtifact:
    return derive.quantize_artifact(
        source,
        derive.plan(("unet",), _LANE_ENCODINGS[lane], max_relative_frobenius=max_relative_frobenius),
        ctx=ctx, tel=tel, output=lane,
    )


# One function per lane: Runtime requires a receipt for EVERY declared weights output. The
# quantizer checkpoints every encoded tensor, so a re-issued run adopts completed tensors.
@invocable(
    memoize=True,
    memo_version="sdxl-quantize/1",
    memo_dependencies=(
        "cozy_runtime.derive.facade",
        "cozy_runtime.derive.quantization",
        "cozy_runtime.derive.microscale",
        "cozy_runtime.derive.safetensors_io",
        "tensorfs.derived",
        MemoDistribution("tensorfs"),
        MemoDistribution("numpy"),
    ),
)
async def fp8(
    ctx: Context,
    *,
    source: SdxlModel,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """Row-wise FP8 UNet weights from one BF16/F16 source; everything else inherits."""
    return _quantize(ctx, tel, source, "fp8", max_relative_frobenius)


@invocable(
    memoize=True,
    memo_version="sdxl-quantize/1",
    memo_dependencies=(
        "cozy_runtime.derive.facade",
        "cozy_runtime.derive.quantization",
        "cozy_runtime.derive.microscale",
        "cozy_runtime.derive.safetensors_io",
        "tensorfs.derived",
        MemoDistribution("tensorfs"),
        MemoDistribution("numpy"),
    ),
)
async def mxfp8(
    ctx: Context,
    *,
    source: SdxlModel,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """MXFP8 UNet weights from one BF16/F16 source; everything else inherits."""
    return _quantize(ctx, tel, source, "mxfp8", max_relative_frobenius)


app.job(fp8, name="fp8", weights=(WeightsOutput("fp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
app.job(mxfp8, name="mxfp8", weights=(WeightsOutput("mxfp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
