"""Anima text-to-image through Diffusers' maintained modular pipeline."""

from __future__ import annotations

import hashlib
import secrets
from enum import Enum, IntEnum
from typing import Annotated

import cozy_runtime.derive as derive
import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    MemoDistribution,
    Memoize,
    ModelArtifact,
    ModelDefault,
    Outputs,
    Shape,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    invocable,
)

from .model import _SAVE, AnimaModel, _Phases

app = App()


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
    #: cache states, so a request with it on runs CFG as two batch-1 forwards.
    #:
    #: DEFAULT OFF, deliberately (se-026, 2026-09-03). Measured at 0.075 rather than assumed:
    #: it extrapolates 44% of forwards, and against the same seed it consistently smooths
    #: faces and coarsens fine texture — the two places the first block is a bad proxy for
    #: the other 27. It also costs VRAM, because those cached residuals stay live: ALONE at
    #: the default 1536 class it refuses `device_shortfall` on an 8 GiB card, reproducibly.
    #: It survives only when `cfg_interval` happens to be narrowing the live cache states,
    #: and a default that works only because another default masks it is not a default.
    first_block_cache: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0
    #: Given, the request is memoized; omitted, the package draws one and reports it.
    seed: int | None = None


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, _WEBP_OUTPUT]
    width: int
    height: int
    steps: int
    guidance: float
    digest: str
    seed: int
    """The request's seed, else the one drawn for it."""


def warmup() -> None:
    """Runtime's off-lock import hook (cr-104). Every import is at module scope (se-041), so
    `import anima` already paid the cost and nothing is left to defer."""


@app.entrypoint(memoize=Memoize(requires=["seed"]))
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
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    if payload.cfg_interval_start > payload.cfg_interval_stop:
        raise UnsupportedInput(
            "cfg_interval_start must not exceed cfg_interval_stop", code="cfg_interval"
        )
    # One forward for both CFG conditions when the stage has room for two (counting weights
    # the runtime may unmap, as ComfyUI does), else two. Decided before the stage, for the
    # whole request; FBCache keeps a cache per condition, so it runs them apart. A request
    # with no uncond pass asks nothing: its peak is one item's.
    guided = payload.guidance > 1.0 and payload.cfg_interval_start < payload.cfg_interval_stop
    batched = guided and payload.first_block_cache <= 0.0 and model.batch_fits("render")
    with _Phases(tel) as phases:
        images = model.render(
            payload.quality_prefix + payload.prompt,
            payload.negative_prompt,
            width,
            height,
            steps,
            payload.guidance,
            (payload.cfg_interval_start, payload.cfg_interval_stop),
            payload.first_block_cache,
            seed,
            phases,
            batched,
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
        asset, width, height, steps, payload.guidance, hashlib.sha256(rgb).hexdigest(), seed
    )


# ------------------------------------------------------------------ quantized lanes

# The Anima recipe: encode the DiT's GEMM weights; every other component inherits
# unchanged. `cozy model quantize <model> --fp8|--mxfp8` runs the job whose weights output is
# that lane. `cozy_runtime.derive` owns the encodings and the round-trip tripwire.
_LANE_BYTES = 16 << 30
_QUANTIZER = (  # the Runtime code and native libraries that decide a lane's bytes
    "cozy_runtime.derive.facade",
    "cozy_runtime.derive.quantization",
    "cozy_runtime.derive.microscale",
    "cozy_runtime.derive.safetensors_io",
    "tensorfs.derived",
    MemoDistribution("tensorfs"),
    MemoDistribution("numpy"),
)


def _quantize(
    ctx: Context, tel: Telemetry, source: AnimaModel, lane: str, encoding: str, tripwire: float | None
) -> ModelArtifact:
    plan = derive.plan(("transformer",), encoding, max_relative_frobenius=tripwire)
    return derive.quantize_artifact(source, plan, ctx=ctx, tel=tel, output=lane)


@invocable(memoize=True, memo_dependencies=_QUANTIZER)
async def fp8(
    ctx: Context,
    *,
    source: AnimaModel,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """Row-wise FP8 DiT weights from one BF16/F16 source; everything else inherits."""
    return _quantize(ctx, tel, source, "fp8", "fp8-rowwise/1", max_relative_frobenius)


@invocable(memoize=True, memo_dependencies=_QUANTIZER)
async def mxfp8(
    ctx: Context,
    *,
    source: AnimaModel,
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """MXFP8 DiT weights from one BF16/F16 source; everything else inherits."""
    return _quantize(ctx, tel, source, "mxfp8", "mxfp8/1", max_relative_frobenius)


app.job(fp8, name="fp8", weights=(WeightsOutput("fp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
app.job(mxfp8, name="mxfp8", weights=(WeightsOutput("mxfp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
