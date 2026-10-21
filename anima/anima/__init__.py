"""Anima text-to-image through Diffusers' maintained modular pipeline."""

from __future__ import annotations

import hashlib
import math
import secrets
from enum import IntEnum
from typing import Annotated

import cozy_runtime.derive as derive
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into
import msgspec
import torch
from cozy_runtime.author import (
    Adjustments,
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
    Weights,
    WeightsOutput,
    invocable,
)

from .model import _SAVE, AnimaModel, _Phases

app = App()


class Megapixels(IntEnum):
    """Anima's native resolution classes (se-024): tier 1 is the ~1 MP training set (1024²),
    tier 2 the 1536 class (~2.4 MP) the model is meant to run at and this package's default.
    Denoise attention is full (no windowing), so time and VRAM scale roughly linearly with area."""

    MP1 = 1
    MP2 = 2


#: Each tier's square side; tier 2 scales tier 1 by exactly 1.5.
_SIDE = {Megapixels.MP1: 1024, Megapixels.MP2: 1536}
#: The Cosmos rotary table holds 128 patches of 16 px on an axis.
_MAX_SIDE = 128 * 16
#: Admission's bound per tier: its area plus 1/16 for rounding to 32 px.
_DEMAND = {tier: (side * 17 // 16, side) for tier, side in _SIDE.items()}


def _rounded(area: int, ratio: float) -> tuple[int, int]:
    """Qwen-Image's `calculate_dimensions`: the area at that ratio, each side rounded to 32 px."""
    width = math.sqrt(area * ratio)
    return round(width / 32) * 32, round(width / ratio / 32) * 32


def output_size(tier: Megapixels, ratio: str) -> tuple[int, int, bool]:
    """The tier's area at `W:H`, held inside the rotary range, 32 px sides and the tier's
    demand; True when that moved it off the rounded size."""
    area, (w, h) = _SIDE[tier] ** 2, map(int, ratio.split(":"))
    widest = min(_MAX_SIDE**2 / area, area / 32**2)
    width, height = _rounded(area, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, 32), _MAX_SIDE), min(max(height, 32), _MAX_SIDE)
    while width * height > _DEMAND[tier][0] * _DEMAND[tier][1]:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != _rounded(area, w / h)


_WEBP_OUTPUT = AssetBound(max_bytes=64 << 20, media_types=("image/webp",))


#: The official Anima model card's negative prompt, verbatim (se-026). Owner-labelled bank
#: 2026-09-02: with seed, steps, guidance and geometry fixed, THIS string is the quality
#: knob — Anima's score-bucket conditioning acts through the negative, so pushing away
#: score_1/score_2/score_3 (plus jpeg artifacts, chromatic aberration, artist name — a
#: literal tag that suppresses signature text) moves output off the model's flat dated
#: aesthetic. The negative is the ISOLATED cause: the labelled loser carried `score_7, safe`
#: in its positive prompt and still lost, and one labelled winner carried no positive quality
#: prefix at all. The card's positive quality tags are prompt text, the caller's to write (the
#: README gives them). Changing this string changes every render that omits the field.
CARD_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, "
    "artist name, blurry, jpeg artifacts, chromatic aberration"
)


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    negative_prompt: str = CARD_NEGATIVE
    aspect_ratio: Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")] = "1:1"
    megapixels: Annotated[Megapixels, Shape(pixels=_DEMAND)] = Megapixels.MP2
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


@app.entrypoint(defaults={"model": "anima@1.0.0"}, memoize=Memoize(requires=["seed"]))
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: AnimaModel,
    out: Outputs,
    tel: Telemetry,
    adj: Adjustments,
) -> ImageOutput:
    """Generate one native-resolution Anima image."""
    width, height, held = output_size(payload.megapixels, payload.aspect_ratio)
    if held:
        size = f"{width}:{height}"
        adj.clamp("aspect_ratio", payload.aspect_ratio, size, "nearest size the model admits")
    steps = payload.steps
    seed = payload.seed if payload.seed is not None else secrets.randbits(53)
    if payload.cfg_interval_start > payload.cfg_interval_stop:
        raise UnsupportedInput(
            "cfg_interval_start must not exceed cfg_interval_stop", code="cfg_interval"
        )
    # Both CFG conditions in one forward: short of memory, the Runtime cuts its rows. FBCache
    # keeps a cache per condition, so it runs them apart.
    guided = payload.guidance > 1.0 and payload.cfg_interval_start < payload.cfg_interval_stop
    batched = guided and payload.first_block_cache <= 0.0
    with _Phases(tel) as phases:
        images = model.render(
            payload.prompt,
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

# The Anima recipes: which weights each lane encodes; every other tensor inherits unchanged.
# `cozy model quantize <model> --int8|--fp8` runs the function whose weights output is
# that lane. `cozy_runtime.derive` owns the encodings and the round-trip tripwire.
_LANE_BYTES = 16 << 30
_QUANTIZER = (  # the Runtime code and native libraries that decide a lane's bytes
    "cozy_runtime.derive.facade",
    "cozy_runtime.derive.quantization",
    "cozy_runtime.derive.microscale",
    "cozy_runtime.derive.safetensors_io",
    "cozy_runtime.tensord_channels",
    MemoDistribution("tensorfs"),
    MemoDistribution("numpy"),
)


#: The fp8 lane: the DiT's block-aligned rank-2 weights, row-wise.
FP8 = ("transformer",)


@invocable(memo_dependencies=_QUANTIZER)
async def fp8(
    ctx: Context,
    *,
    source: Weights[AnimaModel],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    """Row-wise FP8 DiT weights from one BF16/F16 source; everything else inherits."""
    plan = derive.plan(FP8, "fp8.rowwise", max_relative_frobenius=max_relative_frobenius)
    return derive.quantize_artifact(source, plan, ctx=ctx, tel=tel, output="fp8")


#: The int8 lane: every attention and MLP linear of the 28 DiT blocks and of the Qwen3 text
#: encoder's 28 layers. The AdaLN modulation, embeddings, the text conditioner and the VAE keep
#: their source bytes.
INT8 = QuantizationPlan(
    components=("transformer", "text_encoder"),
    include=(
        "transformer.transformer_blocks.*.attn[12].to_[qkv].weight",
        "transformer.transformer_blocks.*.attn[12].to_out.0.weight",
        "transformer.transformer_blocks.*.ff.net.*weight",
        "text_encoder.layers.*.self_attn.[qkvo]_proj.weight",
        "text_encoder.layers.*.mlp.*_proj.weight",
    ),
    output_precision="preserve",
)


@invocable(memo_version="anima-int8/1")
async def int8(ctx: Context, *, source: Weights[AnimaModel], tel: Telemetry) -> ModelArtifact:
    """The int8 lane (`int8.convrot`): W8A8 on int8 tensor cores; see `INT8`."""
    return quantize_into(ctx, tel, source=source, plan=INT8, encoding="int8.convrot", output="int8")


app.entrypoint(int8, name="int8", weights=(WeightsOutput("int8", max_new_bytes=_LANE_BYTES),), accelerator=False)
app.entrypoint(fp8, name="fp8", weights=(WeightsOutput("fp8", max_new_bytes=_LANE_BYTES),), accelerator=False)
