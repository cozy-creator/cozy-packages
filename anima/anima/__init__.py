"""Anima text-to-image through Diffusers' maintained modular pipeline."""

from __future__ import annotations

import math
from typing import Annotated

import msgspec
import torch
from cozy_runtime.author import (
    MEGAPIXEL,
    Adjustments,
    App,
    Context,
    ImageFrame,
    ImageRequest,
    ImageResult,
    Memoize,
    Outputs,
    QuantizationPlan,
    Recipe,
    ReferenceImages,
    Telemetry,
    UnsupportedInput,
    held,
)

from .model import _SAVE, AnimaModel, _Phases

app = App()


#: Anima's native resolution classes (se-024): the ~1 MP training set (1024²) to the 1536 class
#: (1536², 2.25 MiP) the model is meant to run at, this package's default. Denoise attention is full
#: (no windowing), so time and VRAM scale roughly linearly with area.
_MEGAPIXELS = (1.0, 2.25)
#: The Cosmos rotary table holds 128 patches of 16 px on an axis.
_MAX_SIDE = 128 * 16
#: Steps (default, range) and CFG scale (default, range): the model card's.
_STEPS, _GUIDANCE = (30, 8, 50), (4.5, 1.0, 10.0)


def _rounded(area: float, ratio: float) -> tuple[int, int]:
    """Qwen-Image's `calculate_dimensions`: the area at that ratio, each side rounded to 32 px."""
    width = math.sqrt(area * ratio)
    return round(width / 32) * 32, round(width / ratio / 32) * 32


def fitted_size(pixels: float, ratio: tuple[int, int]) -> tuple[int, int, bool]:
    """`pixels` at `W:H`, held inside the rotary range, 32 px sides and 1/16 over `pixels`; True when
    that moved it off the rounded size."""
    w, h = ratio
    widest = min(_MAX_SIDE**2 / pixels, pixels / 32**2)
    width, height = _rounded(pixels, min(max(w / h, 1 / widest), widest))
    width, height = min(max(width, 32), _MAX_SIDE), min(max(height, 32), _MAX_SIDE)
    while width * height > pixels * 17 / 16:
        width, height = (width - 32, height) if width >= height else (width, height - 32)
    return width, height, (width, height) != _rounded(pixels, w / h)


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


class GenerateInput(ImageRequest):
    #: Anima makes images from text only.
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=0)] = msgspec.field(
        default_factory=list
    )
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
) -> ImageResult:
    """Generate one native-resolution Anima image."""
    payload.ignored(adj, "prompt_upsampling", reason="Anima has no prompt upsampler")
    target = payload.target(None, megapixels=_MEGAPIXELS[1])
    megapixels = held(adj, "megapixels", target.pixels / MEGAPIXEL, *_MEGAPIXELS,
                      "Anima's 1 MP training size to its 1536 class")
    width, height, moved = fitted_size(megapixels * MEGAPIXEL, target.ratio)
    if moved:
        ratio = f"{target.ratio[0]}:{target.ratio[1]}"
        adj.clamp("aspect_ratio", ratio, f"{width}:{height}", "nearest size the model admits")
    steps = held(adj, "steps", payload.steps or _STEPS[0], *_STEPS[1:])
    guidance = held(adj, "guidance", _GUIDANCE[0] if payload.guidance is None else payload.guidance,
                    *_GUIDANCE[1:])
    negative = CARD_NEGATIVE if payload.negative_prompt is None else payload.negative_prompt
    seed = payload.drawn_seed()
    if payload.cfg_interval_start > payload.cfg_interval_stop:
        raise UnsupportedInput(
            "cfg_interval_start must not exceed cfg_interval_stop", code="cfg_interval"
        )
    # Both CFG conditions in one forward: short of memory, the Runtime cuts its rows. FBCache
    # keeps a cache per condition, so it runs them apart.
    guided = guidance > 1.0 and payload.cfg_interval_start < payload.cfg_interval_stop
    batched = guided and payload.first_block_cache <= 0.0
    with _Phases(tel) as phases:
        images = model.render(
            payload.prompt,
            negative,
            width,
            height,
            steps,
            guidance,
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
    return ImageResult(asset, width, height, seed, payload.prompt)


# ------------------------------------------------------------------ quantized lanes

# The Anima recipes: which weights each lane encodes; every other tensor inherits unchanged.
# `cozy model quantize <model> --format int8|fp8 --package <this>` runs the one for that lane.
# `cozy_runtime.derive` owns the encodings and the round-trip tripwire.

#: The fp8 lane: the DiT's block-aligned rank-2 weights, row-wise.
FP8 = Recipe(
    lane="fp8",
    model=AnimaModel,
    plan=QuantizationPlan(encoding="fp8.rowwise", components=("transformer",)),
)

#: The int8 lane (`int8.convrot`, W8A8 on int8 tensor cores): every attention and MLP linear of
#: the 28 DiT blocks and of the Qwen3 text encoder's 28 layers. The AdaLN modulation, embeddings,
#: the text conditioner and the VAE keep their source bytes.
INT8 = Recipe(
    lane="int8",
    model=AnimaModel,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=(
            "transformer.transformer_blocks.*.attn[12].to_[qkv].weight",
            "transformer.transformer_blocks.*.attn[12].to_out.0.weight",
            "transformer.transformer_blocks.*.ff.net.*weight",
            "text_encoder.layers.*.self_attn.[qkvo]_proj.weight",
            "text_encoder.layers.*.mlp.*_proj.weight",
        ),
        output_precision="preserve",
    ),
)
