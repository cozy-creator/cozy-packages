"""Matched standard H3 attention videos through the existing 30-step workflow."""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    AttentionContext,
    Context,
    Image,
    ImageAsset,
    Outputs,
    Telemetry,
    VideoAsset,
    sequence_parallel,
)
from cozy_runtime.models.minimax_h3 import H3Model

from h3 import FirstLastFrameToVideoInput
from h3 import H3Model as WorkflowModel
from h3 import fl2va

app = App()
ATTENTION_BACKEND: Literal["sol-attn", "flash-attn3"] = "sol-attn"
KeyframeAssets = Annotated[Assets[Image], AssetLimits(images=2)]


@sequence_parallel(degrees=(2, 4))
class PinnedH3(H3Model, encoded_leaves="accept", fusion="accept"):
    """Select on every rank during preparation; request pins verify the same choice."""

    def choose_attention(self, context: AttentionContext) -> str | None:
        if context.component.rsplit("/", 1)[-1] == "fl2va_dit":
            return ATTENTION_BACKEND
        return None


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int = 7101
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)] = 15


class Result(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    continuation_frame: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    warnings: list[str]


@app.entrypoint
def main(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: PinnedH3,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    """Render standard 30-step H3 with one base checkpoint and no adapters."""
    if not isinstance(model, WorkflowModel):
        raise RuntimeError("capture must use the H3 workflow that imports Runtime's models")
    result = fl2va(
        ctx,
        FirstLastFrameToVideoInput(
            prompt=payload.prompt, seed=payload.seed, duration_s=payload.duration_s, steps=30
        ),
        assets,
        model,
        out,
        tel,
    )
    return Result(result.video, result.continuation_frame, result.warnings)
