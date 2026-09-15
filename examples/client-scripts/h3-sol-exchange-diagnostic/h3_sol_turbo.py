"""Matched PDD-8 videos through the ordinary serving path, including Ulysses.

Requires the paired development Runtime and migrated H3 workflow in README.md.
The workflow owns decoding and output checks; Runtime owns the models and kernels.
"""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    AttentionContext,
    Context,
    ImageAsset,
    Loader,
    Outputs,
    Telemetry,
    VideoAsset,
    sequence_parallel,
)
from cozy_runtime.models.minimax_h3 import H3TurboBase, H3TurboLoRA
from cozy_runtime.models.minimax_h3.official import NumericalChecks, frames_for

from h3 import H3Model as WorkflowModel
from h3 import _finish

app = App()
ATTENTION_BACKEND: Literal["sol-attn", "flash-attn3"] = "sol-attn"


@sequence_parallel(degrees=(2, 4))
class PinnedTurboBase(H3TurboBase, encoded_leaves="accept", fusion="accept"):
    """Choose on every rank before preparation; request pins then verify this choice."""

    def load(self, loader: Loader) -> None:
        super().load(loader)
        from exchange_diagnostic import install

        install()

    def choose_attention(self, context: AttentionContext) -> str | None:
        if context.component.rsplit("/", 1)[-1] == "fl2va_dit":
            return ATTENTION_BACKEND
        return None


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int = 7101
    duration_s: Annotated[int, msgspec.Meta(ge=5, le=15)] = 15
    sol_dense_steps: Literal[3, 4, 8] = 4


class Result(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    continuation_frame: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    warnings: list[str]


@app.entrypoint
def main(
    ctx: Context,
    payload: Input,
    base_model: PinnedTurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    """Run the real separate base/PDD adapter with one explicit sparse warmup policy."""
    # The published workflow predates the model migration. Refuse a mismatched
    # capture rather than mixing its numerical checks with a different model class.
    if not isinstance(base_model, WorkflowModel):
        raise RuntimeError("capture must use the H3 workflow that imports Runtime's models")
    ctx.raise_if_cancelled()
    view = base_model.for_request(ctx, seed=payload.seed)
    checks = NumericalChecks(tel, base_model.pipe.resident)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        state = base_model.pipe.start_fl2va(
            prompt=payload.prompt,
            first_frame=None,
            last_frame=None,
            generator=base_model.pipe.generator(view.generator),
            steps=8,
            frames=frames_for(payload.duration_s),
            task="fl2va_turbo",
        )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        base_model.condition_text("fl2va_turbo", state, checks=checks)
    tel.log("h3 Sol qualification policy", sol_dense_steps=payload.sol_dense_steps)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = base_model.sample_fl2va_turbo(
            state,
            turbo_lora=turbo_lora,
            sol_dense_steps=payload.sol_dense_steps,
            on_step=tel.step_callback(8, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    result = _finish(
        base_model,
        "fl2va_turbo",
        state,
        schedule,
        duration_s=payload.duration_s,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        checks=checks,
    )
    return Result(result.video, result.continuation_frame, result.warnings)
