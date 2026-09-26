"""Controlled motion-context comparison using the package's ordinary managed children."""

from __future__ import annotations

import time
from typing import Annotated

import msgspec
from cozy_runtime.author import (
    AssetBound,
    Assets,
    Context,
    MediaDecoder,
    Mixed,
    Outputs,
    Telemetry,
    VideoAsset,
    invocable,
)
from cozy_runtime.models.minimax_h3.continuation import plan_continuation

from assembly import AssembleVideoRequest, assemble
from h3 import (
    H3Model as H3Model,
    H3TurboBase as H3TurboBase,
    H3TurboLoRA as H3TurboLoRA,
    MotionInput,
    _create_references,
    app,
    motion_segment_turbo,
)
from story import StoryReference as StoryReference
from story import StorySegment, compile_segments


class Comparison(msgspec.Struct):
    context_22: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]
    context_39: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]
    context_56: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]


class ComparisonInput(msgspec.Struct, forbid_unknown_fields=True):
    references: Annotated[list[StoryReference], msgspec.Meta(min_length=1, max_length=9)]
    predecessor: StorySegment
    continuation: StorySegment
    subject_definitions: str
    retention_analysis: str
    shared: str
    seed: int = 41001


@invocable
async def compare(
    ctx: Context,
    *,
    payload: ComparisonInput,
    out: Outputs,
    decoder: MediaDecoder,
    tel: Telemetry,
) -> Comparison:
    """One shared predecessor, three independently generated continuations, three MP4s."""
    references = payload.references
    seed = payload.seed
    prompts = compile_segments(
        payload.shared, [payload.predecessor, payload.continuation], references,
        subject_definitions=payload.subject_definitions,
        retention_analysis=payload.retention_analysis,
    )
    plans = [plan_continuation(240, context_frames=n) for n in (0, 22, 39, 56)]
    images = await _create_references(ctx, references, tel)
    assets = Assets[Mixed](
        [images[ref.name].with_label(f"Picture {slot}") for slot, ref in enumerate(references, 1)]
    )
    started = time.monotonic()
    first = await motion_segment_turbo(  # type: ignore[call-arg]
        payload=MotionInput(prompt=prompts[0], seed=seed, duration_s=10, steps=8),
        assets=assets,
    )
    tel.log(
        "motion comparison predecessor",
        elapsed_s=time.monotonic() - started,
        sampled_frames=plans[0].sample_frames,
        delivered_frames=240,
        context_digest=first.context.digest,
        video_digest=first.video.digest,
    )
    videos = []
    for plan in plans[1:]:
        ctx.raise_if_cancelled()
        started = time.monotonic()
        following = await motion_segment_turbo(  # type: ignore[call-arg]
            payload=MotionInput(
                prompt=prompts[1],
                seed=seed + 1,
                duration_s=10,
                steps=8,
                context_frames=plan.prefix_frames,  # type: ignore[arg-type]
                context=first.context,
                expected_provenance=first.provenance,
            ),
            assets=assets,
        )
        tel.log(
            "motion comparison continuation",
            context_frames=plan.prefix_frames,
            sampled_frames=plan.sample_frames,
            delivered_frames=240,
            elapsed_s=time.monotonic() - started,
            predecessor_digest=first.video.digest,
            source_context_digest=first.context.digest,
            video_digest=following.video.digest,
            seed=seed + 1,
        )
        joined = assemble(
            AssembleVideoRequest([first.video, following.video], transition="cut"),
            decoder=decoder,
            out=out,
            tel=tel,
            check=ctx.raise_if_cancelled,
        )
        if joined.output_frames != 480:
            raise ValueError("comparison must deliver exactly 20 seconds at 24 fps")
        videos.append(joined.video)
    return Comparison(*videos)


app.job(compare, emits_media=True)
