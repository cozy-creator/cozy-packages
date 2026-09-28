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
    held_frames,
    motion_segment_turbo,
)
from story import StoryReference as StoryReference
from story import compile_segments


class Comparison(msgspec.Struct):
    context_22: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]
    context_39: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]
    context_56: Annotated[VideoAsset, AssetBound(max_bytes=256 << 20, media_types=("video/mp4",))]


class ComparisonInput(msgspec.Struct, forbid_unknown_fields=True):
    references: Annotated[list[StoryReference], msgspec.Meta(min_length=1, max_length=9)]
    predecessor: str
    continuation: str
    overall_soundscape: str = ""
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
    calls = compile_segments(
        [payload.predecessor, payload.continuation],
        references,
        overall_soundscape=payload.overall_soundscape,
    )
    plans = [plan_continuation(240, context_frames=n) for n in (0, 22, 39, 56)]
    images = await _create_references(ctx, references, tel)
    first_assets, next_assets = (
        Assets[Mixed]([images[ref.name] for ref in call.references]) for call in calls
    )
    started = time.monotonic()
    first = await motion_segment_turbo(  # type: ignore[call-arg]
        payload=MotionInput(
            prompt=calls[0].prompt,
            seed=seed,
            duration_s=10,
            steps=8,
            frames=held_frames(plans[0]),
            next_context_frames=(22, 39, 56),
        ),
        assets=first_assets,
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
                prompt=calls[1].prompt,
                seed=seed + 1,
                duration_s=10,
                steps=8,
                frames=held_frames(plan),
                context_frames=plan.prefix_frames,  # type: ignore[arg-type]
                context=first.context,
                expected_provenance=first.provenance,
            ),
            assets=next_assets,
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
