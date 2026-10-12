"""Restore videos with SeedVR2 while retaining their presentation clock and soundtrack."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from typing import Annotated

import msgspec
import torch
from PIL import Image
from torchvision.transforms import functional as vision
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    DecodedAudioChunk,
    DecodedMediaHeader,
    DecodedVideoFrame,
    MediaDecoder,
    Outputs,
    Telemetry,
    VideoAsset,
    WeightsOutput,
)
from .model import RestorationModel
from .normalize import normalize
from .upstream.temporal import blend_overlapping_frames

app = App()
app.job(normalize, weights=(WeightsOutput("model", max_new_bytes=2 << 20),), accelerator=False)


class Resolution(StrEnum):
    HD = "1080p"
    QHD = "1440p"
    UHD = "2160p"


class UpscaleInput(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
    resolution: Resolution = Resolution.HD
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] = 42


class UpscaleOutput(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=32 << 30)]
    frame_count: int
    fps: float


@app.entrypoint(
    defaults={"model": [{"gpu": "*", "gpus": 1, "lane": "seedvr2@0.1.0/original"}]},
    memoize=True,
)
def upscale(
    ctx: Context,
    payload: UpscaleInput,
    model: RestorationModel,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> UpscaleOutput:
    # 17-frame windows overlap by five frames (12 new frames / three VAE steps).
    # Keep only one window and its tail; the media writer streams the completed frames.
    window, overlap = 17, 5
    count = 0

    def events() -> Iterator[DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk]:
        nonlocal count
        pending: list[DecodedVideoFrame] = []
        audio: list[DecodedAudioChunk] = []
        tail: list[Image.Image] = []
        width = height = 0

        def restore(last: bool) -> Iterator[DecodedVideoFrame]:
            nonlocal tail, count, pending
            ctx.raise_if_cancelled()
            frames = [Image.frombytes("RGB", (f.width, f.height), f.rgb) for f in pending]
            with tel.stage("restoring video window"):
                restored = model.restore(frames, width, height, payload.seed)
            if tail:
                previous = torch.stack([vision.to_tensor(frame).permute(1, 2, 0) for frame in tail])
                current = torch.stack(
                    [vision.to_tensor(frame).permute(1, 2, 0) for frame in restored[:overlap]]
                )
                blended = blend_overlapping_frames(previous, current, len(tail))
                restored[: len(tail)] = [
                    vision.to_pil_image(frame.permute(2, 0, 1)) for frame in blended
                ]
            emit = len(restored) if last else len(restored) - overlap
            for original, image in zip(pending[:emit], restored[:emit], strict=True):
                count += 1
                yield replace(original, width=width, height=height, rgb=image.tobytes())
            tail = [] if last else restored[emit:]
            pending = [] if last else pending[emit:]

        with decoder.stream_video(payload.video) as stream:
            for event in stream:
                ctx.raise_if_cancelled()
                if isinstance(event, DecodedMediaHeader):
                    if event.video is None:
                        raise ValueError("SeedVR2 requires a video stream")
                    edge = int(payload.resolution.value[:-1])
                    scale = edge / min(event.video.width, event.video.height)
                    width = 2 * round(event.video.width * scale / 2)
                    height = 2 * round(event.video.height * scale / 2)
                    yield replace(event, video=replace(event.video, width=width, height=height))
                elif isinstance(event, DecodedVideoFrame):
                    pending.append(event)
                    if len(pending) == window:
                        yield from restore(False)
                        yield from audio
                        audio.clear()
                else:
                    audio.append(event)
            if pending:
                yield from restore(True)
            yield from audio

    saved = out.save_video_stream(events())
    if saved.frame_count != count:
        raise RuntimeError("video encoder changed the restored frame count")
    return UpscaleOutput(saved.video, count, float(saved.frame_rate))
