"""Restore videos with SeedVR2 while retaining their presentation clock and soundtrack."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from typing import Annotated, Literal

import msgspec
import torch
from PIL import Image, ImageOps
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
    Shape,
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


_RESOLUTIONS: dict[Resolution, tuple[int, int]] = {
    Resolution.HD: (1920, 1080),
    Resolution.QHD: (2560, 1440),
    Resolution.UHD: (3840, 2160),
}


class UpscaleInput(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
    resolution: Annotated[Resolution, Shape(pixels=_RESOLUTIONS)] = Resolution.HD
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] = 42
    aspect_ratio: Literal["source", "16:9"] = "source"


class UpscaleOutput(msgspec.Struct):
    video: Annotated[
        VideoAsset,
        AssetBound(
            max_bytes=1 << 30,
            max_decoded_bytes=32 << 30,
            media_types=("video/mp4",),
        ),
    ]
    frame_count: int
    fps: float


def output_sizes(
    width: int, height: int, edge: int, aspect_ratio: Literal["source", "16:9"]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Restore at the fitted size; the encoder receives the requested even-sized canvas."""
    canvas = (edge * 16 // 9, edge)
    scale = min(canvas[0] / width, canvas[1] / height)
    restored = (max(2, 2 * round(width * scale / 2)), max(2, 2 * round(height * scale / 2)))
    return restored, canvas if aspect_ratio == "16:9" else restored


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
        canvas = (0, 0)

        def restore(last: bool) -> Iterator[DecodedVideoFrame]:
            nonlocal tail, count, pending
            ctx.raise_if_cancelled()
            if last and len(pending) == len(tail):
                restored = tail
                tail = []
            else:
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
                image = ImageOps.pad(image, canvas, method=Image.Resampling.LANCZOS, color="black")
                yield replace(original, width=canvas[0], height=canvas[1], rgb=image.tobytes())
            tail = [] if last else restored[emit:]
            pending = [] if last else pending[emit:]

        with decoder.stream_video(payload.video) as stream:
            for event in stream:
                ctx.raise_if_cancelled()
                if isinstance(event, DecodedMediaHeader):
                    if event.video is None:
                        raise ValueError("SeedVR2 requires a video stream")
                    edge = int(payload.resolution.value[:-1])
                    (width, height), canvas = output_sizes(
                        event.video.width, event.video.height, edge, payload.aspect_ratio
                    )
                    yield replace(
                        event, video=replace(event.video, width=canvas[0], height=canvas[1])
                    )
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
