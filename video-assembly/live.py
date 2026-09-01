#!/usr/bin/env python3
"""Real two/eight-shot, segment/master-audio and refusal proof for se-014."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import subprocess
import sys
import tempfile
import time
from array import array
from collections.abc import Iterator
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import cast

import msgspec
from cozy_runtime.author import (
    Asset,
    AudioAsset,
    AuthorError,
    Context,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedMediaStream,
    DecodedVideo,
    DecodedVideoFormat,
    DecodedVideoFrame,
    InvalidRequest,
    MediaDecoder,
    Outputs,
    VideoAsset,
    decode_request,
)
from cozy_runtime.author.fakes import fake_attempt, fake_input

from video_assembly import (
    AssembleVideoRequest,
    AssembleVideoResponse,
    _nearest,
    _scan_video,
    assemble_video,
)

WIDTH = 128
HEIGHT = 96
FRAMES = 24
SAMPLE_RATE = 48_000

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    PASS += int(ok)
    FAIL += int(not ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""))


class NdLike:
    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return self.data


class EventStream(Iterator[DecodedMediaEvent]):
    def __init__(self, events: list[DecodedMediaEvent]) -> None:
        self.events = iter(events)

    def __enter__(self) -> EventStream:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def __next__(self) -> DecodedMediaEvent:
        return next(self.events)


class EventDecoder:
    def __init__(self, events: list[DecodedMediaEvent]) -> None:
        self.events = events

    def stream_video(self, asset: VideoAsset) -> EventStream:
        del asset
        return EventStream(self.events)


class CorruptingDecoder:
    def __init__(self, decoder: MediaDecoder, target: Path, open_number: int) -> None:
        self.decoder = decoder
        self.target = target
        self.open_number = open_number
        self.opens = 0

    def stream_video(self, asset: VideoAsset) -> DecodedMediaStream:
        self.opens += 1
        if self.opens == self.open_number:
            self.target.write_bytes(b"corrupted after the complete scan")
        return self.decoder.stream_video(asset)


def clock_events(*, offset: int = 0, second_duration: int = 1) -> list[DecodedMediaEvent]:
    video = DecodedVideoFormat(
        width=2,
        height=2,
        time_base=Fraction(1, 24),
        pixel_aspect_ratio=Fraction(1, 1),
        nominal_frame_rate=Fraction(24, 1),
        color_primaries=1,
        color_transfer=1,
        color_matrix=1,
        color_range=1,
    )
    return [
        DecodedMediaHeader(video=video, audio=None),
        DecodedVideoFrame(
            width=2,
            height=2,
            rgb=bytes(12),
            pts=offset,
            duration=1,
            time_base=video.time_base,
            pixel_aspect_ratio=video.pixel_aspect_ratio,
            color_primaries=1,
            color_transfer=1,
            color_matrix=1,
            color_range=1,
        ),
        DecodedVideoFrame(
            width=2,
            height=2,
            rgb=bytes(12),
            pts=offset + 1,
            duration=second_duration,
            time_base=video.time_base,
            pixel_aspect_ratio=video.pixel_aspect_ratio,
            color_primaries=1,
            color_transfer=1,
            color_matrix=1,
            color_range=1,
        ),
    ]


def shot(
    root: Path,
    index: int,
    amplitude: float,
    *,
    audio: bool = True,
    key: str = "",
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = 24,
    frame_count: int = FRAMES,
    sample_rate: int = SAMPLE_RATE,
    audio_samples: int | None = None,
    impulse_sample: int | None = None,
) -> VideoAsset:
    source_id = f"source-{index}{('-' + key) if key else ''}"
    out = Outputs(fake_attempt(source_id, spool=root / source_id))
    frames = bytearray()
    for frame_index in range(frame_count):
        pixel_shot = index - 1 if index > 0 and frame_index == 0 else index
        pixel_frame = frame_count - 1 if index > 0 and frame_index == 0 else frame_index
        pixel = bytes(
            (
                (pixel_shot * 29 + pixel_frame * 3) % 256,
                (pixel_shot * 17 + pixel_frame * 5) % 256,
                (pixel_shot * 11 + pixel_frame * 7) % 256,
            )
        )
        frames += pixel * (width * height)
    if audio:
        samples = audio_samples if audio_samples is not None else frame_count * sample_rate // fps
        pcm = array(
            "f",
            (
                amplitude * math.sin(2 * math.pi * (220 + index * 11) * sample / sample_rate)
                for sample in range(samples)
            ),
        )
        if impulse_sample is not None:
            pcm[impulse_sample] = 1.0
        return out.save_video(
            NdLike(bytes(frames), (frame_count, height, width, 3), "uint8"),
            fps=fps,
            audio=NdLike(pcm.tobytes(), (samples,), "float32"),
            sample_rate=sample_rate,
        )
    return out.save_video(NdLike(bytes(frames), (frame_count, height, width, 3), "uint8"), fps=fps)


def master_audio(root: Path, samples: int) -> AudioAsset:
    source_id = f"master-source-{samples}"
    attempt = fake_attempt(source_id, spool=root / source_id)
    pcm = array(
        "f",
        (0.2 * math.sin(2 * math.pi * 330 * sample / SAMPLE_RATE) for sample in range(samples)),
    )
    return Outputs(attempt).save_audio(
        NdLike(pcm.tobytes(), (samples,), "float32"), sample_rate=SAMPLE_RATE
    )


def stored(root: Path, index: int, key: str = "") -> Path:
    """The file a stored shot left in its spool. The driver wrote it, so it can name it
    without reading an asset's local path — a path never leaves the runtime."""
    source_id = f"source-{index}{('-' + key) if key else ''}"
    return next(path for path in sorted((root / source_id).iterdir()) if path.is_file())


def corrupt_video(root: Path) -> VideoAsset:
    source_id = "source-corrupt"
    spool = root / source_id
    spool.mkdir()
    path = spool / "broken.mp4"
    data = b"not an mp4"
    path.write_bytes(data)
    return VideoAsset(
        "source:corrupt",
        media_type="video/mp4",
        size_bytes=len(data),
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        local=path,
        attempt=source_id,
    )


def granted[AssetT: Asset](source: AssetT, attempt_id: str, field: str) -> AssetT:
    """A stored shot as the granted input the executor would hand the handler."""
    return fake_input(source, attempt=attempt_id, input_id=field, max_decoded_bytes=32 << 20)


def run(
    root: Path,
    sources: list[VideoAsset],
    *,
    run_id: str,
    master: AudioAsset | None = None,
    cancel_after_checks: int | None = None,
    corrupt_on_video_open: tuple[int, Path] | None = None,
    decode_output: bool = True,
) -> tuple[AssembleVideoResponse, DecodedVideo | None]:
    attempt = fake_attempt(run_id, spool=root / run_id, max_output_bytes=256 << 20)
    videos = [granted(source, run_id, f"videos.{index}") for index, source in enumerate(sources)]
    audio = granted(master, run_id, "master_audio") if master is not None else None
    base_decoder = MediaDecoder(attempt, active=lambda: False)
    decoder = (
        cast(MediaDecoder, CorruptingDecoder(base_decoder, *reversed(corrupt_on_video_open)))
        if corrupt_on_video_open is not None
        else base_decoder
    )
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return cancel_after_checks is not None and checks >= cancel_after_checks

    response = assemble_video(
        AssembleVideoRequest(videos, audio),
        Context(run_id, time.monotonic() + 300, _cancel=cancelled),
        decoder,
        Outputs(attempt),
    )
    if not decode_output:
        return response, None
    decoded = decoder.decode_video(granted(response.video, run_id, "result.video"))
    return response, decoded


def no_output(root: Path, run_id: str) -> bool:
    return not any(
        path.name.startswith(("video-", ".video-")) for path in (root / run_id).iterdir()
    )


def selected_rgb_digest(root: Path, source: VideoAsset, index: int) -> str:
    run_id = f"rgb-oracle-{index}"
    attempt = fake_attempt(run_id, spool=root / run_id)
    decoded = MediaDecoder(attempt, active=lambda: False).decode_video(
        granted(source, run_id, "video")
    )
    digest = hashlib.sha256()
    for frame in decoded.frames_rgb[int(index > 0) :]:
        digest.update(frame)
    return "sha256:" + digest.hexdigest()


def rss_child(shots: int, *, poison: bool) -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-video-assembly-rss-") as directory:
        root = Path(directory)
        width = 192
        height = 128
        sources = [
            shot(
                root,
                index,
                0.1,
                key="rss",
                width=width,
                height=height,
                frame_count=345,
                sample_rate=32_000,
            )
            for index in range(shots)
        ]
        if poison:
            retained: list[DecodedVideo] = []
            for index, source in enumerate(sources):
                run_id = f"rss-poison-{index}"
                attempt = fake_attempt(run_id, spool=root / run_id)
                retained.append(
                    MediaDecoder(attempt, active=lambda: False).decode_video(
                        granted(source, run_id, "video")
                    )
                )
            assert len(retained) == shots
        else:
            run(root, sources, run_id=f"rss-stream-{shots}", decode_output=False)
        print(
            json.dumps(
                {
                    "shots": shots,
                    "poison": poison,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
        )
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-video-assembly-") as directory:
        root = Path(directory)
        codec_frame_samples = Outputs(
            fake_attempt("codec-policy", spool=root / "codec-policy")
        ).video_audio_frame_samples
        sources = [shot(root, index, 0.08 if index % 2 == 0 else 0.32) for index in range(8)]

        print("section: exact H3 clock")
        cumulative_frames = 0
        h3_samples: list[int] = []
        prior_samples = 0
        for index in range(8):
            cumulative_frames += 345 - int(index > 0)
            cumulative_samples = _nearest(cumulative_frames * Fraction(32_000, 24))
            h3_samples.append(cumulative_samples - prior_samples)
            prior_samples = cumulative_samples
        check(
            "the 32 kHz cumulative clock lands on the fixed 2,753-frame oracle",
            h3_samples == [460000, 458667, 458666, 458667, 458667, 458666, 458667, 458667]
            and sum(h3_samples) == 3_670_667,
            str(h3_samples),
        )
        check(
            "repeated local replay trimming is a planted two-sample red control",
            460_000 + 7 * (460_000 - round(32_000 / 24)) == 3_670_669,
        )
        h3_clock_sources = [
            shot(
                root,
                index,
                0.08 if index % 2 == 0 else 0.32,
                key="h3-clock",
                width=32,
                height=32,
                frame_count=345,
                sample_rate=32_000,
            )
            for index in range(8)
        ]
        h3_clock, h3_clock_decoded = run(root, h3_clock_sources, run_id="h3-clock-eight")
        assert h3_clock_decoded is not None
        check(
            "real 32 kHz media follows the exact eight-shot H3 clock",
            h3_clock.source_frames == 2_760
            and h3_clock.output_frames == h3_clock_decoded.frame_count == 2_753
            and h3_clock.audio_sample_rate == 32_000
            and h3_clock.submitted_audio_samples == 3_670_667
            and [segment.selected_audio_samples for segment in h3_clock.segments] == h3_samples,
        )

        print("section: segment-audio composition")
        two, two_decoded = run(root, sources[:2], run_id="segments-two")
        assert two_decoded is not None
        check(
            "two shots remove exactly one replay frame",
            two.source_frames == 48 and two.output_frames == 47 and two_decoded.frame_count == 47,
        )
        eight, eight_decoded = run(root, sources, run_id="segments-eight")
        assert eight_decoded is not None
        check(
            "eight shots contribute every non-replayed frame once",
            eight.source_frames == 192
            and eight.output_frames == 185
            and eight_decoded.frame_count == 185,
        )
        check(
            "each selected RGB digest independently proves the replay frame choice",
            [segment.selected_rgb_digest for segment in eight.segments]
            == [selected_rgb_digest(root, source, index) for index, source in enumerate(sources)],
        )
        check(
            "seven fixed seam rides are recorded and clamped to nine dB",
            any(
                abs(segment.head_gain_db) > 0 or abs(segment.tail_gain_db) > 0
                for segment in eight.segments
            )
            and all(
                abs(segment.head_gain_db) <= 9 and abs(segment.tail_gain_db) <= 9
                for segment in eight.segments
            ),
        )
        repeated, _ = run(root, sources, run_id="segments-repeat")
        check(
            "the same stored shots reproduce the same assembled bytes",
            eight.video.digest == repeated.video.digest,
            f"{eight.video.digest} vs {repeated.video.digest}",
        )
        check(
            "the mechanical receipt accounts for codecs, audio and final endpoints",
            eight.video_codec == "h264"
            and eight.audio_codec == "aac"
            and eight.audio_sample_rate == SAMPLE_RATE
            and eight.audio_codec_frame_samples == codec_frame_samples
            and abs(eight.av_endpoint_delta_samples) <= codec_frame_samples
            and sum(segment.selected_audio_samples for segment in eight.segments)
            == eight.submitted_audio_samples
            and sum(segment.audio_trimmed_samples for segment in eight.segments)
            == eight.audio_trimmed_samples
            and sum(segment.audio_padded_samples for segment in eight.segments)
            == eight.audio_padded_samples,
        )
        peak_sources = [
            shot(root, 0, 0.5, key="peak-guard", frame_count=72),
            shot(
                root,
                1,
                0.001,
                key="peak-guard",
                frame_count=72,
                impulse_sample=97_000,
            ),
        ]
        peak_guarded, _ = run(root, peak_sources, run_id="peak-guard")
        check(
            "the peak guard covers a rebased head-ride impulse beyond source two seconds",
            any(
                segment.head_gain_clamped or segment.tail_gain_clamped
                for segment in peak_guarded.segments
            )
            and peak_guarded.global_gain_db < 0,
            f"global={peak_guarded.global_gain_db:.3f} dB",
        )

        print("section: exclusive master audio")
        target_frames = 8 * FRAMES - 7
        master_samples = target_frames * SAMPLE_RATE // 24
        master = master_audio(root, master_samples)
        mastered, mastered_decoded = run(root, sources, run_id="master-eight", master=master)
        assert mastered_decoded is not None
        check(
            "master mode replaces segment soundtracks on the exact final clock",
            mastered.audio_mode == "master"
            and mastered.master_audio_digest == master.digest
            and mastered.submitted_audio_samples == master_samples
            and mastered_decoded.soundtrack is not None
            and mastered_decoded.frame_count == target_frames
            and sum(segment.selected_audio_samples for segment in mastered.segments)
            == master_samples
            and all(
                left.selected_audio_source_stop_sample == right.selected_audio_source_start_sample
                for left, right in pairwise(mastered.segments)
            ),
            f"submitted={mastered.submitted_audio_samples}",
        )
        longer = master_audio(root, master_samples + codec_frame_samples)
        trimmed, _ = run(root, sources, run_id="master-trim", master=longer)
        shorter = master_audio(root, master_samples - codec_frame_samples)
        padded, _ = run(root, sources, run_id="master-pad", master=shorter)
        check(
            "master audio trims or pads only inside the recorded tolerance",
            trimmed.audio_trimmed_samples == codec_frame_samples
            and trimmed.audio_padded_samples == 0
            and padded.audio_padded_samples == codec_frame_samples
            and padded.audio_trimmed_samples == 0
            and sum(segment.audio_padded_samples for segment in padded.segments)
            == codec_frame_samples,
        )
        alternate_sources = [
            shot(root, index, 0.01 + index * 0.02, key="alternate-audio") for index in range(8)
        ]
        alternate_mastered, _ = run(
            root, alternate_sources, run_id="master-alternate-segments", master=master
        )
        check(
            "master mode ignores rather than mixes every segment soundtrack",
            alternate_mastered.video.digest == mastered.video.digest,
        )

        print("section: refusals")
        try:
            decode_request(AssembleVideoRequest, msgspec.json.decode(b'{"videos":["only-one"]}'))
        except InvalidRequest:
            check("the wire schema refuses fewer than two videos", True)
        else:
            check("the wire schema refuses fewer than two videos", False)

        silent = shot(root, 99, 0.0, audio=False)
        try:
            run(root, [sources[0], silent], run_id="silent-refusal")
        except AuthorError as exc:
            check(
                "segment mode refuses a missing soundtrack",
                exc.code == "invalid_request",
                f"code={exc.code}",
            )
        else:
            check("segment mode refuses a missing soundtrack", False)

        overlong = shot(
            root,
            98,
            0.1,
            audio_samples=SAMPLE_RATE + codec_frame_samples + 1,
        )
        try:
            run(root, [sources[0], overlong], run_id="segment-clock-refusal")
        except AuthorError as exc:
            check(
                "segment mode refuses overlong audio before output commit",
                exc.code == "invalid_request" and no_output(root, "segment-clock-refusal"),
                f"code={exc.code}",
            )
        else:
            check("segment mode refuses overlong audio before output commit", False)

        master_overlong = master_audio(root, master_samples + codec_frame_samples + 1)
        try:
            run(root, sources, run_id="master-clock-refusal", master=master_overlong)
        except AuthorError as exc:
            check(
                "master mode refuses overlong audio before output commit",
                exc.code == "invalid_request" and no_output(root, "master-clock-refusal"),
                f"code={exc.code}",
            )
        else:
            check("master mode refuses overlong audio before output commit", False)

        wrong_geometry = shot(root, 97, 0.1, width=WIDTH + 2)
        try:
            run(root, [sources[0], wrong_geometry], run_id="geometry-refusal")
        except AuthorError as exc:
            check(
                "geometry mismatch refuses before output commit",
                exc.code == "invalid_request" and no_output(root, "geometry-refusal"),
                f"code={exc.code}",
            )
        else:
            check("geometry mismatch refuses before output commit", False)

        wrong_rate = shot(root, 96, 0.1, fps=30)
        try:
            run(root, [sources[0], wrong_rate], run_id="rate-refusal")
        except AuthorError as exc:
            check(
                "frame-rate mismatch refuses before output commit",
                exc.code == "invalid_request" and no_output(root, "rate-refusal"),
                f"code={exc.code}",
            )
        else:
            check("frame-rate mismatch refuses before output commit", False)

        clock_refusals = []
        for events in (clock_events(offset=1), clock_events(second_duration=2)):
            try:
                _scan_video(
                    cast(MediaDecoder, EventDecoder(events)), VideoAsset("clock"), lambda: None
                )
            except AuthorError as exc:
                clock_refusals.append(exc.code == "invalid_request")
            else:
                clock_refusals.append(False)
        check(
            "nonzero and variable frame clocks refuse instead of being retimed",
            clock_refusals == [True, True],
        )

        try:
            run(root, [sources[0], corrupt_video(root)], run_id="corrupt-late-refusal")
        except AuthorError as exc:
            check(
                "a corrupt later input refuses before output commit",
                exc.code == "media_malformed" and no_output(root, "corrupt-late-refusal"),
                f"code={exc.code}",
            )
        else:
            check("a corrupt later input refuses before output commit", False)

        late_sources = [shot(root, index, 0.1, key="late-corruption") for index in range(2)]
        try:
            run(
                root,
                late_sources,
                run_id="corrupt-after-scan",
                corrupt_on_video_open=(4, stored(root, 1, "late-corruption")),
            )
        except AuthorError as exc:
            check(
                "second-pass late corruption removes the partial mux and preserves its field",
                exc.code == "media_malformed"
                and exc.fields == ("videos.1",)
                and no_output(root, "corrupt-after-scan"),
                f"code={exc.code} fields={exc.fields}",
            )
        else:
            check("second-pass late corruption removes the partial mux", False)

        try:
            run(root, sources[:2], run_id="cancel-refusal", cancel_after_checks=150)
        except AuthorError as exc:
            check(
                "cooperative cancellation leaves no final or partial output",
                exc.code == "cancelled" and no_output(root, "cancel-refusal"),
                f"code={exc.code}",
            )
        else:
            check("cooperative cancellation leaves no final or partial output", False)

        reordered, _ = run(root, list(reversed(sources)), run_id="reordered")
        check(
            "input order is composition meaning",
            reordered.video.digest != eight.video.digest,
        )

        print("section: fresh-process memory scaling")
        rss: dict[tuple[int, bool], int] = {}
        for shots, poison in ((2, False), (8, False), (8, True)):
            command = [sys.executable, __file__, "--rss-child", str(shots)]
            if poison:
                command.append("--poison")
            child = subprocess.run(command, check=True, capture_output=True, text=True)
            record = json.loads(child.stdout.splitlines()[-1])
            rss[(shots, poison)] = int(record["max_rss_kib"])
        check(
            "streaming RSS stays shot-count bounded and the snapshot poison goes red",
            rss[(8, False)] - rss[(2, False)] < 48 * 1024
            and rss[(8, True)] - rss[(8, False)] > 48 * 1024,
            f"2={rss[(2, False)]} KiB 8={rss[(8, False)]} KiB poison={rss[(8, True)]} KiB",
        )

    print(f"\nvideo assembly: {PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--rss-child":
        raise SystemExit(rss_child(int(sys.argv[2]), poison="--poison" in sys.argv[3:]))
    raise SystemExit(main())
