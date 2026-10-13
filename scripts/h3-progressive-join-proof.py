#!/usr/bin/env python3
"""The incremental cut join, its live HLS revisions, and completed seekable MP4.

Synthetic pictures, real codecs: each segment is an H3-shaped fragmented MP4 (H.264 + AAC).
The proof shows that
- `assembly.CutJoin` decodes to exactly the frames and samples `assemble` writes;
- its continuous join keeps each soundtrack whole, so sound cut on its own clock (H3
  continuation) stays one continuous signal, and refuses a soundtrack off the picture;
- its last revision is the joined video byte for byte;
- live revisions retain immutable fragmented prefixes and play as an HLS EVENT
  playlist; the final indexed container preserves encoded packets and decoded frames,
  with accurate duration and seeks between keyframes.

    python scripts/h3-progressive-join-proof.py <empty work directory>
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import msgspec
import numpy as np
from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    MediaDecoder,
    Outputs,
    Telemetry,
    VideoAsset,
    attempt,
    fakes,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput, file_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import assembly

FPS, RATE = 24, 48000
FRAMES = (30, 41, 50)
#: Continuation's sound/picture offsets, in samples: short at a join, made up by the next head.
SOUND = (30 * 2000 - 400, 41 * 2000 + 400 + 1600, 50 * 2000 - 1600)


class Joined(msgspec.Struct):
    """The joined video; `video` is the output each revision is published to."""

    video: VideoAsset
    samples: int


@invocable
async def progressive(
    ctx: Context,
    *,
    payload: assembly.AssembleVideoRequest,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> Joined:
    """long_form's join: one revision published per segment, the joined video returned."""
    cut = assembly.CutJoin(decoder, out, ctx.raise_if_cancelled)
    with assembly.ScanAhead(decoder, ctx.raise_if_cancelled, pictures=False) as scanning:
        for index, video in enumerate(payload.videos):
            scanning.add(video)
            last = index + 1 == len(payload.videos)
            revision = cut.append(scanning.result(index), FRAMES[index], last=last)
            out.publish("video", revision, label=f"Video (segments 1-{index + 1})")
        joined = cut.finish()
    return Joined(joined.video, joined.submitted_audio_samples)


@invocable
async def continuous(
    ctx: Context,
    *,
    payload: assembly.AssembleVideoRequest,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> Joined:
    """long_form's join of continuation segments."""
    cut = assembly.CutJoin(decoder, out, ctx.raise_if_cancelled, continuous=True)
    with assembly.ScanAhead(decoder, ctx.raise_if_cancelled, pictures=False) as scanning:
        for index, video in enumerate(payload.videos):
            scanning.add(video)
            cut.append(scanning.result(index), FRAMES[index], last=index + 1 == len(FRAMES))
        joined = cut.finish()
    return Joined(joined.video, joined.submitted_audio_samples)


def tone(samples: int) -> np.ndarray:
    return np.sin(np.arange(samples, dtype=np.float64) * (2 * np.pi * 220 / RATE)) * 0.1


def continuous_fixtures(root: Path, sound: tuple[int, ...]) -> list[Path]:
    """One unbroken tone, cut at the given sample counts, under each segment's pictures."""
    root.mkdir()
    signal = tone(sum(sound)).astype(np.float32)
    paths, start = [], 0
    for index, (frames, samples) in enumerate(zip(FRAMES, sound, strict=True)):
        work = root / f"continuous-{index}"
        work.mkdir()
        out = fakes.fake_outputs(fakes.fake_attempt(f"continuous-{index}", spool=work))
        pixels = np.full((frames, 64, 96, 3), 60 + 40 * index, dtype=np.uint8)
        piece = signal[start : start + samples]
        start += samples
        clip = out.save_video(pixels, fps=FPS, audio=np.stack([piece, piece]), sample_rate=RATE)
        path = root / f"continuous-{index}.mp4"
        path.write_bytes(clip.read_bytes())
        paths.append(path)
    return paths


def fixtures(root: Path) -> list[Path]:
    paths = []
    for index, frames in enumerate(FRAMES):
        work = root / f"fixture-{index}"
        work.mkdir()
        out = fakes.fake_outputs(fakes.fake_attempt(f"fixture-{index}", spool=work))
        pixels = np.zeros((frames, 64, 96, 3), dtype=np.uint8)
        pixels[..., index % 3] = 150
        for frame in range(frames):
            pixels[frame, :, frame % 90 : frame % 90 + 6, 2] = 230
        samples = round(frames * RATE / FPS)
        signal = np.sin(
            np.arange(samples, dtype=np.float32) * (2 * np.pi * (220 + 40 * index) / RATE)
        )
        clip = out.save_video(
            pixels, fps=FPS, audio=np.stack([signal, signal]) * 0.1, sample_rate=RATE
        )
        path = root / f"{index}.mp4"
        path.write_bytes(clip.read_bytes())
        paths.append(path)
    return paths


def run(
    root: Path, label: str, handler: Callable[..., Any], paths: list[Path], *, ok: bool = True
) -> tuple[Any, Any]:
    work = root / label
    work.mkdir()
    assets, videos = {}, []
    for index, path in enumerate(paths):
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"payload.videos.{index}"
        assets[key] = GrantedInput(
            input_id=key,
            local=path,
            media_type="video/mp4",
            digest=digest,
            length=path.stat().st_size,
            file_state=file_state(path),
            order=index,
        )
        videos.append(digest)
    app = App()
    app.job(handler, emits_media=True)
    result, outcome, record = attempt(
        app.get(handler.__name__),
        {"payload": {"videos": videos, "transition": "cut"}},
        Invocation(label, work, time.monotonic() + 300, assets=assets),
    )
    assert (outcome.terminal == "succeeded") == ok, outcome
    return result, record


def decoded(path: Path) -> tuple[list[bytes], bytes]:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray().tobytes() for frame in container.decode(video=0)]
    with av.open(str(path)) as container:
        audio = b"".join(frame.to_ndarray().tobytes() for frame in container.decode(audio=0))
    return frames, audio


def decoded_audio(data: bytes, path: Path) -> np.ndarray:
    path.write_bytes(data)
    with av.open(str(path)) as container:
        return np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)


def video_packets(path: Path) -> list[tuple[bytes, Fraction]]:
    with av.open(str(path)) as container:
        return [
            (bytes(packet), Fraction(packet.pts) * packet.time_base)
            for packet in container.demux(container.streams.video[0])
            if packet.size and packet.pts is not None
        ]


def probe(target: str) -> dict[str, Any]:
    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,nb_read_frames:format=duration",
            "-of",
            "json",
            target,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    facts: dict[str, Any] = json.loads(raw)
    return facts


def continuous_join(root: Path) -> None:
    joined, _ = run(root, "continuous", continuous, continuous_fixtures(root / "c", SOUND))
    assert joined.result.samples == sum(SOUND) == sum(FRAMES) * RATE // FPS, joined.result
    sound = decoded_audio(joined.result.video.read_bytes(), root / "continuous.mp4")[0]
    # PyAV keeps the codec's whole frames: find where the tone starts.
    reference = tone(sum(SOUND))
    start = min(range(len(sound) - len(reference) + 1),
                key=lambda k: float(np.abs(sound[k : k + 4800] - reference[:4800]).mean()))
    error = np.abs(sound[start : start + len(reference)] - reference)
    seams = list(itertools.accumulate(SOUND[:-1]))
    near = max(error[max(0, j - 256) : j + 256].max() for j in seams)
    # Each segment's AAC edge costs ~0.01 here; a sound offset by its 400-sample short end,
    # the clocked join's zero pad, would cost the tone's whole 0.1 amplitude.
    assert near < 0.03, (near, np.percentile(error, 99))
    drifted = (SOUND[0] - 2001, SOUND[1] + 2001, SOUND[2])
    run(root, "continuous-drift", continuous, continuous_fixtures(root / "d", drifted), ok=False)
    print(f"continuous join: {len(SOUND)} soundtracks whole, seam error {near:.4f}", flush=True)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    continuous_join(root)
    paths = fixtures(root)
    reference, _ = run(root, "one-shot", assembly.assemble_video, paths)
    joined, record = run(root, "progressive", progressive, paths)
    one_shot = root / "one-shot.mp4"
    one_shot.write_bytes(reference.result.video.read_bytes())
    final = root / "final.mp4"
    final.write_bytes(joined.result.video.read_bytes())
    final_decoded = decoded(final)
    assert final_decoded == decoded(one_shot), "the incremental join decodes differently"
    final_packets = video_packets(final)
    assert final_packets == video_packets(one_shot), "finalization changed encoded video"

    spool = root / "progressive"
    revisions = []
    previous_live = b""
    for index, publish in enumerate(record.published):
        data = b"".join((spool / part.local).read_bytes() for part in publish.parts)
        path = root / f"revision-{index + 1}.mp4"
        path.write_bytes(data)
        facts = probe(str(path))
        video = next(s for s in facts["streams"] if s["codec_type"] == "video")
        frames = sum(FRAMES[: index + 1])
        assert int(video["nb_read_frames"]) == frames, facts
        assert decoded(path)[0] == final_decoded[0][:frames], "revision changed decoded frames"
        assert video_packets(path) == final_packets[:frames], "revision changed video packets"
        if index + 1 < len(FRAMES):
            assert data.startswith(previous_live), "live fragment prefix changed"
            previous_live = data
        revisions.append(publish)
    assert final.read_bytes() == (root / f"revision-{len(FRAMES)}.mp4").read_bytes()

    final_duration = float(probe(str(final))["format"]["duration"])
    assert abs(final_duration - sum(FRAMES) / FPS) < 1 / FPS
    for frame_index in (1, FRAMES[0] - 1, FRAMES[0] + 1, sum(FRAMES) - 2):
        with av.open(str(final)) as container:
            video_stream = container.streams.video[0]
            assert video_stream.time_base is not None
            target = Fraction(frame_index, FPS)
            container.seek(int(target / video_stream.time_base), stream=video_stream, backward=True)
            frame = next(
                frame for frame in container.decode(video=0)
                if frame.pts is not None and frame.time_base is not None
                and frame.pts * frame.time_base >= target
            )
            assert frame.to_ndarray().tobytes() == final_decoded[0][frame_index]

    # Completion replaces the fragmented preview with one regular MP4. The last
    # live revision remains available and immutable for an attached HLS observer.
    assert len(revisions[-1].parts) == 1 and revisions[-1].parts[0].duration_us > 0
    parts = revisions[-2].parts
    assert parts[0].duration_us == 0 and len(parts) == len(FRAMES)
    # As Calcifer serves them: players take an init `.mp4` and media `.m4s` segments.
    stream = root / "stream"
    stream.mkdir()
    (stream / "init.mp4").write_bytes((spool / parts[0].local).read_bytes())
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        "#EXT-X-PLAYLIST-TYPE:EVENT",
        f"#EXT-X-TARGETDURATION:{max(-(-p.duration_us // 1_000_000) for p in parts[1:])}",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        '#EXT-X-MAP:URI="init.mp4"',
    ]
    for index, part in enumerate(parts[1:]):
        (stream / f"{index}.m4s").write_bytes((spool / part.local).read_bytes())
        lines += [f"#EXTINF:{part.duration_us / 1e6:.6f},", f"{index}.m4s"]
    playlist = stream / "video.m3u8"
    playlist.write_text("\n".join([*lines, "#EXT-X-ENDLIST", ""]))
    facts = probe(str(playlist))
    counted = {s["codec_type"]: int(s["nb_read_frames"]) for s in facts["streams"]}
    assert counted["video"] == sum(FRAMES[:-1]), facts
    duration = float(facts["format"]["duration"])
    assert abs(duration - sum(FRAMES[:-1]) / FPS) < 1 / FPS, facts
    stamps = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(playlist),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    times = sorted(float(value) for value in stamps)
    steps = [b - a for a, b in itertools.pairwise(times)]
    assert all(abs(step - 1 / FPS) < 1e-4 for step in steps), f"a gap at a boundary: {steps}"
    evidence = {
        "segments": len(FRAMES),
        "frames": sum(FRAMES),
        "hls_duration_s": duration,
        "completed_duration_s": final_duration,
        "revision_bytes": [
            (root / f"revision-{i + 1}.mp4").stat().st_size for i in range(len(FRAMES))
        ],
        "decoded_equal_to_one_shot_join": True,
        "final_is_last_revision_byte_for_byte": True,
        "hls_gapless": True,
        "final_packets_unchanged": True,
        "final_seeks_match_decoded_frames": True,
        "continuous_join_keeps_each_soundtrack_whole": True,
        "continuous_join_refuses_sound_a_frame_off": True,
        "actual_h3_inference": False,
    }
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
