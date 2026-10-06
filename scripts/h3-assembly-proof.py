#!/usr/bin/env python3
"""Real codec proof; synthetic pictures are not H3 inference. Custody of the files is the
machine's (Runtime 0.19 ships a plane-only TensorFS without a Store), so it is not proved here."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from array import array
from pathlib import Path
from typing import Any

import av
import numpy as np
from cozy_runtime.author import (
    App,
    Context,
    InvalidRequest,
    Invocation,
    MediaDecoder,
    Outputs,
    Telemetry,
    attempt,
    fakes,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput, file_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import assembly

FPS, RATE, FRAMES = 24, 32000, 362


@invocable
async def scanned_assembly(
    ctx: Context,
    *,
    payload: assembly.AssembleVideoRequest,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> assembly.AssembleVideoResponse:
    pictures = payload.transition == "continuous"  # a cut's pictures are copied, not scanned
    with assembly.ScanAhead(decoder, ctx.raise_if_cancelled, pictures=pictures) as scanning:
        for video in payload.videos:
            scanning.add(video)
        scans = scanning.finish(payload.videos)
        try:
            scanning.finish([])
        except InvalidRequest:
            pass
        else:
            raise AssertionError("scan results were accepted for different inputs")
        for scan in scans:
            audio = scan.header.audio
            assert audio is not None
            # Only two half-second audio windows plus the replay trim are retained.
            assert sum(map(len, (*scan.audio_head, *scan.audio_tail))) <= (
                audio.channels * (audio.sample_rate + math.ceil(audio.sample_rate / FPS)) * 4
            )
        if len(payload.videos) == 8:
            # Neither composition mode imposes an eight-shot count limit.
            with assembly.ScanAhead(
                decoder, ctx.raise_if_cancelled, pictures=pictures
            ) as unbounded:
                for video in payload.videos:
                    unbounded.add(video)
                unbounded.add(payload.videos[0])
                assert len(unbounded.finish([*payload.videos, payload.videos[0]])) == 9
        return assembly.assemble(
            payload,
            decoder=decoder,
            out=out,
            tel=tel,
            check=ctx.raise_if_cancelled,
            scanned=scanning,
        )


def unchanged_gain_bytes() -> None:
    """Compare against the original scalar arithmetic, including overlapping rides."""
    rate = 32000
    generator = np.random.default_rng(1234)
    pcm = tuple(generator.uniform(-0.9, 0.9, 1024).astype("<f4").tobytes() for _ in range(2))
    for total in (32000, 160000):
        for start in (0, 1024, 65000, total - 1024):
            for head, tail in ((0.0, 0.0), (0.0, 3.0), (-9.0, 9.0)):
                for global_gain in (1.0, 0.42):
                    gains = assembly._Gains(head, tail)
                    expected = []
                    ride = rate * 2
                    for raw in pcm:
                        values = array("f")
                        values.frombytes(raw)
                        for offset, value in enumerate(values):
                            position, db = start + offset, 0.0
                            if head and position < ride:
                                phase = position / (ride - 1)
                                db += head * (1 + math.cos(math.pi * phase)) / 2
                            if tail and position >= max(0, total - ride):
                                phase = (position - max(0, total - ride)) / (ride - 1)
                                db += tail * (1 - math.cos(math.pi * phase)) / 2
                            values[offset] = float(value) * (10 ** (db / 20)) * global_gain
                        expected.append(values.tobytes())
                    assert assembly._gain_pcm(
                        pcm,
                        start=start,
                        total=total,
                        rate=rate,
                        gains=gains,
                        global_gain=global_gain,
                    ) == tuple(expected)


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def fixtures(root: Path) -> dict[str, Path]:
    source = root / "source"
    source.mkdir()
    files = {}
    for index in range(2):
        work = source / str(index)
        work.mkdir()
        out = fakes.fake_outputs(fakes.fake_attempt(f"fixture-{index}", spool=work))
        pixels = np.zeros((FRAMES, 64, 96, 3), dtype=np.uint8)
        pixels[..., index] = 150
        for frame in range(FRAMES):
            pixels[frame, :, frame % 90 : frame % 90 + 6, 2] = 230
        samples = round(FRAMES * RATE / FPS)
        signal = np.sin(np.arange(samples, dtype=np.float32) * (2 * np.pi * 220 / RATE))
        audio = np.stack([signal, signal]) * (0.05 if index == 0 else 0.12)
        clip = out.save_video(pixels, fps=FPS, audio=audio, sample_rate=RATE)
        path = source / f"{index}.mp4"
        path.write_bytes(clip.read_bytes())
        files[f"{index}.mp4"] = path
    audio_out = fakes.fake_outputs(fakes.fake_attempt("master-fixture", spool=source / "masters"))
    for name, samples in (
        ("master.flac", (2 * FRAMES - 1) * RATE // FPS),
        ("wrong-master.flac", 45 * RATE),
    ):
        encoded_audio = audio_out.save_audio(np.zeros((2, samples), np.float32), sample_rate=RATE)
        path = source / name
        path.write_bytes(encoded_audio.read_bytes())
        files[name] = path
    return files


def execute(
    root: Path,
    files: dict[str, Path],
    count: int,
    *,
    master: str | None = None,
    changed: bool = False,
    scanned: bool = False,
    transition: str = "continuous",
) -> tuple[Any, Any, Path]:
    label = f"{count}-{'changed' if changed else master or 'segments'}-{transition}"
    work = root / (label + ("-scanned" if scanned else ""))
    work.mkdir()
    assets: dict[str, GrantedInput] = {}
    videos = []
    for index in range(count):
        path = files[f"{index % 2}.mp4"]
        if changed and index == 0:
            changed_path = work / "changed-input.mp4"
            changed_path.write_bytes(path.read_bytes())
            path = changed_path
        digest = sha(path.read_bytes())
        videos.append(digest)
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
    master_digest = None
    if master:
        path = files[master]
        master_digest = sha(path.read_bytes())
        assets["payload.master_audio"] = GrantedInput(
            input_id="payload.master_audio",
            local=path,
            media_type="audio/flac",
            digest=master_digest,
            length=path.stat().st_size,
            file_state=file_state(path),
        )
    if changed:
        path = assets["payload.videos.0"].local
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
    app = App()
    handler = scanned_assembly if scanned else assembly.assemble_video
    app.job(handler, emits_media=True)
    result, outcome, _ = attempt(
        app.get(handler.__name__),
        {"payload": {"videos": videos, "master_audio": master_digest, "transition": transition}},
        Invocation(label, work, time.monotonic() + 180, assets=assets),
    )
    return result, outcome, work


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    files = fixtures(root)
    unchanged_gain_bytes()
    evidence = []
    for count in (1, 2, 4, 8):
        result, outcome, work = execute(root, files, count)
        assert outcome.terminal == "succeeded", outcome
        response = result.result
        cached, cached_outcome, _ = execute(root, files, count, scanned=True)
        assert cached_outcome.terminal == "succeeded", cached_outcome
        assert cached.result.video.read_bytes() == response.video.read_bytes()
        assert cached.result.segments == response.segments
        expected = count * FRAMES - (count - 1)
        assert response.source_frames == count * FRAMES
        assert response.output_frames == expected
        assert response.replay_frames_removed == count - 1
        assert response.submitted_audio_samples == round(expected * RATE / FPS)
        assert abs(response.av_endpoint_delta_samples) <= response.audio_codec_frame_samples
        recovered = work / "delivered.mp4"
        recovered.write_bytes(response.video.read_bytes())
        digest = sha(recovered.read_bytes())
        with av.open(str(recovered)) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.name == "h264"
            assert sum(1 for _ in container.decode(video=0)) == expected
        evidence.append(
            {
                "shots": count,
                "frames": expected,
                "seconds": expected / FPS,
                "audio_samples": response.submitted_audio_samples,
                "endpoint_delta_samples": response.av_endpoint_delta_samples,
                "video_digest": digest,
                "scan_ahead_preserves_video_bytes": True,
                "scan_audio_windows_bounded": True,
                "actual_h3_inference": False,
            }
        )
        print(json.dumps(evidence[-1]), flush=True)
    for count in (1, 2, 8):
        # Cuts copy every shot's H.264 packets; only the soundtrack is encoded.
        result, outcome, work = execute(root, files, count, transition="cut")
        assert outcome.terminal == "succeeded", outcome
        response = result.result
        scanned_result, scanned_outcome, _ = execute(
            root, files, count, scanned=True, transition="cut"
        )
        assert scanned_outcome.terminal == "succeeded", scanned_outcome
        assert scanned_result.result.video.read_bytes() == response.video.read_bytes()
        assert response.output_frames == count * FRAMES and response.replay_frames_removed == 0
        assert response.submitted_audio_samples == round(count * FRAMES * RATE / FPS)
        assert abs(response.av_endpoint_delta_samples) <= response.audio_codec_frame_samples
        path = work / "joined.mp4"
        path.write_bytes(response.video.read_bytes())
        with av.open(str(path)) as container:
            joined = [bytes(p) for p in container.demux(container.streams.video[0]) if p.size]
        sources = []
        for index in range(count):
            with av.open(str(files[f"{index % 2}.mp4"])) as container:
                video = container.streams.video[0]
                sources += [bytes(p) for p in container.demux(video) if p.size]
        assert joined == sources, "a cut re-encoded its pictures"
        print(json.dumps({"cut_shots": count, "frames": response.output_frames}), flush=True)
    result, outcome, _ = execute(root, files, 2, master="master.flac")
    assert outcome.terminal == "succeeded", outcome
    assert result.result.audio_mode == "master"
    assert all(s.head_gain_db == s.tail_gain_db == 0 for s in result.result.segments)
    _, wrong, _ = execute(root, files, 2, master="wrong-master.flac")
    assert wrong.terminal != "succeeded" and wrong.code == "invalid_request", wrong
    _, changed, _ = execute(root, files, 1, changed=True)
    assert changed.terminal != "succeeded" and changed.code == "input_changed", changed
    (root / "evidence.json").write_text(
        json.dumps(
            {
                "codec_native_arms": evidence,
                "gain_fast_path_matches_original_bytes": True,
                "cuts_copy_packets": True,
                "master_exact": True,
                "master_mismatch_refused": True,
                "changed_input_refused": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("H3 assembly codec proof passed; no H3 model was loaded")


if __name__ == "__main__":
    main()
