#!/usr/bin/env python3
"""Two-file long-form delivery through Runtime's broker and real codecs.

The CPU renderer produces synthetic clips with the real served clock. Runtime's own
tests prove native host projection enforcement; this fixture returns explicit byte grants.
This proves composition and output custody, not H3 rendering or ordinary CLI qualification.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event
from threading import enumerate as threads
from typing import Any, Literal, cast

import av
import msgspec
import numpy as np
from cozy_runtime.author import (
    App,
    Context,
    ImageAsset,
    ImageFrame,
    Invocation,
    MediaDecoder,
    Outputs,
    Telemetry,
    VideoAsset,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput, file_state
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._codec import encode_frame
from cozy_runtime.author._services import ProgressFrame, settle_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
from cozy_runtime.models.minimax_h3.official import FPS, frames_for

import assembly
import h3
from long_form_state import RenderProvenance, compatible

WIDTH, HEIGHT, RATE = 96, 64, 32000
SHOT_INTERFACE = "sha256:" + "5c" * 32
PROVENANCE = RenderProvenance("sha256:" + "11" * 32)

TURBO_PROVENANCE = msgspec.structs.replace(PROVENANCE, turbo_lora_manifest="sha256:" + "44" * 32)


@invocable
async def segment(
    ctx: Context, *, payload: h3.SegmentInput, progress_steps: int, out: Outputs, tel: Telemetry
) -> h3.SegmentOutput:
    """CPU stand-in; the real segment is a nonmemoized serving entrypoint."""
    compatible(PROVENANCE, payload.expected_provenance)
    ctx.raise_if_cancelled()
    # A genuine author callback, forwarded while the child poll is still pending.
    tel.step_callback(progress_steps, stage="denoise", overall_range=(0.15, 0.85))(2)
    count = frames_for(payload.duration_s)
    pixels = np.zeros((count, HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[..., 0] = payload.seed % 251
    for index in range(count):
        pixels[index, :, index % 90 : index % 90 + 6, 2] = 230
    frame = out.save_image(ImageFrame(WIDTH, HEIGHT, pixels[-1].tobytes()), format="png")
    audio = np.zeros((2, round(count * RATE / FPS)), np.float32)
    video = out.save_video(pixels, fps=FPS, audio=audio, sample_rate=RATE)
    return h3.SegmentOutput(video, frame, [], PROVENANCE)


child_app = App()
child_app.job(segment, emits_media=True)


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _drive(
    root: Path,
    shots: list[h3.Shot],
    *,
    opening: ImageAsset | None = None,
    fail_at: int = -1,
    cancel_at: int = -1,
    observed: RenderProvenance | None = None,
    mode: Literal["turbo", "standard"] = "standard",
    steps: int | None = None,
    overlap: bool = False,
    request_id: str | None = None,
    progress: list[ProgressFrame] | None = None,
    fail_after_progress_at: int = -1,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    root.mkdir()
    spool = root / "bytes"
    spool.mkdir()
    parent = root.name if request_id is None else request_id
    child_name = "segment_turbo" if mode == "turbo" else "segment"
    real = next(surface for surface in describe(h3.app) if surface.name == child_name)
    observed = observed or (TURBO_PROVENANCE if mode == "turbo" else PROVENANCE)
    calls: list[dict[str, Any]] = []
    answers: dict[int, str] = {}
    byte_grants: dict[int, list[dict[str, Any]]] = {}
    child_frames: dict[int, str] = {}
    child_progress: dict[int, dict[str, Any]] = {}
    cancelled = False
    scan_started, child_started, scan_finished = Event(), Event(), Event()
    original_scan = assembly._scan_video

    def observe(event: Any) -> None:
        if progress is not None and isinstance(event, ProgressFrame):
            progress.append(event)

    def scan(decoder: MediaDecoder, asset: VideoAsset, check: Callable[[], None]) -> assembly._Scan:
        if overlap and not scan_started.is_set():
            scan_started.set()
            assert child_started.wait(10), "scan did not overlap a later child"
            result = original_scan(decoder, asset, check)
            scan_finished.set()
            return result
        return original_scan(decoder, asset, check)

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal cancelled
        if kind == "child_call":
            index = int(value["call_index"])
            calls.append(value)
            if overlap and index == 1:
                child_started.set()
                assert scan_started.wait(10), "validation was deferred until all children finished"
                if index != cancel_at:
                    assert scan_finished.wait(10), (
                        "validation cannot progress while child is active"
                    )
            if index in (fail_at, cancel_at):
                cancelled = index == cancel_at
                return {
                    "ok": False,
                    "code": "child.failed",
                    "detail": "renderer refused",
                    "child_request_id": f"{parent}-child-{index}",
                }
            document = json.loads(value["payload"])
            for slot in ("base_model", "turbo_lora") if mode == "turbo" else ("model",):
                assert document.pop(slot) is None, document
            if mode == "turbo":
                assert "steps" not in document["payload"]
                # The CPU stand-in below uses the standard schema only for codec generation.
                document["payload"]["steps"] = 30
                document["payload"]["expected_provenance"] = None
            assert list(document) == ["payload"], document
            document["progress_steps"] = 8 if mode == "turbo" else document["payload"]["steps"]
            # The self-call keeps the omitted model slot. Runtime resolves its frozen
            # default; this CPU stand-in has no model and consumes only the shot payload.
            incoming = document["payload"].get("first_frame")
            grants = {}
            if incoming:
                local = spool / f"{incoming[7:]}.png"
                if not local.exists():
                    assert opening is not None and incoming == sha(opening.read_bytes())
                    local = root / "opening.png"
                grants["payload.first_frame"] = GrantedInput(
                    input_id="payload.first_frame",
                    local=local,
                    media_type="image/png",
                    digest=incoming,
                    length=local.stat().st_size,
                    file_state=file_state(local),
                )
            work = root / f"child-{index}"
            emitted: list[Any] = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                produced, outcome, record = pool.submit(
                    attempt,
                    child_app.get("segment"),
                    document,
                    Invocation(
                        f"{parent}-child-{index}",
                        work,
                        time.monotonic() + 60,
                        assets=grants,
                        progress=emitted.append,
                    ),
                ).result()
            assert outcome.terminal == "succeeded", outcome
            assert produced is not None
            step = next(event for event in emitted if isinstance(event, ProgressFrame))
            child_progress[index] = {"sequence": 1, "payload": asdict(step)}
            response: dict[str, Any] = {"warnings": [], "provenance": msgspec.to_builtins(observed)}
            outputs: list[dict[str, Any]] = []
            for field, suffix in (("video", ".mp4"), ("continuation_frame", ".png")):
                asset = getattr(produced.result, field)
                frame = record.frames[asset.ref]
                raw = encode_frame(frame.codec, frame.facts, frame.raw.read_bytes())
                digest = sha(raw)
                local = spool / (digest[7:] + suffix)
                local.write_bytes(raw)
                response[field] = {
                    "asset_ref": digest,
                    "kind": asset.kind,
                    "digest": digest,
                    "size_bytes": len(raw),
                    "media_type": asset.media_type,
                }
                outputs.append(
                    {
                        "output_id": field,
                        "kind": asset.kind,
                        "digest": digest,
                        "length": len(raw),
                        "media_type": asset.media_type,
                        "local": str(local),
                    }
                )
            child_frames[index] = response["continuation_frame"]["digest"]
            answers[index], byte_grants[index] = json.dumps(response), outputs
            return {"ok": True, "child_request_id": f"{parent}-child-{index}"}
        if kind in ("child_forget", "child_cancel"):
            return {"ok": True}
        assert kind == "child_poll", kind
        index = int(value["call_index"])
        if index in child_progress:
            return {"ok": True, "state": "running", "progress": child_progress.pop(index)}
        if index == fail_after_progress_at:
            return {"ok": False, "code": "child.failed", "detail": "renderer failed after step 3"}
        return {
            "ok": True,
            "state": "succeeded",
            "result": answers[index],
            "byte_grants": byte_grants[index],
        }

    broker = _Broker(
        parent,
        {
            ("h3", child_name): _CallType(
                SHOT_INTERFACE,
                "h3",
                child_name,
                cast(type[msgspec.Struct], real.payload_type),
                h3.SegmentOutput,
            )
        },
        exchange,
    )
    wire = msgspec.to_builtins(
        h3.LongFormInput(
            shots=shots,
            mode=mode,
            steps=steps,
            subject_definitions="A small red rover.",
            overall_soundscape="Stream water and wind in trees; no vocals.",
            non_diegetic_music="N/A",
        )
    )
    grants = {}
    if opening is not None:
        local = root / "opening.png"
        raw = opening.read_bytes()
        local.write_bytes(raw)
        # A new request receives the host's SHA-256 identity of the encoded bytes,
        # not Outputs' process-local handle from the preceding parent attempt.
        digest = sha(raw)
        wire["opening_frame"] = digest
        grants["opening_frame"] = GrantedInput(
            input_id="opening_frame",
            local=local,
            media_type="image/png",
            digest=digest,
            length=len(raw),
            file_state=file_state(local),
        )
    assembly._scan_video = scan
    try:
        result, outcome, record = attempt(
            h3.app.get("long_form"),
            wire,
            Invocation(
                parent,
                root / "parent",
                time.monotonic() + 180,
                calls=broker,
                assets=grants,
                cancel=lambda: cancelled,
                progress=observe,
            ),
        )
    finally:
        assembly._scan_video = original_scan
    assert not any(thread.name.startswith("h3-scan") for thread in threads())
    if result is not None:
        assert len(result.outputs) == 2 and {row["kind"] for row in result.outputs} == {
            "video",
            "image",
        }
        assert not record.pending_trees
        final_frame = settle_frame(record, result.result.continuation_frame.ref)
        assert final_frame._attempt == parent
        child_image = spool / (child_frames[result.result.delivered - 1][7:] + ".png")
        with av.open(io.BytesIO(final_frame.read_bytes()), mode="r") as saved, av.open(
            child_image, mode="r"
        ) as child:
            assert np.array_equal(
                next(saved.decode(video=0)).to_ndarray(format="rgb24"),
                next(child.decode(video=0)).to_ndarray(format="rgb24"),
            )
    return result, outcome, calls


def canonical_media_digest(video: Any) -> str:
    """Hash decoded media, excluding muxer/container metadata.

    MP4 container bytes are not a stable identity across retries: codec/muxer metadata
    and packet layout may differ even when decoded frames and samples are identical.
    The retry proof therefore compares this canonical decoded representation instead.
    """
    digest = hashlib.sha256()
    with av.open(io.BytesIO(video.read_bytes()), mode="r") as container:
        video_stream = container.streams.video[0]
        digest.update(
            f"video:{video_stream.width}x{video_stream.height}:{video_stream.average_rate}".encode()
        )
        for video_frame in container.decode(video=0):
            digest.update(video_frame.to_ndarray(format="rgb24").tobytes())
    with av.open(io.BytesIO(video.read_bytes()), mode="r") as container:
        if container.streams.audio:
            audio_stream = container.streams.audio[0]
            digest.update(f"audio:{audio_stream.sample_rate}:{audio_stream.channels}".encode())
            for audio_frame in container.decode(audio=0):
                digest.update(audio_frame.to_ndarray().tobytes())
    return "sha256:" + digest.hexdigest()


def check_video(video: Any, frames: int) -> None:
    with av.open(io.BytesIO(video.read_bytes()), mode="r") as container:
        count = 0
        for _ in container.decode(video=0):
            count += 1
        assert count == frames


def check_progress(events: list[ProgressFrame], *, complete: bool) -> None:
    overall = [event.overall_fraction for event in events if event.overall_fraction is not None]
    assert overall and overall == sorted(overall), overall
    assert (overall[-1] == 1.0) is complete, overall[-1]
    # Streamed assembly work cannot claim completion before the final frame is saved.
    assembly_events = [event for event in events if event.stage.startswith("Assembling video")]
    assert assembly_events and all(event.overall_fraction != 1.0 for event in assembly_events)
    assert any(event.position is not None for event in assembly_events)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    shots = [h3.Shot(f"The rover reaches landmark {index}.", 1000 + index, 5) for index in range(7)]
    progress: list[ProgressFrame] = []
    result, outcome, calls = _drive(root / "single", shots[:1], progress=progress)
    assert outcome.terminal == "succeeded", outcome
    single = result.result
    assert single.complete and single.delivered == single.requested == 1 and len(calls) == 1
    check_video(single.video, 124)
    check_progress(progress, complete=True)
    assert progress[0].stage == "Shot 1 of 1"
    assert progress[0].stage_fraction is None and progress[0].overall_fraction == 0.0

    progress = []
    result, outcome, calls = _drive(
        root / "seven", shots, mode="turbo", overlap=True, progress=progress
    )
    assert outcome.terminal == "succeeded", outcome
    seven = result.result
    assert seven.complete and seven.delivered == seven.requested == 7 and len(calls) == 7
    assert seven.delivered_frames == 862 and seven.failed_index == -1
    check_video(seven.video, 862)
    check_progress(progress, complete=True)
    sent = [json.loads(call["payload"])["payload"] for call in calls]
    assert all(
        item["expected_provenance"] == msgspec.to_builtins(TURBO_PROVENANCE) for item in sent[1:]
    )
    assert all(item["first_frame"] for item in sent[1:])
    step = next(
        event for event in progress if event.position == 3 and event.stage.startswith("Shot 2 of 7")
    )
    assert step.total == 8 and step.stage_fraction == 0.375
    assert step.overall_fraction is not None
    assert math.isclose(step.overall_fraction, (992 + 0.4125 * 992) / 7807, abs_tol=2e-6)

    progress = []
    result, outcome, calls = _drive(
        root / "partial", shots, mode="turbo", fail_at=4, overlap=True, progress=progress
    )
    assert outcome.terminal == "succeeded", outcome
    partial = result.result
    assert not partial.complete and partial.delivered == 4 and partial.requested == 7
    assert partial.failed_index == 4 and partial.failure_code == "child.failed" and len(calls) == 5
    assert "PARTIAL DELIVERY" in partial.warnings[-1] and "opening_frame" in partial.warnings[-1]
    check_video(partial.video, 493)
    check_progress(progress, complete=False)
    assert not any(event.stage.startswith("Shot 6 of 7") for event in progress)

    result, outcome, calls = _drive(
        root / "continue", shots[4:], mode="turbo", opening=partial.continuation_frame
    )
    assert outcome.terminal == "succeeded", outcome
    assert result.result.complete and result.result.delivered == 3 and len(calls) == 3
    assert (
        json.loads(calls[0]["payload"])["payload"]["first_frame"]
        == sha(partial.continuation_frame.read_bytes())
    )
    check_video(result.result.video, 370)

    progress = []
    result, outcome, _ = _drive(
        root / "unequal",
        [shots[0], h3.Shot("The rover takes a longer path.", 42, 10)],
        mode="turbo",
        progress=progress,
    )
    assert outcome.terminal == "succeeded", outcome
    check_progress(progress, complete=True)
    boundary = next(event for event in progress if event.stage == "Shot 2 of 2")
    assert boundary.overall_fraction is not None
    assert math.isclose(boundary.overall_fraction, 992 / 3303, abs_tol=2e-6)

    progress = []
    result, outcome, _ = _drive(
        root / "fails-after-step", shots, mode="turbo", fail_after_progress_at=1, progress=progress
    )
    assert outcome.terminal == "succeeded" and result.result.delivered == 1, outcome
    check_progress(progress, complete=False)
    assert any(event.position == 3 and event.stage.startswith("Shot 2 of 7") for event in progress)

    automatic = [
        h3.Shot("A rover moves.", duration_s=5),
        h3.Shot("The rover stops.", seed=0, duration_s=5),
    ]
    result, outcome, calls = _drive(root / "automatic", automatic, mode="turbo")
    assert outcome.terminal == "succeeded", outcome
    chosen = json.loads(calls[0]["payload"])["payload"]["seed"]
    assert isinstance(chosen, int) and json.loads(calls[1]["payload"])["payload"]["seed"] == 0
    original_media_digest = canonical_media_digest(result.result.video)
    result, outcome, calls = _drive(
        root / "automatic-retry", automatic, mode="turbo", request_id="automatic"
    )
    assert outcome.terminal == "succeeded", outcome
    assert json.loads(calls[0]["payload"])["payload"]["seed"] == chosen
    assert canonical_media_digest(result.result.video) == original_media_digest
    result, outcome, calls = _drive(root / "automatic-new", automatic[:1], mode="turbo")
    assert (
        outcome.terminal == "succeeded"
        and json.loads(calls[0]["payload"])["payload"]["seed"] != chosen
    )

    result, outcome, _ = _drive(root / "cancel", shots, cancel_at=1, overlap=True)
    assert result is None and outcome.terminal == "canceled", outcome
    result, outcome, _ = _drive(root / "first-fails", shots, fail_at=0)
    assert result is None and outcome.terminal != "succeeded", outcome
    result, outcome, calls = _drive(root / "turbo-steps", shots, mode="turbo", steps=30)
    assert result is None and outcome.terminal != "succeeded" and not calls, outcome

    facts = {
        "public_output_kinds": ["video", "image"],
        "parent_owns_final_frame": True,
        "final_frame_matches_last_child_pixels": True,
        "no_intermediate_tree_output": True,
        "seven_shots_output_frames": seven.delivered_frames,
        "partial_output_frames": partial.delivered_frames,
        "partial_continues_from_opening_frame": True,
        "scan_overlaps_next_child": True,
        "child_steps_forwarded_with_shot_scope": True,
        "overall_progress_monotonic": True,
        "unequal_shots_weighted_by_frame_work": True,
        "partial_delivery_stays_below_completion": True,
        "automatic_seeds_stable_on_retry": True,
        "explicit_zero_seed_preserved": True,
        "cancellation_remains_cancelled": True,
        "actual_h3_inference": False,
    }
    (root / "evidence.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
