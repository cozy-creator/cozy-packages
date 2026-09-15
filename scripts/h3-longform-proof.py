#!/usr/bin/env python3
"""Package recovery through Runtime's broker, real codecs and native retained prefixes.

The CPU renderer produces synthetic clips with the real served clock. Runtime's own
tests prove native host projection enforcement; this fixture returns explicit byte grants.
This proves composition and recovery, not H3 rendering or ordinary CLI qualification.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from threading import enumerate as threads
from typing import Any, Literal, cast

import av
import msgspec
import numpy as np
import tensorfs
from cozy_runtime.author import (
    App,
    Context,
    ImageFrame,
    Invocation,
    MediaDecoder,
    Outputs,
    Tree,
    VideoAsset,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput, file_state
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._codec import encode_frame
from cozy_runtime.author._media import SNIFF_BYTES, sniff

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import assembly
import h3
from long_form_state import PrefixManifest, RenderProvenance, SoftwareVersion, compatible
from official import FPS, frames_for

WIDTH, HEIGHT, RATE = 96, 64, 32000
SHOT_INTERFACE = "sha256:" + "5c" * 32
PROVENANCE = RenderProvenance(
    "sha256:" + "11" * 32,
    "sha256:" + "22" * 32,
    [SoftwareVersion("synthetic-renderer", "1")],
)

TURBO_PROVENANCE = msgspec.structs.replace(PROVENANCE, turbo_lora_manifest="sha256:" + "44" * 32)


@invocable
async def segment(ctx: Context, *, payload: h3.SegmentInput, out: Outputs) -> h3.SegmentOutput:
    """CPU stand-in; the real segment is a nonmemoized serving entrypoint."""
    compatible(PROVENANCE, payload.expected_provenance)
    ctx.raise_if_cancelled()
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


def persist_prefix(store: tensorfs.Store, prefix: Tree, destination: Path, owner: str) -> Tree:
    """Retain independently, release the producer, GC, then use a new native checkout."""
    files = {str(path.relative_to(prefix.path)): path for path in prefix.files()}
    entries = [
        {
            "path": name,
            "kind": "file",
            "blob": {"sha256": sha(path.read_bytes())[7:], "length": path.stat().st_size},
        }
        for name, path in sorted(files.items())
    ]
    raw = json.dumps({"entries": entries}, sort_keys=True, separators=(",", ":")).encode()
    producer, reader = sha((owner + ":producer").encode()), sha((owner + ":reader").encode())
    root = store.import_tree(producer, raw, sorted(files.items()))
    store.retain_tree_root(producer, reader)
    store.release_tree_root(producer)
    tensorfs.gc(store.root)
    assert store.tree_root(producer)["released"]  # type: ignore[index]
    digest = root["manifest_digest"]
    store.checkout(digest, destination, symlink=False)
    return Tree(digest, digest=digest, root=destination)


def _drive(
    root: Path,
    shots: list[h3.Shot],
    *,
    resume: Tree | None = None,
    fail_at: int = -1,
    cancel_at: int = -1,
    observed: RenderProvenance | None = None,
    mode: Literal["turbo", "standard"] = "standard",
    steps: int | None = None,
    overlap: bool = False,
    request_id: str | None = None,
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
    cancelled = False
    scan_started, child_started, scan_finished = Event(), Event(), Event()
    original_scan = assembly._scan_video

    def scan(
        decoder: MediaDecoder, asset: VideoAsset, check: Callable[[], None]
    ) -> assembly._Scan:
        if overlap and not scan_started.is_set():
            scan_started.set()
            assert child_started.wait(10), "scan did not overlap a later child"
            result = original_scan(decoder, asset, check)
            scan_finished.set()
            return result
        return original_scan(decoder, asset, check)

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal cancelled
        if kind == "tree_member":
            assert resume is not None and value["tree"] == resume.digest
            local = resume.path / value["path"]
            assert local.is_file() and not local.is_symlink()
            raw = local.read_bytes()
            media = sniff(raw[:SNIFF_BYTES]) or "application/octet-stream"
            assert len(raw) <= value["max_bytes"]
            assert not value["media_types"] or media in value["media_types"]
            return {
                "ok": True,
                "tree": resume.digest,
                "path": value["path"],
                "digest": sha(raw),
                "length": len(raw),
                "media_type": media,
                "local": str(local),
                "file_state": list(file_state(local)),
            }
        if kind == "child_call":
            index = int(value["call_index"])
            calls.append(value)
            if overlap and index == (0 if resume is not None else 1):
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
            # The self-call keeps the omitted model slot. Runtime resolves its frozen
            # default; this CPU stand-in has no model and consumes only the shot payload.
            incoming = document["payload"].get("first_frame")
            grants = {}
            if incoming:
                local = spool / f"{incoming[7:]}.png"
                if not local.exists():
                    assert resume is not None
                    matches = [p for p in resume.files() if sha(p.read_bytes()) == incoming]
                    assert len(matches) == 1
                    local = matches[0]
                grants["payload.first_frame"] = GrantedInput(
                    input_id="payload.first_frame",
                    local=local,
                    media_type="image/png",
                    digest=incoming,
                    length=local.stat().st_size,
                    file_state=file_state(local),
                )
            work = root / f"child-{index}"
            with ThreadPoolExecutor(max_workers=1) as pool:
                produced, outcome, record = pool.submit(
                    attempt,
                    child_app.get("segment"),
                    document,
                    Invocation(
                        f"{parent}-child-{index}", work, time.monotonic() + 60, assets=grants
                    ),
                ).result()
            assert outcome.terminal == "succeeded", outcome
            assert produced is not None
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
            answers[index], byte_grants[index] = json.dumps(response), outputs
            return {"ok": True, "child_request_id": f"{parent}-child-{index}"}
        if kind in ("child_forget", "child_cancel"):
            return {"ok": True}
        assert kind == "child_poll", kind
        index = int(value["call_index"])
        return {
            "ok": True,
            "state": "succeeded",
            "result": answers[index],
            "byte_grants": byte_grants[index],
        }

    broker = _Broker(
        parent,
        {
            ("", "h3", child_name): _CallType(
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
    trees = {}
    if resume is not None:
        wire["resume_from"] = resume.digest
        trees[resume.digest] = (resume.path, resume.digest)
    assembly._scan_video = scan
    try:
        result, outcome, _ = attempt(
            h3.app.get("long_form"),
            wire,
            Invocation(
                parent,
                root / "parent",
                time.monotonic() + 180,
                calls=broker,
                trees=trees,
                cancel=lambda: cancelled,
            ),
        )
    finally:
        assembly._scan_video = original_scan
    assert not any(thread.name.startswith("h3-scan") for thread in threads())
    return result, outcome, calls


def check_video(video: Any, frames: int) -> None:
    with av.open(io.BytesIO(video.read_bytes()), mode="r") as container:
        count = 0
        for _ in container.decode(video=0):
            count += 1
        assert count == frames


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    store = tensorfs.Store.init(root / "tensorfs")
    shots = [h3.Shot(f"The rover reaches landmark {index}.", 1000 + index, 5) for index in range(4)]
    result, outcome, calls = _drive(root / "single", shots[:1])
    assert outcome.terminal == "succeeded", outcome
    single = result.result
    assert single.complete and single.delivered == single.requested == 1
    assert single.reused == 0 and len(calls) == 1 and single.failed_index == -1
    assert single.delivered_frames == 124
    check_video(single.video, 124)
    single_prefix = persist_prefix(store, single.prefix, root / "retained-single", "single")
    result, outcome, calls = _drive(root / "second", shots[:2], resume=single_prefix, overlap=True)
    assert outcome.terminal == "succeeded", outcome
    second = result.result
    assert second.complete and second.delivered == 2 and second.reused == 1
    assert len(calls) == 1 and second.segments[:1] == single.segments
    sent = json.loads(calls[0]["payload"])["payload"]
    assert sent["first_frame"] == single.segments[0].continuation_frame_digest
    assert sent["expected_provenance"] == msgspec.to_builtins(PROVENANCE)
    check_video(second.video, 247)

    result, outcome, calls = _drive(root / "partial", shots, fail_at=2, overlap=True)
    assert outcome.terminal == "succeeded", outcome
    partial = result.result
    assert not partial.complete and partial.delivered == 2 and partial.requested == 4
    assert partial.reused == 0 and partial.failed_index == 2 and len(calls) == 3
    assert partial.failure_code == "child.failed" and "PARTIAL DELIVERY" in partial.warnings[-1]
    assert partial.delivered_frames == 247
    check_video(partial.video, 247)
    prefix = persist_prefix(store, partial.prefix, root / "retained-partial", "partial")

    edited = [*shots[:2], h3.Shot("The rover crosses a bridge.", 2002, 5), shots[3]]
    result, outcome, calls = _drive(root / "extended", edited, resume=prefix)
    assert outcome.terminal == "succeeded", outcome
    completed = result.result
    assert completed.complete and completed.reused == 2 and completed.delivered == 4
    assert len(calls) == 2 and completed.segments[:2] == partial.segments
    assert [record.start_frame for record in completed.segments] == [0, 124, 247, 370]
    assert completed.delivered_frames == 493
    payloads = [json.loads(call["payload"])["payload"] for call in calls]
    assert payloads[0]["first_frame"] == partial.segments[-1].continuation_frame_digest
    assert payloads[1]["first_frame"] == completed.segments[2].continuation_frame_digest
    assert all(sent["expected_provenance"] == msgspec.to_builtins(PROVENANCE) for sent in payloads)
    check_video(completed.video, 493)
    complete_prefix = persist_prefix(
        store, completed.prefix, root / "retained-complete", "complete"
    )

    result, outcome, calls = _drive(root / "reassemble", edited, resume=complete_prefix)
    assert outcome.terminal == "succeeded", outcome
    assert result.result.complete and result.result.reused == 4 and not calls
    assert result.result.segments == completed.segments
    assert result.result.video.read_bytes() == completed.video.read_bytes()
    check_video(result.result.video, 493)

    # Valid byte grants do not make a truncated fragment a completed shot. Update the
    # manifest's byte custody honestly while preserving the claimed full frame count.
    damaged = root / "truncated-source"
    shutil.copytree(complete_prefix.path, damaged, copy_function=shutil.copyfile)
    clip = damaged / "shots/000000/video.mp4"
    raw = clip.read_bytes()
    clip.write_bytes(raw[: len(raw) // 2])
    broken = msgspec.json.decode((damaged / "manifest.json").read_bytes(), type=PrefixManifest)
    first = msgspec.structs.replace(
        broken.shots[0], video_digest=sha(clip.read_bytes()), video_bytes=clip.stat().st_size
    )
    (damaged / "manifest.json").write_bytes(
        msgspec.json.encode(msgspec.structs.replace(broken, shots=[first, *broken.shots[1:]]))
    )
    truncated = persist_prefix(
        store, Tree("truncated", root=damaged), root / "retained-truncated", "truncated"
    )
    result, outcome, calls = _drive(root / "truncated", edited, resume=truncated)
    assert result is None and outcome.terminal != "succeeded" and not calls, outcome

    changed = [h3.Shot(shots[0].prompt, partial.segments[0].seed + 1, 5), *edited[1:]]
    result, outcome, calls = _drive(root / "changed-prefix", changed, resume=prefix)
    assert result is None and outcome.code == "prefix_intent" and not calls, outcome
    changed_code = msgspec.structs.replace(PROVENANCE, code_digest="sha256:" + "33" * 32)
    result, outcome, _ = _drive(root / "changed-code", edited, resume=prefix, observed=changed_code)
    assert result is None and outcome.code == "prefix_provenance", outcome

    result, outcome, calls = _drive(root / "turbo", shots[:2], mode="turbo")
    assert outcome.terminal == "succeeded", outcome
    turbo = result.result
    assert turbo.complete and len(calls) == 2
    assert all(item.provenance == TURBO_PROVENANCE for item in turbo.segments)
    turbo_prefix = persist_prefix(store, turbo.prefix, root / "retained-turbo", "turbo")
    turbo_manifest = msgspec.json.decode(
        (turbo_prefix.path / "manifest.json").read_bytes(), type=PrefixManifest
    )
    assert all(item.intent.steps == 8 for item in turbo_manifest.shots)
    result, outcome, calls = _drive(root / "turbo-resume", shots, mode="turbo", resume=turbo_prefix)
    assert outcome.terminal == "succeeded", outcome
    assert result.result.reused == 2 and len(calls) == 2
    assert result.result.segments[:2] == turbo.segments
    changed_adapter = msgspec.structs.replace(
        TURBO_PROVENANCE, turbo_lora_manifest="sha256:" + "55" * 32
    )
    result, outcome, _ = _drive(
        root / "changed-adapter", shots, mode="turbo", resume=turbo_prefix, observed=changed_adapter
    )
    assert result is None and outcome.code == "prefix_provenance", outcome
    result, outcome, calls = _drive(
        root / "mode-change", shots, mode="standard", resume=turbo_prefix
    )
    assert result is None and outcome.code == "prefix_intent" and not calls, outcome
    result, outcome, calls = _drive(
        root / "turbo-standard-prefix", shots, mode="turbo", resume=prefix
    )
    assert result is None and outcome.code == "prefix_intent" and not calls, outcome
    result, outcome, calls = _drive(root / "turbo-steps", shots, mode="turbo", steps=30)
    assert result is None and outcome.terminal != "succeeded" and not calls, outcome

    # Optional seeds still become concrete child inputs and retained rendering facts.
    automatic = msgspec.json.decode(
        b'[{"prompt":"A rover moves.","duration_s":5},'
        b'{"prompt":"The rover stops.","seed":0,"duration_s":5}]',
        type=list[h3.Shot],
    )
    result, outcome, calls = _drive(root / "automatic", automatic, mode="turbo", fail_at=1)
    assert outcome.terminal == "succeeded", outcome
    auto_partial = result.result
    chosen = auto_partial.segments[0].seed
    assert isinstance(chosen, int) and len(calls) == 2
    assert json.loads(calls[0]["payload"])["payload"]["seed"] == chosen
    assert json.loads(calls[1]["payload"])["payload"]["seed"] == 0
    result, outcome, calls = _drive(
        root / "automatic-retry", automatic, mode="turbo", fail_at=1, request_id="automatic"
    )
    assert outcome.terminal == "succeeded" and result.result.segments[0].seed == chosen
    assert result.result.video.read_bytes() == auto_partial.video.read_bytes()
    result, outcome, _ = _drive(root / "automatic-new-run", automatic[:1], mode="turbo")
    assert outcome.terminal == "succeeded" and result.result.segments[0].seed != chosen
    auto_prefix = persist_prefix(store, auto_partial.prefix, root / "retained-auto", "auto")
    result, outcome, calls = _drive(
        root / "automatic-resume", automatic, mode="turbo", resume=auto_prefix
    )
    assert outcome.terminal == "succeeded", outcome
    assert result.result.complete and result.result.reused == 1 and len(calls) == 1
    assert [receipt.seed for receipt in result.result.segments] == [chosen, 0]
    assert result.result.segments[:1] == auto_partial.segments
    conflicting = [h3.Shot(automatic[0].prompt, chosen + 1, 5), automatic[1]]
    result, outcome, calls = _drive(
        root / "automatic-conflict", conflicting, mode="turbo", resume=auto_prefix
    )
    assert result is None and outcome.code == "prefix_intent" and not calls, outcome

    result, outcome, _ = _drive(root / "cancel", shots, cancel_at=1, overlap=True)
    assert result is None and outcome.terminal == "canceled", outcome
    result, outcome, _ = _drive(root / "first-fails", shots, fail_at=0)
    assert result is None and outcome.terminal != "succeeded", outcome
    manifest = msgspec.json.decode(
        (prefix.path / "manifest.json").read_bytes(), type=PrefixManifest
    )
    assert [record.child_request_id for record in manifest.shots] == [
        receipt.child_request_id for receipt in partial.segments
    ]
    facts = {
        "single_frames": single.delivered_frames,
        "single_prefix": single_prefix.digest,
        "single_extension_reused": second.reused,
        "single_extension_rendered": 1,
        "partial_frames": partial.delivered_frames,
        "completed_frames": completed.delivered_frames,
        "reused_shots": completed.reused,
        "native_prefix": complete_prefix.digest,
        "scan_overlaps_next_child": True,
        "retained_scan_overlaps_next_child": True,
        "reassembly_preserves_video_bytes": True,
        "changed_intent_refused": True,
        "changed_code_refused": True,
        "truncated_prefix_refused": True,
        "scan_worker_joined_on_cancellation": True,
        "cancellation_remains_cancelled": True,
        "turbo_prefix": turbo_prefix.digest,
        "turbo_evaluations": 8,
        "turbo_adapter_change_refused": True,
        "mode_change_refused": True,
        "turbo_step_override_refused": True,
        "automatic_seeds_recorded": True,
        "automatic_seeds_stable_on_retry": True,
        "automatic_seeds_differ_for_new_run": True,
        "omitted_seed_reuses_retained_shot": True,
        "explicit_zero_seed_preserved": True,
        "actual_h3_inference": False,
    }
    (root / "evidence.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
