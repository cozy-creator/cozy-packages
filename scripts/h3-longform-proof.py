#!/usr/bin/env python3
"""Real broker, real payload conversion, real asset grants: the long-form chain's seam.

`long_form` composes shots by EMITTING ordinary `segment` requests; the only thing stood
in here is the renderer itself, which needs a GPU and is already proven by fl2va's own
conformance. Everything the chain depends on is the shipped code path: the child broker,
the canonical intent encoding, the grant check that forwards only THIS attempt's verified
assets, and the digest-bound hand-off between shots.

Proves four things:
  1. N shots emit N ordinary child calls, in order.
  2. Shot N+1's `first_frame` is shot N's `continuation_frame`, carried BY DIGEST.
  3. A failing shot is REPORTED, not raised: the prefix before it survives and is
     assemblable, which is what content-addressed hand-off is for.
  4. The cumulative frame clock is exact integer arithmetic with one replayed frame
     removed per seam — never a per-seam float residue.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import msgspec
import numpy
from cozy_runtime.author import (
    App,
    Context,
    ImageFrame,
    Invocation,
    Outputs,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._calls import _Broker, _CallType

import h3

WIDTH, HEIGHT = 64, 36
SHOT_INTERFACE = "sha256:" + "5c" * 32


@invocable
async def segment(ctx: Context, *, payload: h3.SegmentInput, out: Outputs) -> h3.SegmentOutput:
    """A renderer stand-in with `segment`'s exact request and result types.

    It returns a continuation frame whose bytes are a function of the shot's own identity,
    so the digest that binds the next shot is a real content address, not a fixture.
    """
    seed = payload.seed % 251
    pixels = bytes(((seed + index) % 256) for index in range(WIDTH * HEIGHT * 3))
    frame = out.save_image(ImageFrame(WIDTH, HEIGHT, pixels), format="png")
    clip = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(1, HEIGHT, WIDTH, 3)
    video = out.save_video(numpy.repeat(clip, 4, axis=0), fps=24.0)
    return h3.SegmentOutput(video, frame, [])


child_app = App()
child_app.job(segment, emits_media=True)


def _drive(
    shots: list[h3.Shot], *, fail_at: int, spool: Path
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Run `long_form` against a real broker whose children really execute."""
    real = next(s for s in describe(h3.app) if s.name == "segment")
    calls: list[dict[str, Any]] = []
    answers: dict[int, str] = {}
    byte_grants: dict[int, list[dict[str, Any]]] = {}

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        if kind == "child_call":
            index = int(value["call_index"])
            calls.append(value)
            if index == fail_at:
                return {
                    "ok": False,
                    "code": "child.failed",
                    "detail": "stand-in renderer refused this shot",
                    "child_request_id": f"child-{index}",
                }
            document = json.loads(value["payload"])
            # The real export carries a model binding; the stand-in renders nothing, so it
            # is dropped here after proving the child really asked for the H3 ladder.
            assert "model" in document, document
            document = {key: item for key, item in document.items() if key != "model"}
            grants = {}
            digest = document["payload"].get("first_frame")
            if digest:
                path = spool / f"{digest.split(':')[1]}.png"
                grants["payload.first_frame"] = GrantedInput(
                    input_id="payload.first_frame",
                    local=path,
                    media_type="image/png",
                    digest=digest,
                    length=path.stat().st_size,
                )
            work = spool / f"child-{index}"
            with ThreadPoolExecutor(max_workers=1) as pool:
                produced, outcome, _ = pool.submit(
                    attempt,
                    child_app.get("segment"),
                    document,
                    Invocation(
                        f"child-{index}", work, time.monotonic() + 60, assets=grants
                    ),
                ).result()
            if outcome.terminal != "succeeded":
                raise AssertionError(f"stand-in child {index} failed: {outcome}")
            assert produced is not None
            # Stand in for the worker's post phase: a package emits unencoded refs and the
            # HOST encodes, digests and grants the bytes back. Digesting the raw payload is
            # the same content address one layer earlier, which is all the chain relies on.
            document_result: dict[str, Any] = {"warnings": []}
            grants_out: list[dict[str, Any]] = []
            for field in ("video", "continuation_frame"):
                asset = getattr(produced.result, field)
                kind_name, number = asset.ref.rsplit("/", 2)[-2:]
                raw = work / f"{kind_name}-{number}.raw"
                payload_bytes = raw.read_bytes()
                digest = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
                local = spool / f"{digest.split(':')[1]}.{'png' if kind_name == 'image' else 'mp4'}"
                local.write_bytes(payload_bytes)
                document_result[field] = {
                    "asset_ref": digest,
                    "kind": asset.kind,
                    "digest": digest,
                    "size_bytes": len(payload_bytes),
                    "media_type": asset.media_type,
                }
                grants_out.append(
                    {
                        "output_id": field,
                        "kind": asset.kind,
                        "digest": digest,
                        "length": len(payload_bytes),
                        "media_type": asset.media_type,
                        "local": str(local),
                    }
                )
            answers[index] = json.dumps(document_result)
            byte_grants[index] = grants_out
            return {"ok": True, "child_request_id": f"child-{index}"}
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
        "parent",
        {
            # The binding names the REAL export the parent calls; only the implementation
            # behind it is stood in, and it is driven through its own real payload type.
            ("", "h3", "segment"): _CallType(
                SHOT_INTERFACE,
                "h3",
                "segment",
                cast(type[msgspec.Struct], real.payload_type),
                h3.SegmentOutput,
            )
        },
        exchange,
    )
    wire = msgspec.to_builtins(
        h3.LongFormInput(
            shots=shots,
            subject_definitions="AZUL: a woman in a red rain jacket.",
            overall_soundscape="A crowded night market.",
            non_diegetic_music="None.",
        )
    )
    result, outcome, _ = attempt(
        h3.app.get("long_form"),
        wire,
        Invocation("parent", spool / "parent", time.monotonic() + 120, calls=broker),
    )
    return result, outcome, calls


def main() -> None:
    shots = [
        h3.Shot(prompt=f"Shot {index + 1} of the night market.", seed=1000 + index, duration_s=5)
        for index in range(4)
    ]

    with TemporaryDirectory() as raw:
        spool = Path(raw)
        (spool / "parent").mkdir()

        # --- 1. the whole chain -------------------------------------------------------
        result, outcome, calls = _drive(shots, fail_at=-1, spool=spool)
        assert outcome.terminal == "succeeded", outcome
        out = result.result
        assert out.failed_index == -1 and len(out.segments) == 4 and out.requested == 4, out
        assert len(calls) == 4, len(calls)

        # 2. every hand-off is the previous shot's continuation frame, by digest
        payloads = [json.loads(call["payload"])["payload"] for call in calls]
        assert payloads[0]["first_frame"] is None, payloads[0]
        for index in range(1, 4):
            expected = out.segments[index - 1].continuation_frame_digest
            assert payloads[index]["first_frame"] == expected, (index, payloads[index])
            assert out.segments[index].first_frame_digest == expected
        # the wire carried the digest and NOT a path
        assert "/" not in json.dumps(payloads), "an intent leaked a filesystem path"

        # the anchors ride verbatim in every shot
        for index, sent in enumerate(payloads):
            assert "AZUL: a woman in a red rain jacket." in sent["prompt"], index
            assert "A crowded night market." in sent["prompt"], index
            assert f"[Shot {index + 1}]" in sent["prompt"], index

        # 4. the clock: 124 frames a shot, one replayed frame off every seam
        assert [s.start_frame for s in out.segments] == [0, 124, 247, 370], out.segments
        assert out.delivered_frames == 4 * 124 - 3 == 493, out.delivered_frames
        assert Fraction(out.delivered_frames, out.fps) == Fraction(493, 24)

        # --- 3. a shot fails: the prefix survives ------------------------------------
        result, outcome, calls = _drive(shots, fail_at=2, spool=spool)
        assert outcome.terminal == "succeeded", outcome
        partial = result.result
        assert len(partial.segments) == 2 and partial.requested == 4, partial
        assert partial.failed_index == 2 and partial.failure_code == "child.failed", partial
        assert len(partial.segments) == 2
        # the delivered prefix is a real, assemblable two-shot video with its own exact clock
        assert partial.delivered_frames == 2 * 124 - 1 == 247, partial.delivered_frames
        assert Fraction(partial.delivered_frames, partial.fps) == Fraction(247, 24)
        # and it is the SAME prefix the complete run produced — the hand-off is deterministic
        assert [s.continuation_frame_digest for s in partial.segments] == [
            s.continuation_frame_digest for s in out.segments[:2]
        ]

        # --- the opening shot failing leaves nothing to deliver, and says so ----------
        result, outcome, _ = _drive(shots, fail_at=0, spool=spool)
        assert result is None and outcome.terminal != "succeeded", outcome

    print("PASS  4 shots -> 4 ordinary child calls, digest-bound, clock 493/24 s")
    print("PASS  shot 3 fails -> 2 delivered, prefix identical, clock 247/24 s")
    print("PASS  shot 1 fails -> refused, nothing to deliver")


if __name__ == "__main__":
    main()
