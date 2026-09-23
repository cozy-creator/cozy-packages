#!/usr/bin/env python3
"""Native cut routing and reference custody under Runtime's real broker and codecs.

The GPU renderer is synthetic. This establishes composition semantics, not video quality.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import av
import msgspec
import numpy as np
import torch
from cozy_runtime.author import (
    App,
    Context,
    ImageAsset,
    ImageFrame,
    Invocation,
    Outputs,
    Telemetry,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._assets import Asset, GrantedInput, file_state
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._codec import encode_frame
from cozy_runtime.author._services import ProgressFrame, settle_frame
from cozy_runtime.models.minimax_h3.official import FPS, MAX_CONDITIONER_VISION_TOKENS, frames_for
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import h3
from long_form_state import RenderProvenance
from story import ReferenceFrame, SelectedReference, select_references, shot_prompt, signature

MODEL, CODE, LORA = ("sha256:" + value * 64 for value in ("1", "2", "3"))
WIDTH, HEIGHT, RATE = 96, 64, 32000
ROUTES: list[str] = []


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@invocable
async def renderer(
    ctx: Context,
    *,
    payload: h3.CutInput,
    assets: h3.CutAssets,
    turbo: bool,
    out: Outputs,
    tel: Telemetry,
) -> h3.CutOutput:
    def pixels(seed: int, count: int) -> np.ndarray[Any, Any]:
        # Distinct seeds/views make deduplication observable, without an H3 model.
        result = np.empty((count, HEIGHT, WIDTH, 3), dtype=np.uint8)
        for index in range(count):
            result[index, :, :, :] = ((seed * 31 + index) % 251, index % 127, seed % 239)
            result[index, :, index % 90 : index % 90 + 6, 2] = 230
        return result

    def render(task: str, seed: int, duration: int, capture: Any) -> h3.H3VideoOutput:
        ROUTES.append(task)
        steps = 8 if turbo else payload.steps
        tel.step_callback(steps, stage="denoise", overall_range=(0.15, 0.85))(2)
        rgb = pixels(seed, frames_for(duration))
        frame = out.save_image(ImageFrame(WIDTH, HEIGHT, rgb[-1].tobytes()), format="png")
        audio = np.zeros((2, round(len(rgb) * RATE / FPS)), np.float32)
        video = out.save_video(rgb, fps=FPS, audio=audio, sample_rate=RATE)
        if capture is not None:
            capture(torch.from_numpy(rgb))
        return h3.H3VideoOutput(video, frame, [])

    def refs(_ctx: Any, task: str, request: Any, images: Any, *_: Any, **kwargs: Any) -> Any:
        assert task == ("ref2va_turbo" if turbo else "ref2va")
        assert images and "first" not in kwargs and "last" not in kwargs
        # Execute actual image presentation sizing and native-reference conversion.
        converted, sizing = h3.assets_to_h3_refs(
            images, pipe=cast(Any, SimpleNamespace(image_reference=lambda image: image))
        )
        assert len(converted) <= 9 and sizing.total <= MAX_CONDITIONER_VISION_TOKENS
        assert all(f"<Picture {index + 1}>" in request.prompt for index in range(len(images)))
        return render(task, request.seed, request.duration_s, kwargs["capture"])

    def text(_ctx: Any, task: str, *_: Any, **kwargs: Any) -> Any:
        assert task == ("fl2va_turbo" if turbo else "fl2va")
        assert kwargs["first"] is None and kwargs["last"] is None
        return render(task, kwargs["seed"], kwargs["duration_s"], kwargs["capture"])

    stubbed = cast(Any, h3)
    original = h3._references_to_video, h3._render_keyframes, stubbed.provenance
    h3._references_to_video, h3._render_keyframes = cast(Any, refs), cast(Any, text)
    stubbed.provenance = lambda model, adapter="": RenderProvenance(model, CODE, [], adapter)
    try:
        return h3._render_cut(
            ctx,
            payload,
            assets,
            cast(Any, SimpleNamespace(checkpoint_ref=MODEL)),
            out,
            tel,
            steps=8 if turbo else payload.steps,
            turbo_lora=cast(Any, SimpleNamespace(checkpoint_ref=LORA)) if turbo else None,
        )
    finally:
        h3._references_to_video, h3._render_keyframes = original[:2]
        stubbed.provenance = original[2]


CHILD = App()
CHILD.job(renderer, emits_media=True)


def drive(
    root: Path,
    *,
    count: int = 3,
    history: int | None = None,
    anchor: bool = False,
    fail: int = -1,
    cancel: int = -1,
    turbo: bool = True,
    request_id: str | None = None,
    prompt: str | None = None,
    refuse: bool = False,
) -> dict[str, Any]:
    root.mkdir()
    parent = root.name if request_id is None else request_id
    export = "cut_segment_turbo" if turbo else "cut_segment"
    surface = next(item for item in describe(h3.app) if item.name == export)
    calls: list[dict[str, Any]] = []
    events: list[Any] = []
    source: dict[str, Path] = {}
    answers: dict[int, dict[str, Any]] = {}
    pending: dict[int, dict[str, Any]] = {}
    cancelled = False

    def grant(path: str, digest: str, order: int = 0) -> GrantedInput:
        local = source[digest]
        return GrantedInput(
            input_id=path,
            local=local,
            digest=digest,
            media_type="image/png",
            length=local.stat().st_size,
            file_state=file_state(local),
            order=order,
        )

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal cancelled
        index = int(value["call_index"])
        if kind in ("child_cancel", "child_forget"):
            return {"ok": True}
        if kind == "child_poll":
            if index in pending:
                return {"ok": True, "state": "running", "progress": pending.pop(index)}
            return answers[index]
        assert kind == "child_call"
        calls.append(value)
        if index in (fail, cancel):
            cancelled = index == cancel
            return {"ok": False, "code": "child.failed", "detail": "synthetic interruption"}
        wire = json.loads(value["payload"])
        for name in ("base_model", "turbo_lora") if turbo else ("model",):
            assert wire.pop(name) is None
        assert "first_frame" not in wire["payload"] and "last_frame" not in wire["payload"]
        if turbo:
            wire["payload"]["steps"] = 30
        wire["turbo"] = turbo
        grants = {
            f"assets.{i}.asset": grant(f"assets.{i}.asset", item["asset"], i)
            for i, item in enumerate(wire["assets"])
        }
        emitted: list[Any] = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            result, outcome, record = pool.submit(
                attempt,
                CHILD.get("renderer"),
                wire,
                Invocation(
                    f"{parent}-child-{index}",
                    root / f"child-{index}",
                    time.monotonic() + 60,
                    assets=grants,
                    progress=emitted.append,
                ),
            ).result()
        assert outcome.terminal == "succeeded", outcome
        assert result is not None
        outputs: list[dict[str, Any]] = []

        def project(item: Any, path: str = "") -> Any:
            if isinstance(item, Asset):
                frame = record.frames[item.ref]
                raw = encode_frame(frame.codec, frame.facts, frame.raw.read_bytes())
                digest = sha(raw)
                local = root / (digest[7:] + (".png" if item.kind == "image" else ".mp4"))
                local.write_bytes(raw)
                source[digest] = local
                outputs.append(
                    {
                        "output_id": path,
                        "kind": item.kind,
                        "digest": digest,
                        "length": len(raw),
                        "media_type": item.media_type,
                        "local": str(local),
                    }
                )
                return {
                    "asset_ref": digest,
                    "kind": item.kind,
                    "digest": digest,
                    "size_bytes": len(raw),
                    "media_type": item.media_type,
                }
            if isinstance(item, msgspec.Struct):
                return {
                    name: project(getattr(item, name), f"{path}.{name}" if path else name)
                    for name in item.__struct_fields__
                }
            if isinstance(item, list):
                return [project(child, f"{path}.{i}") for i, child in enumerate(item)]
            return item

        answer = project(result.result)
        answers[index] = {
            "ok": True,
            "state": "succeeded",
            "result": json.dumps(answer),
            "byte_grants": outputs,
        }
        pending[index] = {
            "sequence": 1,
            "payload": asdict(next(event for event in emitted if isinstance(event, ProgressFrame))),
        }
        return {"ok": True, "child_request_id": f"{parent}-child-{index}"}

    wire: dict[str, Any] = {
        "shots": [
            {
                "prompt": f"Camera angle {index}: the red rover moves across the bridge.",
                "duration_s": 5,
                **({"seed": 0} if index == 0 else {}),
            }
            for index in range(count)
        ],
        "prompt": "A red rover with four wheels beside a woodland stream. No dialogue.",
        "mode": "turbo" if turbo else "standard",
    }
    if history is not None:
        wire["history_frames"] = history
    if prompt is not None:
        wire["prompt"] = prompt
    assets = {}
    if anchor:
        local = root / "rover.png"
        image = PILImage.new("RGB", (WIDTH, HEIGHT), (255, 0, 0))
        image.save(local)
        digest = sha(local.read_bytes())
        source[digest] = local
        wire["references"] = [
            {"image": digest, "description": "The rover's red body and wheels", "subject": "rover"}
        ]
        assets["references.0.image"] = grant("references.0.image", digest)
    broker = _Broker(
        parent,
        {
            ("", "h3", export): _CallType(
                "sha256:" + "5" * 64,
                "h3",
                export,
                cast(type[msgspec.Struct], surface.payload_type),
                h3.CutOutput,
            )
        },
        exchange,
    )
    result, outcome, record = attempt(
        h3.app.get("long_form_cuts"),
        wire,
        Invocation(
            parent,
            root / "parent",
            time.monotonic() + 180,
            calls=broker,
            assets=assets,
            cancel=lambda: cancelled,
            progress=events.append,
        ),
    )
    if refuse:
        assert result is None and outcome.terminal != "succeeded" and not calls, outcome
        return {}
    if cancel >= 0:
        assert result is None and outcome.terminal == "canceled", outcome
        return {}
    if fail == 0:
        assert result is None and outcome.terminal != "succeeded", outcome
        return {}
    assert result is not None and outcome.terminal == "succeeded", outcome
    delivered = count if fail < 0 else fail
    assert result.result.delivered == delivered and result.result.complete == (fail < 0)
    assert len(result.outputs) == 2 and {item["kind"] for item in result.outputs} == {
        "video",
        "image",
    }
    final = settle_frame(record, result.result.continuation_frame.ref)
    last = json.loads(answers[delivered - 1]["result"])["continuation_frame"]["digest"]
    assert final._attempt == parent
    with PILImage.open(io.BytesIO(final.read_bytes())) as a, PILImage.open(source[last]) as b:
        assert a.tobytes() == b.tobytes()
    with av.open(io.BytesIO(result.result.video.read_bytes()), mode="r") as container:
        assert sum(1 for _ in container.decode(video=0)) == delivered * frames_for(5)
    overall = [
        event.overall_fraction
        for event in events
        if isinstance(event, ProgressFrame) and event.overall_fraction is not None
    ]
    assert overall == sorted(overall) and (overall[-1] == 1) == (fail < 0)
    sent = [json.loads(call["payload"]) for call in calls]
    assert all("[Shot 1]" in row["payload"]["prompt"] for row in sent)
    assert sent[0]["payload"]["seed"] == 0
    assert len(sent[0]["assets"]) == int(anchor)
    assert all(len(row["assets"]) <= 9 for row in sent)
    if history != 0 and delivered > 1:
        prior = json.loads(answers[0]["result"])["reference_frames"][0]["image"]["digest"]
        assert any(item["asset"] == prior for item in sent[1]["assets"])
    return {"frames": result.result.delivered_frames, "calls": sent, "events": len(events)}


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True)
    results = {
        "history": drive(root / "history"),
        "anchors": drive(root / "anchors", anchor=True, history=3),
        "text_only": drive(root / "text-only", history=0, count=2),
        "standard": drive(root / "standard", turbo=False, count=2),
        "arbitrary_cut_count": drive(root / "arbitrary", count=10),
        "partial": drive(root / "partial", fail=2),
    }
    # Automatic history pools every completed shot and gives recent shots room for
    # additional interior candidates, while the camera-cut path accepts more than the
    # continuous eight-shot bound.
    assert len(results["history"]["calls"][2]["assets"]) >= 2
    assert len(results["arbitrary_cut_count"]["calls"]) == 10
    drive(root / "canceled", cancel=1)
    drive(root / "first-fails", fail=0)
    repeated = drive(root / "retry", request_id="history")
    assert (
        repeated["calls"][1]["payload"]["seed"] == results["history"]["calls"][1]["payload"]["seed"]
    )
    drive(root / "oversized-history", anchor=True, history=3, prompt="x" * 3500, refuse=True)
    # Repeating a picture preserves distinct role descriptions but consumes one slot.
    image = ImageAsset("same", digest="sha256:" + "7" * 64)
    fingerprint = signature(PILImage.new("RGB", (32, 32), "red"))
    selected = select_references(
        [
            SelectedReference(image, "red body", fingerprint, subject="rover"),
            SelectedReference(image, "four wheels", fingerprint, subject="rover"),
        ],
        [],
        history_frames=6,
    )
    assert len(selected) == 1
    prompt = shot_prompt("stream", "Overhead view of the rover turning", selected, index=0)
    assert prompt.count("<Subject ") == 1 and "<Picture 1>" in prompt
    assert "red body" in prompt and "four wheels" in prompt
    assert "first frame" not in prompt and "Overhead view" in prompt
    roles = select_references(
        [
            SelectedReference(image, "pilot", fingerprint, subject="pilot"),
            SelectedReference(image, "mechanic", fingerprint, subject="mechanic"),
        ],
        [],
        history_frames=6,
    )
    assert len(roles) == 1
    grouped = shot_prompt("hangar", "New angle", roles, index=5)
    assert grouped.count("<Subject ") == 2 and grouped.count("<Picture 1>") == 2
    assert "<Picture 2>" not in grouped and "[Shot 6]" not in grouped
    near = ReferenceFrame(ImageAsset("near", digest="sha256:" + "8" * 64), 50, fingerprint)
    assert select_references(selected, [[near]], history_frames=6) == selected
    nine = [
        msgspec.structs.replace(
            selected[0], image=ImageAsset(str(i), digest="sha256:" + f"{i:064x}"), subject=str(i)
        )
        for i in range(9)
    ]
    assert len(select_references(nine, [[near]], history_frames=6)) == 9
    assert "ref2va" in ROUTES and "ref2va_turbo" in ROUTES and "fl2va_turbo" in ROUTES
    (root / "evidence.json").write_text(
        json.dumps(
            {
                "cases": results,
                "native_routes": ROUTES,
                "actual_h3_inference": False,
                "two_public_assets": True,
                "cuts_preserve_all_frames": True,
                "reference_slots_and_tokens_bounded": True,
                "grouped_appearance_roles": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("cut routing, references, assembly, two-file custody, progress and interruption: PASS")


if __name__ == "__main__":
    main()
