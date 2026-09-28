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
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Annotated, Any, Literal, cast

import av
import msgspec
import numpy as np
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
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
from cozy_runtime.author._services import ProgressFrame
from cozy_runtime.internal import interface_wheel, package_interface, static_interface
from cozy_runtime.internal.worker import machine_byte_results
from cozy_runtime.models.minimax_h3.continuation import AVContext
from cozy_runtime.models.minimax_h3.official import FPS, MAX_CONDITIONER_VISION_TOKENS, frames_for


class ReferenceOutput(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(max_bytes=64 << 20, media_types=("image/png",))]
    width: int
    height: int
    seed: int


@invocable
async def reference_renderer(
    ctx: Context,
    *,
    prompt: str,
    aspect_ratio: str,
    megapixels: int,
    steps: int,
    seed: int,
    background: Literal["normal", "white"],
    reference_images: list[ImageAsset],
    out: Outputs,
    tel: Telemetry,
) -> ReferenceOutput:
    ctx.raise_if_cancelled()
    assert aspect_ratio == "1:1" and megapixels == 1 and steps == 40 and not reference_images
    tel.step_callback(steps, stage="denoise", overall_range=(0.1, 0.9))(2)
    rgb = bytes((seed % 251, 128, 255)) * WIDTH * HEIGHT
    image = out.save_image(ImageFrame(WIDTH, HEIGHT, rgb), format="png")
    return ReferenceOutput(image, WIDTH, HEIGHT, seed)


# Compile the real package's caller contract. Only rendering is synthetic: H3
# calls the actual generated flat-argument proxy, including its model envelope.
ROOT = Path(__file__).resolve().parents[1]
reference_interface = package_interface.canonical_bytes(
    static_interface.build(ROOT / "qwen-image-2")
)
reference_code = interface_wheel.generate(reference_interface)["qwen_image_2/__init__.py"]
reference_module = ModuleType("qwen_image_2")
sys.modules["qwen_image_2"] = reference_module
exec(compile(reference_code, "generated-reference-image-caller", "exec"), reference_module.__dict__)
REFERENCE_BINDING = cast(Any, reference_module).__cozy_bindings__["generate_image"]
sys.path.insert(0, str(ROOT / "minimax-h3"))
import h3  # noqa: E402
from long_form_state import RenderProvenance  # noqa: E402
from story import (  # noqa: E402
    StoryReference,
    reference_seed,
    StorySegment,
    segment_prompt,
    validate_references,
)

MODEL, CODE, LORA = ("sha256:" + value * 64 for value in ("1", "2", "3"))
WIDTH, HEIGHT, RATE = 96, 64, 32000
ROUTES: list[str] = []


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def present(assets: Any, prompt: str) -> Any:
    """Run the package's actual image and audio reference conversion and label checks."""
    kinds = [assets.info(index).kind for index in range(len(assets))]
    converted = h3.assets_to_h3_refs(
        assets,
        pipe=cast(
            Any,
            SimpleNamespace(image_reference=lambda item: item, audio_reference=lambda item: item),
        ),
    )
    for kind, label in (("image", "Picture"), ("audio", "Audio")):
        assert all(f"<{label} {n + 1}>" in prompt for n in range(kinds.count(kind))), kinds
    return converted


@invocable
async def renderer(
    ctx: Context,
    *,
    payload: h3.CutInput,
    assets: h3.CutAssets,
    turbo: bool,
    out: Outputs,
    tel: Telemetry,
) -> h3.SegmentOutput:
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
        audio = np.zeros((2, round(len(rgb) * RATE / FPS)), np.float32)
        video = out.save_video(rgb, fps=FPS, audio=audio, sample_rate=RATE)
        if capture is not None:
            capture(torch.from_numpy(rgb))
        return h3.H3VideoOutput(video, [])

    def refs(_ctx: Any, task: str, request: Any, images: Any, *_: Any, **kwargs: Any) -> Any:
        assert task == ("ref2va_turbo" if turbo else "ref2va")
        assert images and "first" not in kwargs and "last" not in kwargs
        # Execute actual image presentation sizing and native-reference conversion.
        converted, sizing = present(images, request.prompt)
        assert len(converted) <= 12 and sizing.total <= MAX_CONDITIONER_VISION_TOKENS
        return render(task, request.seed, request.duration_s, kwargs.get("capture"))

    def text(_ctx: Any, task: str, *_: Any, **kwargs: Any) -> Any:
        assert task == ("fl2va_turbo" if turbo else "fl2va")
        assert kwargs["first"] is None and kwargs["last"] is None
        return render(task, kwargs["seed"], kwargs["duration_s"], kwargs.get("capture"))

    stubbed = cast(Any, h3)
    original = h3._references_to_video, h3._render_keyframes, stubbed.provenance
    h3._references_to_video, h3._render_keyframes = cast(Any, refs), cast(Any, text)
    stubbed.provenance = lambda model, adapter="": RenderProvenance(model, adapter)
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


@invocable
async def motion_renderer(
    ctx: Context,
    *,
    payload: h3.MotionInput,
    assets: h3.CutAssets,
    turbo: bool,
    out: Outputs,
    tel: Telemetry,
) -> h3.MotionOutput:
    def refs(_ctx: Any, task: str, request: Any, images: Any, *_: Any, **kwargs: Any) -> Any:
        assert task == ("ref2va_turbo" if turbo else "ref2va")
        ROUTES.append(task)
        assert images and kwargs["expected_context_provenance"] == CODE
        present(images, request.prompt)
        context, delivery = kwargs["context"], kwargs["delivery"]
        if context is not None:
            assert context.frame_count == payload.context_frames
            assert context.provenance == CODE
        assert delivery.prefix_frames == (0 if context is None else payload.context_frames)
        tel.step_callback(payload.steps, stage="denoise", overall_range=(0.15, 0.85))(2)
        count = delivery.delivered_frames
        rgb = np.full((count, HEIGHT, WIDTH, 3), request.seed % 251, dtype=np.uint8)
        for index in range(count):
            rgb[index, :, index % 90 : index % 90 + 6, 2] = 230
        audio = np.zeros((2, count * RATE // FPS), np.float32)
        video = out.save_video(rgb, fps=FPS, audio=audio, sample_rate=RATE)
        if payload.next_context_frames:
            tail = AVContext(
                {
                    frames: (
                        torch.zeros(
                            (1, 24, 5 * ((frames - 5) // 17) + 2, HEIGHT // 16, WIDTH // 16)
                        ),
                        torch.zeros((2 * ((frames * 5 + 2) // 3), 32)),
                    )
                    for frames in payload.next_context_frames
                },
                HEIGHT, WIDTH, count, CODE, max(payload.next_context_frames),
            )
            kwargs["completed"](tail, rgb)
        else:
            assert kwargs["completed"] is None
        return h3.H3VideoOutput(video, [])

    def export(state: Any, *, frames: Any, windows: Any, provenance: str) -> Any:
        # The delivered frames as landed, and only the windows the successor selects.
        assert len(frames) == payload.duration_s * FPS and provenance == CODE
        assert tuple(windows) == payload.next_context_frames == tuple(state.windows)
        return state

    stubbed = cast(Any, h3)
    original = h3._references_to_video, stubbed.provenance, stubbed.context_provenance
    h3._references_to_video = cast(Any, refs)
    stubbed.provenance = lambda model, adapter="": RenderProvenance(model, adapter)
    stubbed.context_provenance = lambda _: CODE
    model = SimpleNamespace(
        checkpoint_ref=MODEL,
        export_completed_av_tail=export,
    )
    try:
        return h3._render_motion(
            ctx,
            payload,
            assets,
            cast(Any, model),
            out,
            tel,
            turbo_lora=cast(Any, SimpleNamespace(checkpoint_ref=LORA)) if turbo else None,
        )
    finally:
        h3._references_to_video, stubbed.provenance, stubbed.context_provenance = original


CHILD = App()
CHILD.job(renderer, emits_media=True)
CHILD.job(motion_renderer, emits_media=True)
CHILD.job(reference_renderer, emits_media=True)


def drive(
    root: Path,
    *,
    count: int = 3,
    fail: int = -1,
    cancel: int = -1,
    turbo: bool = True,
    request_id: str | None = None,
    prompt: str | None = None,
    refuse: bool = False,
    bad: str = "",
    reference_failure: bool = False,
    reference_cancel: bool = False,
    continuous: bool = False,
    audio: bool = False,
    refuse_code: str = "",
    context_frames: int = 39,
    duration_s: int = 5,
) -> dict[str, Any]:
    root.mkdir()
    parent = root.name if request_id is None else request_id
    export = ("motion_segment" if continuous else "cut_segment") + ("_turbo" if turbo else "")
    surface = next(item for item in describe(h3.app) if item.name == export)
    calls: list[dict[str, Any]] = []
    prefetches: list[dict[str, Any]] = []
    events: list[Any] = []
    source: dict[str, Path] = {}
    media: dict[str, str] = {}
    answers: dict[int, dict[str, Any]] = {}
    pending: dict[int, dict[str, Any]] = {}
    cancelled = False
    completed_references: set[int] = set()
    canceled_children: set[int] = set()
    settled: set[int] = set()
    releases: list[int] = []

    def grant(path: str, digest: str, order: int = 0) -> GrantedInput:
        local = source[digest]
        return GrantedInput(
            input_id=path,
            local=local,
            digest=digest,
            media_type=(
                "application/octet-stream"
                if path == "payload.context"
                else media.get(digest, "image/png")
            ),
            length=local.stat().st_size,
            file_state=file_state(local),
            order=order,
        )

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal cancelled
        if kind == "model_prefetch":
            prefetches.append(value)
            return {"ok": True}
        if kind == "gpu_release":
            # Only after every child's result is in: assembly needs no GPU.
            assert not pending and set(answers) <= settled, (sorted(answers), sorted(settled))
            releases.append(len(calls))
            return {"ok": True}
        index = int(value["call_index"])
        if kind in ("child_cancel", "child_forget"):
            if kind == "child_cancel":
                canceled_children.add(index)
            return {"ok": True}
        if kind == "child_poll":
            if index < 2:
                # Both requests must be submitted before awaiting either result.
                # Settle the second first to prove identity is definition ordered.
                assert len(calls) == 2
                if index == 0 and 1 not in completed_references:
                    return {"ok": True, "state": "running"}
                if index == 1 and reference_cancel:
                    cancelled = True
                    return {"ok": True, "state": "running"}
                if index == 1 and reference_failure:
                    return {"ok": False, "code": "child.failed", "detail": "reference failed"}
                if index not in pending:
                    completed_references.add(index)
            if index in pending:
                return {"ok": True, "state": "running", "progress": pending.pop(index)}
            settled.add(index)
            return answers[index]
        assert kind == "child_call"
        calls.append(value)
        shot_index = index - 2
        if index >= 2 and shot_index in (fail, cancel):
            cancelled = shot_index == cancel
            return {"ok": False, "code": "child.failed", "detail": "synthetic interruption"}
        wire = json.loads(value["payload"])
        is_reference = value["export"] == "generate_image"
        if is_reference:
            assert wire["models"] == {"model": None}
            wire = wire["payload"]
            assert wire["background"] == ("white" if index == 0 else "normal")
            assert (
                "white studio" in wire["prompt"]
                if index == 0
                else "without people" in wire["prompt"]
            )
            grants = {}
        else:
            assert completed_references == {0, 1}, "H3 started before every reference settled"
            for name in ("base_model", "turbo_lora") if turbo else ("model",):
                assert wire.pop(name) is None
            assert "first_frame" not in wire["payload"] and "last_frame" not in wire["payload"]
            if turbo and not continuous:
                wire["payload"]["steps"] = 30
            wire["turbo"] = turbo
            grants = {
                f"assets.{i}.asset": grant(f"assets.{i}.asset", item["asset"], i)
                for i, item in enumerate(wire["assets"])
            }
            if continuous and wire["payload"].get("context") is not None:
                grants["payload.context"] = grant("payload.context", wire["payload"]["context"])
        emitted: list[Any] = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            result, outcome, record = pool.submit(
                attempt,
                CHILD.get(
                    "reference_renderer"
                    if is_reference
                    else ("motion_renderer" if continuous else "renderer")
                ),
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
                if item.kind == "file":
                    raw = item.read_bytes()
                else:
                    frame = record.frames[item.ref]
                    raw = encode_frame(frame.codec, frame.facts, frame.raw.read_bytes())
                digest = sha(raw)
                local = root / (digest[7:] + (".png" if item.kind == "image" else ".mp4"))
                if not local.exists():  # content-addressed; a rewrite would change a granted file
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
        # Only a shot with a successor exports context: the last one would pay for nothing.
        successor = continuous and not is_reference and shot_index + 1 < count
        if continuous and not is_reference:
            assert wire["payload"].get("next_context_frames", []) == (
                [context_frames] if successor else []
            )
        assert {item["output_id"] for item in outputs} == (
            {"image"}
            if is_reference
            else {"video"} | ({"context"} if continuous else set())
        )
        if continuous and not is_reference:
            # Every shot fills the fixed slot; only a shot with a successor pays for windows.
            context = next(item for item in outputs if item["output_id"] == "context")
            assert (context["length"] > 0) == successor, context
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
        "segments": [
            {
                "summary": "[reference generation] <Rover> travels across <Bridge>.",
                "detailed_description": "[Shot 1] <Rover> moves across <Bridge>.",
                "overall_soundscape": "A flowing woodland stream.",
                "non_diegetic_music": "N/A",
                "duration_s": duration_s,
                **({"seed": 0} if index == 0 else {}),
            }
            for index in range(count)
        ],
        "style": "Photorealistic nature photography.",
        "references": [
            {
                "name": "Rover",
                "kind": "character",
                "description": "A red robot with four wheels.",
                "seed": 21,
            },
            {
                "name": "Bridge",
                "kind": "scene",
                "description": "A stone bridge over a woodland stream.",
                "seed": 22,
            },
        ],
        "mode": "turbo" if turbo else "standard",
    }
    if continuous:
        wire["context_frames"] = context_frames
    if prompt is not None:
        wire["style"] = prompt
    if bad == "empty-references":
        wire["references"] = []
    elif bad == "duplicate":
        wire["references"][1]["name"] = "ROVER"
    elif bad == "empty":
        wire["references"][0]["description"] = " "
    elif bad == "seed":
        wire["references"][1]["seed"] = -1
    elif bad == "seed-too-large":
        wire["references"][1]["seed"] = 9007199254740992
    elif bad == "empty-description":
        wire["segments"][-1]["detailed_description"] = " "
    elif bad == "too-many-references":
        wire["references"] = [
            {"name": f"Person{i}", "kind": "character", "description": "A person"} for i in range(10)
        ]
    elif bad == "legacy":
        wire["history_frames"] = 0
    parent_assets: dict[str, GrantedInput] = {}
    voice = ""
    if audio:
        # Three seconds of mono speech-band tone: a supplied voice needs no Qwen image.
        path = root / "voice.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            tone = np.sin(np.arange(3 * RATE) * (2 * np.pi * 220 / RATE)) * 8000
            out.writeframes(tone.astype("<i2").tobytes())
        voice = sha(path.read_bytes())
        source[voice], media[voice] = path, "audio/wav"
        reference = {
            "name": "RoverVoice",
            "kind": "audio",
            "description": "It is the voice-timbre reference for <Rover> (S1).",
            "retention-analysis": "reference - its timbre guides <Rover>'s speech.",
            "audio": voice,
        }
        wire["references"] = (
            [reference] if bad == "audio-only" else [*wire["references"], reference]
        )
        slot = len(wire["references"]) - 1
        parent_assets[f"references.{slot}.audio"] = grant(f"references.{slot}.audio", voice, slot)
    broker = _Broker(
        parent,
        {
            ("h3", export): _CallType(
                "sha256:" + "5" * 64,
                "h3",
                export,
                cast(type[msgspec.Struct], surface.payload_type),
                h3.MotionOutput if continuous else h3.SegmentOutput,
            ),
            ("qwen_image_2", "generate_image"): REFERENCE_BINDING,
        },
        exchange,
    )
    result, outcome, record = attempt(
        h3.app.get("long_form" if continuous else "long_form_cuts"),
        wire,
        Invocation(
            parent,
            root / "parent",
            time.monotonic() + 180,
            calls=broker,
            cancel=lambda: cancelled,
            progress=events.append,
            assets=parent_assets,
        ),
    )
    if reference_cancel:
        assert result is None and outcome.terminal == "canceled", outcome
        assert len(calls) == 2
        return {}
    if reference_failure:
        assert result is None and outcome.terminal != "succeeded", outcome
        assert len(calls) == 2 and 0 in canceled_children
        return {}
    if refuse:
        # Refused before any reference image, model preparation or render child.
        assert result is None and outcome.terminal != "succeeded", outcome
        assert not calls and not prefetches, (calls, prefetches)
        assert not refuse_code or outcome.code == refuse_code, outcome
        return {}
    if cancel >= 0:
        assert result is None and outcome.terminal == "canceled", outcome
        return {}
    if fail == 0:
        assert result is None and outcome.terminal != "succeeded", outcome
        return {}
    assert result is not None and outcome.terminal == "succeeded", outcome
    assert releases == [len(calls)], releases
    delivered = count if fail < 0 else fail
    assert result.result.delivered == delivered and result.result.complete == (fail < 0)
    # The joined video is the parent's only output; no last-frame image is delivered.
    assert [item["kind"] for item in result.outputs] == ["video"], result.outputs
    with av.open(io.BytesIO(result.result.video.read_bytes()), mode="r") as container:
        assert sum(1 for _ in container.decode(video=0)) == sum(
            min(duration_s, (362 - (context_frames if index else 0)) // FPS) * FPS
            if continuous else frames_for(duration_s)
            for index in range(delivered)
        )
    overall = [
        event.overall_fraction
        for event in events
        if isinstance(event, ProgressFrame) and event.overall_fraction is not None
    ]
    assert overall == sorted(overall) and (overall[-1] == 1) == (fail < 0)
    sent = [json.loads(call["payload"]) for call in calls[2:]]
    assert all("[Shot 1]" in row["payload"]["prompt"] for row in sent)
    assert sent[0]["payload"]["seed"] == 0
    assert len(prefetches) == 1
    fixed = [json.loads(answers[index]["result"])["image"]["digest"] for index in range(2)]
    fixed += [voice] if audio else []
    for row in sent:
        assert [item["asset"] for item in row["assets"]] == fixed
        if audio:
            prompt = row["payload"]["prompt"]
            assert "<Audio 1> is the supplied audio reference for <RoverVoice>." in prompt
            assert "<Audio 1>: reference - its timbre guides <Rover>'s speech." in prompt
        assert (
            "<Rover>" in row["payload"]["prompt"]
            and "fully_preserved" in row["payload"]["prompt"]
        )
    if continuous:
        assert sent[0]["payload"].get("context") is None
        for index, row in enumerate(sent[1:], 1):
            assert (
                row["payload"]["context"]
                == json.loads(answers[index + 1]["result"])["context"]["digest"]
            )
    return {"frames": result.result.delivered_frames, "calls": sent, "events": len(events)}


def admissible_results() -> None:
    """Runtime grants native result outputs by fixed field path at child admission."""
    built = static_interface.build(ROOT / "minimax-h3")
    interface = json.loads(package_interface.canonical_bytes(built))
    for entry in [*interface["entrypoints"], *interface["jobs"]]:
        list(machine_byte_results.paths(entry["result"]))


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True)
    admissible_results()
    results = {
        "fixed": drive(root / "fixed"),
        "standard": drive(root / "standard", turbo=False, count=2),
        "arbitrary_cut_count": drive(root / "arbitrary", count=10),
        "partial": drive(root / "partial", fail=2),
    }
    assert len(results["arbitrary_cut_count"]["calls"]) == 10
    drive(root / "reference-failed", reference_failure=True)
    drive(root / "reference-canceled", reference_cancel=True)
    drive(root / "canceled", cancel=1)
    drive(root / "first-fails", fail=0)
    repeated = drive(root / "retry", request_id="fixed")
    assert (
        repeated["calls"][1]["payload"]["seed"] == results["fixed"]["calls"][1]["payload"]["seed"]
    )
    drive(root / "oversized-prompt", prompt="x" * 4000, refuse=True)
    audio = drive(root / "audio", audio=True)
    assert all(len(row["assets"]) == 3 for row in audio["calls"])
    drive(
        root / "audio-only", audio=True, bad="audio-only", refuse=True, refuse_code="reference_policy"
    )
    for bad in (
        "empty-references",
        "duplicate",
        "empty",
        "seed",
        "seed-too-large",
        "empty-description",
        "legacy",
        "too-many-references",
    ):
        drive(root / bad, bad=bad, refuse=True)
    refs = [StoryReference(f"Person{i}", "character", "A person.") for i in range(9)]
    named = validate_references(refs)
    assert len(named) == 9
    assert reference_seed(refs[0], "request") == reference_seed(refs[0], "request")
    assert reference_seed(refs[0], "request") != reference_seed(refs[1], "request")
    prompt = segment_prompt(
        "A gathering", StorySegment(
            summary="[reference generation] A gathering.",
            detailed_description="[Shot 1] <Person8> waves.",
            overall_soundscape="Room tone.", non_diegetic_music="N/A", duration_s=10,
        ),
        index=0, subject_definitions="<Person8> is in <Picture 9>.", retention_analysis="<Person8>: fully_preserved - identity.",
    )
    assert "<Person8> waves" in prompt and "<Picture 9>" in prompt
    assert prompt.endswith("overall_soundscape:\nRoom tone.\n\nnon_diegetic_music:\nN/A")
    assert "first frame" not in prompt and "preceding shot" not in prompt
    assert set(ROUTES) == {"ref2va", "ref2va_turbo"}
    (root / "evidence.json").write_text(
        json.dumps(
            {
                "cases": results,
                "native_routes": ROUTES,
                "actual_h3_inference": False,
                "reference_interface_digest": REFERENCE_BINDING.interface_digest,
                "reference_caller": "Runtime-generated qwen_image_2.generate_image",
                "reference_submission_concurrent": True,
                "reverse_reference_completion": True,
                "reference_failure_cancels_siblings": True,
                "two_public_assets": True,
                "cuts_preserve_all_frames": True,
                "reference_slots_and_tokens_bounded": True,
                "grouped_appearance_roles": True,
                "supplied_audio_reference_skips_qwen": True,
                "audio_only_refused_before_model_preparation": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("cut routing, references, assembly, one-file custody, progress and interruption: PASS")


if __name__ == "__main__":
    main()
