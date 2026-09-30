#!/usr/bin/env python3
"""Real managed-call serialization/custody for the private pair; GPU math is synthetic."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import av
import msgspec
import numpy as np
import torch
from PIL import Image
from cozy_runtime.author import (
    Invocation,
    ModelArtifact,
    ObjectRef,
    SourceArtifact,
    attempt,
    describe,
)
from cozy_runtime.author._assets import Asset, GrantedInput, file_state
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._executor_requests import (
    Answer,
    CallState,
    ChildCall,
    ChildForget,
    ChildPoll,
    GpuRelease,
    Request,
)
from cozy_runtime.author.fakes import fake_attempt, fake_input, fake_outputs
from cozy_runtime.internal import canonical, source_interfaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3"))
import preview  # noqa: E402


def run(root: Path) -> None:
    parent = "preview-pair-cpu"
    surfaces = {s.name: s for s in describe(preview.app)}
    bindings = {
        ("preview", name): _CallType(
            "sha256:" + "5" * 64,
            "preview",
            name,
            cast(type[msgspec.Struct], surfaces[name].payload_type),
            cast(type[msgspec.Struct], surfaces[name].result_type),
        )
        for name in ("prepare_upscaler", "generate_latents", "upscale_latents", "decode_latents")
    }
    bindings.update(
        {(source_interfaces.MODULE, n): v for n, v in source_interfaces.bindings().items()}
    )
    raw_source = SourceArtifact(
        "source", "source", ObjectRef("sha256:" + "1" * 64, 10), "sha256:" + "2" * 64
    )
    native_source = ModelArtifact(
        "native", "model", ObjectRef("sha256:" + "3" * 64, 10), "sha256:" + "4" * 64
    )
    checkpoint = ModelArtifact(
        "prepared", "upscaler", ObjectRef("sha256:" + "6" * 64, 10), "sha256:" + "7" * 64
    )
    answers: dict[int, CallState] = {}
    held: dict[str, Asset] = {}
    sequence: list[str] = []
    generations: list[dict[str, Any]] = []
    source_inputs: list[dict[str, Any]] = []

    def hydrate(raw: dict[str, Any], consumer: str) -> preview.Latents:
        ref = raw["file"]
        asset = held[ref]
        assert asset.kind == "file"
        return preview.Latents(
            fake_input(asset, attempt=consumer), msgspec.convert(raw["info"], preview.LatentInfo)
        )

    def exchange[A: Answer](request: Request, into: type[A], /) -> A:
        response = worker(request)
        assert isinstance(response, into)
        return response

    def worker(request: Request) -> Answer:
        if isinstance(request, GpuRelease):
            return Answer(ok=True)
        assert isinstance(request, ChildCall | ChildPoll | ChildForget), request
        if isinstance(request, ChildForget):
            return CallState(ok=True)
        if isinstance(request, ChildPoll):
            return answers[request.call_index]
        sequence.append(request.export)
        raw = json.loads(request.payload)
        name = f"{parent}-{request.call_index}"
        out = fake_outputs(fake_attempt(name, spool=root / name))
        if request.export == "download_huggingface":
            assert raw["repository"] == "LBH-123-AI/Minimax_h3_latent_Upscaler"
            assert raw["revision"] == "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"
            assert len(raw["carriers"]) == 1 and raw["carriers"][0].endswith("_fp16.safetensors")
            value: Any = raw_source
        elif request.export == "convert_cozytensors":
            assert raw["source"] == msgspec.to_builtins(raw_source)
            assert raw["profiles"] == ["as-is/1"]
            value = native_source
        elif request.export == "prepare_upscaler":
            assert raw["source"] == msgspec.to_builtins(native_source)
            value = checkpoint
        elif request.export == "generate_latents":
            generations.append(raw)
            assert raw["base_model"] is None and raw["turbo_lora"] is None
            data = raw["payload"]
            assert data["duration_s"] == 10 and data["seed"] == 2768991793
            width, height = preview.CANVASES[data["canvas"]]
            info = preview.LatentInfo(
                schema="h3-preview-latents/1",
                width=width,
                height=height,
                duration_s=10,
                sampled_frames=243,
                delivered_frames=240,
                seed=data["seed"],
                torch_seed="4073474745462816035",
                model_manifest="sha256:" + "1" * 64,
                turbo_lora_manifest="sha256:" + "2" * 64,
                prompt=data["prompt"],
                reference_digests=[row["asset"] for row in raw["assets"]],
                reference_sizes=[[1024, 1024], [1024, 1024]],
                video_shape=[],
                audio_shape=[],
                video_digest="",
                audio_digest="",
                schedule=preview.ScheduleInfo("plan", 8, 9, "v", "a", "vt", "at"),
                stages_s=preview.StageTimes(denoise=1.0),
            )
            value = preview.retain(
                out, torch.zeros(1, 24, 2, height // 16, width // 16), torch.ones(2, 32, 9), info
            )
        elif request.export == "upscale_latents":
            assert raw["model"] == msgspec.to_builtins(checkpoint)
            retained = hydrate(raw["payload"]["latents"], name)
            video, audio = preview.restore(retained)
            assert (retained.info.width, retained.info.height) == (960, 480)
            info = msgspec.structs.replace(
                retained.info,
                width=1536,
                height=768,
                source_video_digest=retained.info.video_digest,
                upscale_scale=1.6,
                upscale_weights=checkpoint.manifest.digest,
            )
            value = preview.retain(out, torch.zeros(1, 24, video.shape[2], 48, 96), audio, info)
            assert value.info.audio_digest == retained.info.audio_digest
        elif request.export == "decode_latents":
            retained = hydrate(raw["payload"]["latents"], name)
            preview.restore(retained)
            assert (retained.info.width, retained.info.height) == (1536, 768)
            source_inputs.append(msgspec.to_builtins(retained.info))
            pixels = np.full((240, 64, 96, 3), 63 if len(source_inputs) == 1 else 127, np.uint8)
            video = out.save_video(
                pixels, fps=24, audio=np.zeros((2, 320000), np.float32), sample_rate=32000
            )
            value = preview.Rendered(video, retained.info, 0.5, [])
        else:
            raise AssertionError(request.export)
        outputs = []

        def project(item: Any, path: str = "") -> Any:
            if isinstance(item, Asset):
                raw = item.read_bytes()
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                local = root / digest[7:]
                if not local.exists():
                    local.write_bytes(raw)
                held[digest] = item
                outputs.append(
                    dict(
                        output_id=path,
                        kind=item.kind,
                        digest=digest,
                        length=len(raw),
                        media_type=item.media_type,
                        local=str(local),
                    )
                )
                return dict(
                    asset_ref=digest,
                    kind=item.kind,
                    digest=digest,
                    size_bytes=len(raw),
                    media_type=item.media_type,
                )
            if isinstance(item, msgspec.Struct):
                return {
                    field: project(getattr(item, field), f"{path}.{field}" if path else field)
                    for field in item.__struct_fields__
                }
            if isinstance(item, list):
                return [project(v, f"{path}.{i}") for i, v in enumerate(item)]
            return item

        answers[request.call_index] = CallState(
            ok=True,
            state="succeeded",
            result=canonical.write(project(value)).decode(),
            byte_grants=tuple(outputs),
        )
        return CallState(ok=True, child_request_id=name)

    assets = {}
    refs = []
    for i in range(2):
        file = root / f"ref{i}.png"
        Image.new("RGB", (64, 64), (80 + i, 90, 100)).save(file)
        digest = "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest()
        refs.append(digest)
        key = f"reference_images.{i}"
        assets[key] = GrantedInput(
            input_id=key,
            local=file,
            media_type="image/png",
            digest=digest,
            length=file.stat().st_size,
            order=i,
            file_state=file_state(file),
        )
    broker = _Broker(parent, bindings, exchange)
    result, outcome, _record = attempt(
        preview.app.get("compare"),
        dict(
            prompt="summary:\nSame authored prompt.",
            seed=2768991793,
            reference_images=refs,
            duration_s=10,
        ),
        Invocation(parent, root / "parent", time.monotonic() + 120, calls=broker, assets=assets),
    )
    assert result is not None and outcome.terminal == "succeeded", outcome
    assert sequence == [
        "download_huggingface",
        "convert_cozytensors",
        "prepare_upscaler",
        "generate_latents",
        "upscale_latents",
        "decode_latents",
        "generate_latents",
        "decode_latents",
    ], sequence
    assert len(generations) == 2 and generations[0]["assets"] == generations[1]["assets"]
    assert generations[0]["payload"]["prompt"] == generations[1]["payload"]["prompt"]
    assert (
        generations[0]["payload"]["canvas"] == "preview"
        and generations[1]["payload"]["canvas"] == "native"
    )
    assert sorted(item["kind"] for item in result.outputs) == ["file", "video", "video"], (
        result.outputs
    )
    metadata = json.loads(result.result.metadata.read_bytes())
    assert (
        metadata["A"]["latents"]["upscale_scale"] == 1.6
        and metadata["B"]["latents"]["upscale_scale"] is None
    )
    assert metadata["A"]["latents"]["torch_seed"] == "4073474745462816035"
    assert metadata["B"]["latents"]["torch_seed"] == "4073474745462816035"
    for video in [result.result.preview_upscaled, result.result.native]:
        with av.open(str(video._local)) as container:
            assert len(list(container.decode(video=0))) == 240
    print(
        "PASS: actual managed broker routes all8 calls, retained native/model/AV receipts, matched inputs, separate upscale, exact3 final assets, real10s MP4 codec"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="h3-preview-broker-", dir=Path(__file__).resolve().parents[1]
    ) as area:
        run(Path(area))
