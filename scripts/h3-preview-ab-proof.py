#!/usr/bin/env python3
"""CPU proofs for the private paired benchmark; no model downloads or GPU work."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import msgspec
import torch
from cozy_runtime.author import describe
from cozy_runtime.author.fakes import fake_attempt, fake_input, fake_outputs
from cozy_runtime.internal.static_interface import build
from cozy_runtime.models.minimax_h3.continuation import plan_continuation
from cozy_runtime.author._loader import Config
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3"))
import preview  # noqa: E402
from latent_upscale_net import LatentResizer3D  # noqa: E402


def run(evidence: Path) -> None:
    assert not torch.cuda.is_initialized()
    interface = build(ROOT / "minimax-h3")
    assert interface["application"] == "preview:app"
    assert {s.name for s in describe(preview.app)} == {
        "generate_latents",
        "upscale_latents",
        "decode_latents",
        "prepare_upscaler",
        "compare",
        "software",
    }
    model_calls = {s["name"]: s for s in interface["entrypoints"]}
    assert set(model_calls) == {"generate_latents", "upscale_latents", "decode_latents"}
    assert all(s["internal"] for s in model_calls.values())
    assert [
        f["name"]
        for f in next(s for s in interface["jobs"] if s["name"] == "compare")["result"]["fields"]
    ] == ["preview_upscaled", "native", "metadata"]
    (ROOT / "minimax-h3/metadata/package-interface.json").write_text(
        json.dumps(interface, indent=2) + "\n"
    )
    pair_job = next(row for row in interface["jobs"] if row["name"] == "compare")
    image_field = next(field for field in pair_job["request"]["fields"] if field["name"] == "reference_images")
    assert image_field["constraints"] == {"min_length": 2, "max_length": 2}
    assert "image/png" in image_field["asset_bound"]["media_types"]
    # Client admission uses this emitted bound before creating a remote run.
    inputs = json.loads((evidence / "inputs.json").read_text())
    assert all(Path(row["path"]).stat().st_size <= image_field["asset_bound"]["max_bytes"]
               for row in inputs["reference_files"])
    plan = plan_continuation(240)
    assert (plan.sample_frames, plan.delivered_frames, plan.prefix_frames) == (243, 240, 0)
    step = MiniMaxH3Ref2VASetupStep()
    pipe = MiniMaxH3ModularPipeline(blocks=step)
    pipe.register_components(image_processor=VaeImageProcessor(vae_scale_factor=16))
    for width, height in preview.CANVASES.values():
        state = PipelineState()
        for k, v in dict(
            height=height,
            width=width,
            num_frames=240,
            prompt="same",
            references=[MiniMaxH3ImageReference(Image.new("RGB", (64, 64)))],
            generator=torch.Generator().manual_seed(2768991793),
        ).items():
            state.set(k, v)
        step(pipe, state)
        assert (state.width, state.height, state.num_frames) == (width, height, 243)
    # Safe checkpoint header and meta-only census: no 1.3GB random model allocation.
    weight = evidence / "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
    with weight.open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    with torch.device("meta"):
        net = LatentResizer3D(**preview.WEIGHT_CONFIG)
        configured = preview.build_upscaler(Config(preview.WEIGHT_CONFIG))
    expected = net.state_dict()
    assert set(header) == set(expected)
    assert all(
        header[k]["shape"] == list(expected[k].shape) and header[k]["dtype"] == "F16"
        for k in header
    )
    assert sum(t.numel() for t in expected.values()) == 345280216
    assert set(configured.components) == {"upscaler"}
    # Compare the vendored architecture against the exact source extraction numerically.
    spec = importlib.util.spec_from_file_location(
        "upstream_upscale", evidence / "upstream_h3_latent_upscale.py"
    )
    assert spec and spec.loader
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    small = dict(
        in_channels=24,
        in_blocks=1,
        out_blocks=1,
        channels=32,
        dropout=0.1,
        temporal_every=1,
        temporal_kernel=3,
    )
    torch.manual_seed(31)
    original = upstream.LatentResizer3D(**small).eval()
    port = LatentResizer3D(**small).eval()
    port.load_state_dict(original.state_dict())
    video = torch.randn(1, 24, 3, 5, 10)
    with torch.inference_mode():
        expected_lift = original(video, 1.6, (3, 8, 16))
        actual_lift = port(video, 1.6, (3, 8, 16))
    assert torch.equal(expected_lift, actual_lift)
    # Real FileAsset persistence and rehydration keep the source audio bitwise identical.
    with tempfile.TemporaryDirectory(prefix="h3-preview-cpu-", dir=evidence) as area:
        out = fake_outputs(fake_attempt(spool=Path(area)))
        audio = torch.randn(2, 32, 9)
        info = preview.LatentInfo(
            schema="h3-preview-latents/1",
            width=160,
            height=80,
            duration_s=10,
            sampled_frames=243,
            delivered_frames=240,
            seed=2768991793,
            torch_seed="4073474745462816035",
            model_manifest="sha256:" + "1" * 64,
            turbo_lora_manifest="sha256:" + "2" * 64,
            prompt="same",
            reference_digests=["a", "b"],
            reference_sizes=[[1024, 1024], [1024, 1024]],
            video_shape=[],
            audio_shape=[],
            video_digest="",
            audio_digest="",
            schedule=preview.ScheduleInfo("plan", 8, 9, "v", "a", "vt", "at"),
            stages_s=preview.StageTimes(),
        )
        retained = preview.retain(out, video, audio, info)
        value = preview.Latents(
            fake_input(retained.file, attempt="decode"),
            msgspec.json.decode(msgspec.json.encode(info), type=preview.LatentInfo),
        )
        restored_video, restored_audio = preview.restore(value)
        assert torch.equal(video, restored_video) and torch.equal(audio, restored_audio)
        model = preview.LatentUpscaler.for_test(
            pipe=type("Pipeline", (), {"components": {"upscaler": port}})()
        )
        lifted = model.upscale(restored_video, 256, 128)
        mean = video.new_tensor(preview.LATENTS_MEAN).view(1, -1, 1, 1, 1)
        std = video.new_tensor(preview.LATENTS_STD).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            director = original((video - mean) / std, 1.6, (3, 8, 16)) * std + mean
        assert torch.equal(lifted, director)
        assert [call.components for call in model.harness.calls] == [("upscaler",)]
        assert torch.equal(restored_audio, audio)
        bad = msgspec.structs.replace(info, video_digest="bad")
        try:
            preview.restore(preview.Latents(value.file, bad))
        except ValueError as exc:
            assert "content" in str(exc)
        else:
            raise AssertionError("changed latent receipt was accepted")
    assert not torch.cuda.is_initialized()
    print(
        "PASS: registered native calls, exact canvases/clocks, full checkpoint census, upstream 1.6x numerical parity, scoped model, retained AV custody"
    )


if __name__ == "__main__":
    run(Path(os.environ["H3_PREVIEW_EVIDENCE"]))
