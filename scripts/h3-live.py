#!/usr/bin/env python
"""Installed-artifact MiniMax-H3 production probe.

Run this inside the fixed RunPod worker only after provider readback and Runtime `doctor`
agree on the exact admitted GPU. It executes both public actions through the installed
Runtime, probes their stored MP4/PNG media, and writes one machine receipt. It does not
select/rent a pod, fetch an artifact, or turn automated checks into viewed/listened proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "h3"
PLAN_DIGESTS = {
    "first_last_frame_to_video": hashlib.sha256(
        (H3 / "timestep-plans" / "fl2va.json").read_bytes()
    ).hexdigest(),
    "reference_media_to_video": hashlib.sha256(
        (H3 / "timestep-plans" / "ref2va.json").read_bytes()
    ).hexdigest(),
}


def command_json(command: list[str], *, timeout: int = 7200) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(command)}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command printed invalid JSON: {' '.join(command)}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"command JSON is not an object: {' '.join(command)}")
    return value


def verify_bindings(
    document: dict[str, Any], *, expected_ref: str, expected_checkpoint: str
) -> list[dict[str, Any]]:
    if not expected_checkpoint.startswith("sha256:") or len(expected_checkpoint) != 71:
        raise RuntimeError(
            "--expected-checkpoint must be one exact sha256:<64 lowercase hex> value"
        )
    if any(character not in "0123456789abcdef" for character in expected_checkpoint[7:]):
        raise RuntimeError("--expected-checkpoint is not lowercase hexadecimal")

    bindings = document.get("bindings")
    if not isinstance(bindings, list) or not all(isinstance(row, dict) for row in bindings):
        raise RuntimeError("Runtime bindings result has no binding-record list")
    records = cast(list[dict[str, Any]], bindings)
    expected_paths = {
        "first_last_frame_to_video.models.model",
        "reference_media_to_video.models.model",
    }
    if len(records) != 2 or {row.get("model_binding_path") for row in records} != expected_paths:
        raise RuntimeError("Runtime bindings do not contain exactly the two H3 model slots")

    expected_components = {
        "audio_vae",
        "text_encoder",
        "transformer",
        "transformer_ref",
        "video_vae",
    }
    for row in records:
        path = row["model_binding_path"]
        if row.get("ref") != expected_ref:
            raise RuntimeError(f"{path} resolved {row.get('ref')!r}, expected {expected_ref!r}")
        if row.get("installed") is not True:
            raise RuntimeError(f"{path} is not installed")
        if set(row.get("components", ())) != expected_components:
            raise RuntimeError(f"{path} does not resolve the complete dual H3 component set")
        snapshots = row.get("snapshots")
        if not isinstance(snapshots, dict) or set(snapshots) != expected_components:
            raise RuntimeError(f"{path} has an incomplete component snapshot map")
        if set(snapshots.values()) != {expected_checkpoint}:
            raise RuntimeError(f"{path} does not resolve only {expected_checkpoint}")
        stamps = row.get("stamps")
        tasks = stamps.get("task") if isinstance(stamps, dict) else None
        if not isinstance(tasks, list) or set(tasks) != {"fl2va", "ref2va"}:
            raise RuntimeError(f"{path} does not carry the exact dual task stamp")
    return records


def _result(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("result", document)
    if not isinstance(value, dict):
        raise RuntimeError("Runtime result is not an object")
    return value


def verify_result(
    action: str, document: dict[str, Any], *, expected_checkpoint: str
) -> dict[str, Any]:
    result = _result(document)
    expected = {
        "frames": 345,
        "fps": 24,
        "sample_rate": 32000,
        "sigma_grid_points": 30,
        "transformer_evaluations": 29,
        "timestep_plan_digest": PLAN_DIGESTS[action],
        "checkpoint": expected_checkpoint,
    }
    mismatches = {
        key: {"got": result.get(key), "want": want}
        for key, want in expected.items()
        if result.get(key) != want
    }
    if mismatches:
        raise RuntimeError(f"{action} receipt mismatch: {mismatches}")
    for key in (
        "video_sigma_digest",
        "audio_sigma_digest",
        "video_timestep_digest",
        "audio_timestep_digest",
        "video_pixel_digest",
        "audio_sample_digest",
        "continuation_frame_digest",
        "continuation_pixel_digest",
    ):
        if not result.get(key):
            raise RuntimeError(f"{action} result has no {key}")
    return result


def _digest(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("digest")
    if not isinstance(value, str):
        raise RuntimeError(f"output asset has no digest: {value!r}")
    return value.removeprefix("sha256:")


def probe_media(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    import av

    videos = sorted(output.rglob("*.mp4"))
    images = sorted(output.rglob("*.png"))
    if len(videos) != 1 or len(images) != 1:
        raise RuntimeError(
            f"expected one MP4 and one PNG under {output}, got {len(videos)} and {len(images)}"
        )

    video_path, image_path = videos[0], images[0]
    with av.open(str(video_path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise RuntimeError("output MP4 must contain exactly one video and one audio stream")
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        frames = 0
        first_mean = last_mean = 0.0
        for frame in container.decode(video=0):
            pixels = frame.to_ndarray(format="rgb24")
            mean = float(pixels.mean())
            if frames == 0:
                first_mean = mean
            last_mean = mean
            frames += 1
        rate = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        if frames != 345 or rate != 24.0:
            raise RuntimeError(f"stored video clock is {frames} frames at {rate} fps")
        if not math.isfinite(first_mean + last_mean):
            raise RuntimeError("stored video probe observed a non-finite frame mean")
        sample_rate = int(audio_stream.codec_context.sample_rate)
        if sample_rate != 32000:
            raise RuntimeError(f"stored audio rate is {sample_rate}, expected 32000")

    with av.open(str(video_path)) as container:
        audio_samples = sum(frame.samples for frame in container.decode(audio=0))
    audio_seconds = audio_samples / sample_rate
    video_seconds = frames / rate
    if abs(audio_seconds - video_seconds) > 1 / rate:
        raise RuntimeError(
            f"stored A/V durations disagree: {video_seconds:.6f}s video, {audio_seconds:.6f}s audio"
        )

    with av.open(str(image_path)) as container:
        decoded = list(container.decode(video=0))
        if len(decoded) != 1:
            raise RuntimeError(f"continuation PNG decoded {len(decoded)} frames")
        image = decoded[0].to_ndarray(format="rgb24")

    video_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    continuation_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    continuation_pixels = hashlib.sha256(image).hexdigest()
    if video_sha != _digest(result["video"]):
        raise RuntimeError("stored MP4 digest differs from the returned VideoAsset")
    if continuation_sha != _digest(result["continuation_frame"]):
        raise RuntimeError("stored PNG digest differs from the returned ImageAsset")
    if continuation_sha != _digest(result["continuation_frame_digest"]):
        raise RuntimeError("stored PNG digest differs from continuation_frame_digest")
    if continuation_pixels != result["continuation_pixel_digest"]:
        raise RuntimeError("decoded PNG pixels differ from the pre-encode continuation pixels")
    if (int(image.shape[1]), int(image.shape[0])) != (result["width"], result["height"]):
        raise RuntimeError("continuation PNG dimensions differ from the result geometry")

    return {
        "video": str(video_path),
        "video_sha256": video_sha,
        "video_bytes": video_path.stat().st_size,
        "video_frames": frames,
        "video_fps": rate,
        "audio_sample_rate": sample_rate,
        "audio_samples": audio_samples,
        "audio_seconds": audio_seconds,
        "first_frame_mean": first_mean,
        "last_frame_mean": last_mean,
        "continuation": str(image_path),
        "continuation_sha256": continuation_sha,
        "continuation_pixel_sha256": continuation_pixels,
        "continuation_bytes": image_path.stat().st_size,
        "continuation_width": int(image.shape[1]),
        "continuation_height": int(image.shape[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="cozy-runtime")
    parser.add_argument("--endpoint", type=Path, default=H3)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expected-binding-ref", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument("--fl-input", type=Path)
    parser.add_argument("--ref-input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not args.inspect_only and (args.fl_input is None or args.ref_input is None):
        parser.error("--fl-input and --ref-input are required unless --inspect-only is set")
    return args


def main() -> int:
    args = parse_args()
    base = [args.runtime, "--dir", str(args.endpoint), "--json"]
    doctor = command_json([*base, "doctor"])
    device = doctor.get("device")
    actual_gpu = device.get("name") if isinstance(device, dict) else None
    if actual_gpu != args.expected_gpu:
        raise RuntimeError(
            f"doctor GPU identity {actual_gpu!r} differs from admitted identity "
            f"{args.expected_gpu!r}; refusing before inference"
        )
    bindings = command_json([*base, "bindings"])
    verify_bindings(
        bindings,
        expected_ref=args.expected_binding_ref,
        expected_checkpoint=args.expected_checkpoint,
    )
    receipt: dict[str, Any] = {
        "schema": "cozy.minimax_h3.production_probe/1",
        "doctor": doctor,
        "bindings": bindings,
        "expected_gpu": args.expected_gpu,
        "expected_binding_ref": args.expected_binding_ref,
        "expected_checkpoint": args.expected_checkpoint,
        "automated_status": "provider-and-binding-inspected",
        "human_viewed_listened_status": "pending",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.inspect_only:
        assert args.fl_input is not None
        assert args.ref_input is not None
        actions = (
            ("first_last_frame_to_video", args.fl_input, args.out / "fl2va"),
            ("reference_media_to_video", args.ref_input, args.out / "ref2va"),
        )
        results = {}
        for action, payload, output in actions:
            document = command_json(
                [
                    *base,
                    "run",
                    action,
                    "--in",
                    str(payload),
                    "--out",
                    str(output),
                    "--offline",
                ]
            )
            results[action] = {
                "result": verify_result(
                    action, document, expected_checkpoint=args.expected_checkpoint
                ),
            }
            results[action]["media"] = probe_media(output, results[action]["result"])
        receipt["actions"] = results
        receipt["automated_status"] = "both-actions-stored-media-green"
    receipt_path = args.out / "h3-production-probe.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
