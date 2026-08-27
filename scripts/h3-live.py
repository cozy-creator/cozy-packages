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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "h3"
PLAN_FILES = {
    "first_last_frame_to_video": "fl2va.json",
    "reference_media_to_video": "ref2va.json",
}


@dataclass(frozen=True)
class RequestSnapshot:
    raw: bytes
    sha256: str
    seed: int


def exact_sha256(value: str, *, option: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RuntimeError(f"{option} must be one exact sha256:<64 lowercase hex> value")
    return value


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def verify_plan_digests(
    endpoint: Path, expected: dict[str, str]
) -> dict[str, str]:
    observed = {}
    for action, filename in PLAN_FILES.items():
        expected_digest = exact_sha256(
            expected[action], option=f"expected {action} plan digest"
        )
        path = endpoint / "timestep-plans" / filename
        try:
            digest = _sha256(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"cannot read selected endpoint plan {path}: {exc}") from exc
        if digest != expected_digest:
            raise RuntimeError(
                f"selected endpoint {action} plan is {digest}, expected {expected_digest}"
            )
        observed[action] = digest
    return observed


def verify_surface(document: dict[str, Any], *, expected: str) -> str:
    expected = exact_sha256(expected, option="--expected-surface-digest")
    actual = document.get("surface_digest")
    if actual != expected:
        raise RuntimeError(f"endpoint surface is {actual!r}, expected {expected}")
    return expected


def load_request(path: Path, *, expected_sha256: str) -> RequestSnapshot:
    expected_sha256 = exact_sha256(expected_sha256, option=f"expected digest for {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read request {path}: {exc}") from exc
    actual = _sha256(raw)
    if actual != expected_sha256:
        raise RuntimeError(f"request {path} is {actual}, expected {expected_sha256}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request {path} is not one UTF-8 JSON document") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"request {path} JSON root is not an object")
    seed = value.get("seed")
    if type(seed) is not int:
        raise RuntimeError(f"request {path} must contain one integer seed")
    return RequestSnapshot(raw=raw, sha256=actual, seed=seed)


def reserve_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"proof output already exists: {output}") from exc
    return output


def verify_file_digest(path: Path, *, expected_sha256: str) -> None:
    try:
        actual = _sha256(path.read_bytes())
    except OSError as exc:
        raise RuntimeError(f"cannot re-read staged request {path}: {exc}") from exc
    if actual != expected_sha256:
        raise RuntimeError(f"staged request {path} changed: {actual}, expected {expected_sha256}")


def stage_request(path: Path, snapshot: RequestSnapshot) -> Path:
    try:
        with path.open("xb") as staged:
            staged.write(snapshot.raw)
    except OSError as exc:
        raise RuntimeError(f"cannot stage request {path}: {exc}") from exc
    verify_file_digest(path, expected_sha256=snapshot.sha256)
    return path


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
    exact_sha256(expected_checkpoint, option="--expected-checkpoint")

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
    value = document.get("result")
    if not isinstance(value, dict):
        raise RuntimeError("Runtime result is not an object")
    return value


def verify_outcome(document: dict[str, Any]) -> None:
    if document.get("status") != "OUTCOME_STATUS_SUCCEEDED":
        raise RuntimeError(f"Runtime outcome is not succeeded: {document.get('status')!r}")
    expected_types = {
        "result": dict,
        "outputs": dict,
        "plan": dict,
        "ledger": dict,
        "timings": dict,
        "warnings": list,
    }
    malformed = {
        key: type(document.get(key)).__name__
        for key, expected_type in expected_types.items()
        if not isinstance(document.get(key), expected_type)
    }
    if malformed:
        raise RuntimeError(f"Runtime outcome is incomplete or malformed: {malformed}")


def verify_result(
    action: str,
    document: dict[str, Any],
    *,
    expected_checkpoint: str,
    expected_plan_digest: str,
) -> dict[str, Any]:
    verify_outcome(document)
    result = _result(document)
    expected_plan_digest = exact_sha256(
        expected_plan_digest, option=f"expected {action} plan digest"
    )
    expected = {
        "frames": 345,
        "fps": 24,
        "sample_rate": 32000,
        "sigma_grid_points": 30,
        "transformer_evaluations": 29,
        "timestep_plan_digest": expected_plan_digest.removeprefix("sha256:"),
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
    parser.add_argument("--expected-surface-digest", required=True)
    parser.add_argument("--expected-fl-plan-digest", required=True)
    parser.add_argument("--expected-ref-plan-digest", required=True)
    parser.add_argument("--fl-input", type=Path)
    parser.add_argument("--expected-fl-input-sha256")
    parser.add_argument("--ref-input", type=Path)
    parser.add_argument("--expected-ref-input-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not args.inspect_only and any(
        value is None
        for value in (
            args.fl_input,
            args.expected_fl_input_sha256,
            args.ref_input,
            args.expected_ref_input_sha256,
        )
    ):
        parser.error(
            "--fl-input, --expected-fl-input-sha256, --ref-input, and "
            "--expected-ref-input-sha256 are required unless --inspect-only is set"
        )
    return args


def main() -> int:
    args = parse_args()
    expected_checkpoint = exact_sha256(
        args.expected_checkpoint, option="--expected-checkpoint"
    )
    expected_surface = exact_sha256(
        args.expected_surface_digest, option="--expected-surface-digest"
    )
    expected_plans = {
        "first_last_frame_to_video": exact_sha256(
            args.expected_fl_plan_digest, option="--expected-fl-plan-digest"
        ),
        "reference_media_to_video": exact_sha256(
            args.expected_ref_plan_digest, option="--expected-ref-plan-digest"
        ),
    }
    endpoint_path = args.endpoint.expanduser().resolve()
    if not endpoint_path.is_dir():
        raise RuntimeError(f"selected endpoint is not a directory: {endpoint_path}")
    plan_digests = verify_plan_digests(endpoint_path, expected_plans)

    snapshots: dict[str, RequestSnapshot] = {}
    if not args.inspect_only:
        assert args.fl_input is not None
        assert args.ref_input is not None
        assert args.expected_fl_input_sha256 is not None
        assert args.expected_ref_input_sha256 is not None
        snapshots = {
            "first_last_frame_to_video": load_request(
                args.fl_input.expanduser().resolve(),
                expected_sha256=args.expected_fl_input_sha256,
            ),
            "reference_media_to_video": load_request(
                args.ref_input.expanduser().resolve(),
                expected_sha256=args.expected_ref_input_sha256,
            ),
        }

    output_root = reserve_output(args.out)
    staged_requests: dict[str, Path] = {}
    if snapshots:
        requests_root = output_root / "requests"
        requests_root.mkdir()
        staged_requests = {
            "first_last_frame_to_video": stage_request(
                requests_root / "fl2va.json", snapshots["first_last_frame_to_video"]
            ),
            "reference_media_to_video": stage_request(
                requests_root / "ref2va.json", snapshots["reference_media_to_video"]
            ),
        }

    base = [args.runtime, "--dir", str(endpoint_path), "--json"]
    description = command_json([*base, "describe", "--check"])
    surface_digest = verify_surface(description, expected=expected_surface)
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
        expected_checkpoint=expected_checkpoint,
    )
    receipt: dict[str, Any] = {
        "schema": "cozy.minimax_h3.production_probe/2",
        "endpoint": {
            "resolved_path": str(endpoint_path),
            "surface_digest": surface_digest,
            "plan_digests": plan_digests,
        },
        "doctor": doctor,
        "bindings": bindings,
        "expected_gpu": args.expected_gpu,
        "expected_binding_ref": args.expected_binding_ref,
        "expected_checkpoint": expected_checkpoint,
        "automated_status": "provider-and-binding-inspected",
        "human_viewed_listened_status": "pending",
    }
    if not args.inspect_only:
        actions = (
            ("first_last_frame_to_video", output_root / "fl2va"),
            ("reference_media_to_video", output_root / "ref2va"),
        )
        results = {}
        for action, output in actions:
            payload = staged_requests[action]
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
            verify_file_digest(payload, expected_sha256=snapshots[action].sha256)
            result = verify_result(
                action,
                document,
                expected_checkpoint=expected_checkpoint,
                expected_plan_digest=plan_digests[action],
            )
            results[action] = {
                "request": {
                    "sha256": snapshots[action].sha256,
                    "bytes": len(snapshots[action].raw),
                    "seed": snapshots[action].seed,
                    "staged_path": str(payload),
                },
                "runtime": document,
            }
            results[action]["media"] = probe_media(output, result)
        receipt["actions"] = results
        receipt["automated_status"] = "both-actions-stored-media-green"
    receipt_path = output_root / "h3-production-probe.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
