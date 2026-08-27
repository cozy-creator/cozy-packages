#!/usr/bin/env python
"""Installed-endpoint MiniMax-H3 local proof stage.

Run this inside the fixed RunPod worker only after external provider and release receipts exist.
It executes the selected public actions through the installed Runtime, probes their stored
MP4/PNG media, and writes one local machine receipt. It does not inspect provider state,
establish release or container identity, or turn automated checks into viewed/listened proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
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
BINDING_PATHS = {
    "first_last_frame_to_video": "first_last_frame_to_video.models.model",
    "reference_media_to_video": "reference_media_to_video.models.model",
}
RECEIPT_SCHEMA = "cozy.minimax_h3.production_probe/3"


@dataclass(frozen=True)
class RequestSnapshot:
    raw: bytes
    sha256: str
    seed: int


def exact_sha256(value: Any, *, option: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RuntimeError(f"{option} must be one exact sha256:<64 lowercase hex> value")
    return value


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _bare_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_bare_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def _float_digest(values: list[float]) -> str:
    return _bare_sha256(b"".join(struct.pack("<f", value) for value in values))


def _schedule_digests(document: dict[str, Any]) -> dict[str, str]:
    try:
        evaluations = document["evaluations"]
        terminal = document["terminal"]
        video_sigmas = [float.fromhex(row["video_sigma"]) for row in evaluations]
        audio_sigmas = [float.fromhex(row["audio_sigma"]) for row in evaluations]
        video_sigmas.append(float.fromhex(terminal["video_sigma"]))
        audio_sigmas.append(float.fromhex(terminal["audio_sigma"]))

        def timesteps(name: str) -> list[float]:
            return [
                float.fromhex(
                    next(
                        item["timestep"]
                        for item in row["modulation_classes"]
                        if item["name"] == name
                    )
                )
                for row in evaluations
            ]

        return {
            "video_sigma_digest": _float_digest(video_sigmas),
            "audio_sigma_digest": _float_digest(audio_sigmas),
            "video_timestep_digest": _float_digest(timesteps("target_video")),
            "audio_timestep_digest": _float_digest(timesteps("target_audio")),
        }
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise RuntimeError("selected endpoint timestep plan is malformed") from exc


def load_plan_facts(endpoint: Path, expected: dict[str, str]) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for action, filename in PLAN_FILES.items():
        expected_digest = exact_sha256(expected[action], option=f"expected {action} plan digest")
        path = endpoint / "timestep-plans" / filename
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"cannot read selected endpoint plan {path}: {exc}") from exc
        digest = _sha256(raw)
        if digest != expected_digest:
            raise RuntimeError(
                f"selected endpoint {action} plan is {digest}, expected {expected_digest}"
            )
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"selected endpoint plan {path} is not UTF-8 JSON") from exc
        if not isinstance(document, dict):
            raise RuntimeError(f"selected endpoint plan {path} is not a JSON object")
        observed[action] = {
            "document_digest": digest,
            **_schedule_digests(document),
        }
    return observed


def verify_descriptor_digest(document: dict[str, Any], *, expected: str) -> str:
    expected = exact_sha256(expected, option="--expected-descriptor-digest")
    actual = document.get("descriptor_digest")
    if actual != expected:
        raise RuntimeError(f"endpoint descriptor is {actual!r}, expected {expected}")
    return expected


def visible_actions(document: dict[str, Any]) -> set[str]:
    entrypoints = document.get("entrypoints")
    if not isinstance(entrypoints, list) or not all(isinstance(row, dict) for row in entrypoints):
        raise RuntimeError("Runtime descriptor has no entrypoint list")
    actions = {
        row.get("name")
        for row in cast(list[dict[str, Any]], entrypoints)
        if row.get("hidden") is not True
    }
    if not actions or not all(
        isinstance(action, str) and action in PLAN_FILES for action in actions
    ):
        rendered = sorted(str(action) for action in actions)
        raise RuntimeError(f"Runtime descriptor has invalid visible H3 actions: {rendered}")
    return cast(set[str], actions)


def select_actions(values: list[str] | None) -> set[str]:
    return set(PLAN_FILES) if not values else set(values)


def require_visible(selected: set[str], descriptor: dict[str, Any]) -> set[str]:
    visible = visible_actions(descriptor)
    missing = selected - visible
    if missing:
        raise RuntimeError(
            f"selected actions are not visible in the installed descriptor: {sorted(missing)}"
        )
    return visible


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
    if value.get("mute") is not False:
        raise RuntimeError(f"request {path} must explicitly contain mute:false for listened proof")
    return RequestSnapshot(raw=raw, sha256=actual, seed=seed)


def reserve_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"proof output already exists: {output}") from exc
    return output


def verify_staged_request(path: Path, snapshot: RequestSnapshot) -> None:
    if path.stat().st_mode & 0o222:
        raise RuntimeError(f"staged request is writable: {path}")
    try:
        actual = _sha256(path.read_bytes())
    except OSError as exc:
        raise RuntimeError(f"cannot re-read staged request {path}: {exc}") from exc
    if actual != snapshot.sha256:
        raise RuntimeError(f"staged request {path} changed: {actual}, expected {snapshot.sha256}")


def stage_request(path: Path, snapshot: RequestSnapshot) -> Path:
    try:
        with path.open("xb") as staged:
            staged.write(snapshot.raw)
            staged.flush()
            os.fsync(staged.fileno())
    except OSError as exc:
        raise RuntimeError(f"cannot stage request {path}: {exc}") from exc
    path.chmod(0o444)
    verify_staged_request(path, snapshot)
    return path


def make_requests_read_only(directory: Path, paths: list[Path]) -> None:
    directory.chmod(0o555)
    for path in paths:
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"staged request is writable: {path}")


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
    document: dict[str, Any],
    *,
    expected_ref: str,
    expected_checkpoint: str,
    expected_actions: set[str] | None = None,
) -> list[dict[str, Any]]:
    exact_sha256(expected_checkpoint, option="--expected-checkpoint")

    bindings = document.get("bindings")
    if not isinstance(bindings, list) or not all(isinstance(row, dict) for row in bindings):
        raise RuntimeError("Runtime bindings result has no binding-record list")
    records = cast(list[dict[str, Any]], bindings)
    actions = set(PLAN_FILES) if expected_actions is None else expected_actions
    expected_paths = {BINDING_PATHS[action] for action in actions}
    if (
        len(records) != len(expected_paths)
        or {row.get("model_binding_path") for row in records} != expected_paths
    ):
        raise RuntimeError(
            f"Runtime bindings do not contain exactly the selected H3 model slots: "
            f"{sorted(expected_paths)}"
        )

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


def verify_outcome(document: dict[str, Any]) -> dict[str, str]:
    if document.get("status") != "OUTCOME_STATUS_SUCCEEDED":
        raise RuntimeError(f"Runtime outcome is not succeeded: {document.get('status')!r}")
    expected_keys = {
        "status",
        "result",
        "outputs",
        "invocation",
        "plan",
        "ledger",
        "timings",
        "warnings",
    }
    if set(document) != expected_keys:
        raise RuntimeError(
            f"Runtime outcome fields differ: missing={sorted(expected_keys - set(document))}, "
            f"unexpected={sorted(set(document) - expected_keys)}"
        )
    plan = document.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("Runtime outcome has no accepted plan")
    exact_sha256(plan.get("plan_digest"), option="Runtime accepted plan digest")
    exact_sha256(
        plan.get("model_construction_digest"),
        option="Runtime accepted model-construction digest",
    )
    invocation = document.get("invocation")
    if not isinstance(invocation, dict) or set(invocation) != {"digest", "document"}:
        raise RuntimeError("Runtime outcome has no canonical invocation receipt")
    invocation_digest = exact_sha256(
        invocation.get("digest"), option="Runtime canonical invocation digest"
    )
    if not isinstance(invocation.get("document"), dict) or not invocation["document"]:
        raise RuntimeError("Runtime canonical invocation receipt has no document")
    if plan.get("invocation_spec_digest") != invocation_digest:
        raise RuntimeError("Runtime accepted plan and canonical invocation receipt disagree")

    outputs = document.get("outputs")
    if (
        not isinstance(outputs, dict)
        or set(outputs) != {"video", "continuation_frame"}
        or not all(isinstance(value, str) and value for value in outputs.values())
    ):
        raise RuntimeError("Runtime outcome must contain exactly video and continuation_frame")

    ledger = document.get("ledger")
    if not isinstance(ledger, dict) or not ledger:
        raise RuntimeError("Runtime outcome has no ledger")
    timings = document.get("timings")
    if not isinstance(timings, dict) or not timings:
        raise RuntimeError("Runtime outcome has no timings")
    if document.get("warnings") != []:
        raise RuntimeError(f"Runtime outcome carries warnings: {document.get('warnings')!r}")
    return cast(dict[str, str], outputs)


def verify_result(action: str, document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The customer result is EXACTLY the two-field catalog shape (se-012): checkpoint,
    plan, geometry, and digest facts are attempt observations, never result fields."""
    runtime_outputs = verify_outcome(document)
    result = _result(document)
    if set(result) != {"video", "continuation_frame"}:
        raise RuntimeError(
            f"{action} result is not the two-field catalog shape: got {sorted(result)}"
        )
    _digest(result["video"])
    _digest(result["continuation_frame"])
    return result, runtime_outputs


def _digest(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("digest")
    if not isinstance(value, str):
        raise RuntimeError(f"output asset has no digest: {value!r}")
    if value.startswith("sha256:"):
        return exact_sha256(value, option="output asset digest").removeprefix("sha256:")
    return exact_bare_sha256(value, field="output asset digest")


def verify_media_contract(
    *,
    video_format: str,
    video_major_brand: str,
    video_codec: str,
    audio_codec: str,
    video_width: int,
    video_height: int,
    image_format: str,
    image_codec: str,
    continuation_width: int,
    continuation_height: int,
) -> None:
    if "mp4" not in video_format.split(","):
        raise RuntimeError(f"stored video container is {video_format!r}, expected MP4")
    if video_major_brand != "isom":
        raise RuntimeError(
            f"stored video major brand is {video_major_brand!r}, expected exact MP4 brand 'isom'"
        )
    if video_codec != "h264" or audio_codec != "aac":
        raise RuntimeError(
            f"stored MP4 codecs are {video_codec!r}/{audio_codec!r}, expected H264/AAC"
        )
    if (continuation_width, continuation_height) != (video_width, video_height):
        raise RuntimeError("stored continuation dimensions differ from the stored video's")
    if image_format != "png_pipe" or image_codec != "png":
        raise RuntimeError(
            f"stored continuation container/codec is {image_format!r}/{image_codec!r}, expected PNG"
        )


def probe_media(
    output: Path, result: dict[str, Any], runtime_outputs: dict[str, str]
) -> dict[str, Any]:
    import av

    output = output.resolve()
    video_path = Path(runtime_outputs["video"]).resolve()
    image_path = Path(runtime_outputs["continuation_frame"]).resolve()
    try:
        video_path.relative_to(output)
        image_path.relative_to(output)
    except ValueError as exc:
        raise RuntimeError("Runtime output grant escaped the fresh action directory") from exc
    if not video_path.is_file() or not image_path.is_file():
        raise RuntimeError("Runtime output grant does not name two stored media files")
    with av.open(str(video_path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise RuntimeError("output MP4 must contain exactly one video and one audio stream")
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        video_format = container.format.name
        video_major_brand = container.metadata.get("major_brand", "")
        video_codec = video_stream.codec_context.name
        audio_codec = audio_stream.codec_context.name
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
        image_format = container.format.name
        decoded = list(container.decode(video=0))
        if len(decoded) != 1:
            raise RuntimeError(f"continuation PNG decoded {len(decoded)} frames")
        image = decoded[0].to_ndarray(format="rgb24")
        image_codec = container.streams.video[0].codec_context.name

    verify_media_contract(
        video_format=video_format,
        video_major_brand=video_major_brand,
        video_codec=video_codec,
        audio_codec=audio_codec,
        video_width=int(video_stream.width),
        video_height=int(video_stream.height),
        image_format=image_format,
        image_codec=image_codec,
        continuation_width=int(image.shape[1]),
        continuation_height=int(image.shape[0]),
    )

    video_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    continuation_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    continuation_pixels = hashlib.sha256(image.tobytes(order="C")).hexdigest()
    if video_sha != _digest(result["video"]):
        raise RuntimeError("stored MP4 digest differs from the returned VideoAsset")
    if continuation_sha != _digest(result["continuation_frame"]):
        raise RuntimeError("stored PNG digest differs from the returned ImageAsset")

    return {
        "video": str(video_path),
        "video_sha256": video_sha,
        "video_bytes": video_path.stat().st_size,
        "video_frames": frames,
        "video_fps": rate,
        "video_container": video_format,
        "video_major_brand": video_major_brand,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
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
        "continuation_container": image_format,
        "continuation_codec": image_codec,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="cozy-runtime")
    parser.add_argument("--endpoint", type=Path, default=H3)
    parser.add_argument("--expected-binding-ref", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument("--expected-descriptor-digest", required=True)
    parser.add_argument("--expected-fl-plan-digest", required=True)
    parser.add_argument("--expected-ref-plan-digest", required=True)
    parser.add_argument(
        "--action",
        action="append",
        choices=tuple(PLAN_FILES),
        help="action to prove; repeat for both. Default: both actions",
    )
    parser.add_argument("--fl-input", type=Path)
    parser.add_argument("--expected-fl-input-sha256")
    parser.add_argument("--ref-input", type=Path)
    parser.add_argument("--expected-ref-input-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    selected = select_actions(args.action)
    if not args.inspect_only:
        supplied = {
            "first_last_frame_to_video": (args.fl_input, args.expected_fl_input_sha256),
            "reference_media_to_video": (args.ref_input, args.expected_ref_input_sha256),
        }
        missing = [action for action in sorted(selected) if None in supplied[action]]
        if missing:
            parser.error(f"selected actions have no exact input/digest pair: {missing}")
    return args


def main() -> int:
    args = parse_args()
    selected_actions = select_actions(args.action)
    expected_checkpoint = exact_sha256(args.expected_checkpoint, option="--expected-checkpoint")
    expected_descriptor = exact_sha256(
        args.expected_descriptor_digest, option="--expected-descriptor-digest"
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
    plan_facts = load_plan_facts(endpoint_path, expected_plans)

    snapshots: dict[str, RequestSnapshot] = {}
    if not args.inspect_only:
        supplied = {
            "first_last_frame_to_video": (args.fl_input, args.expected_fl_input_sha256),
            "reference_media_to_video": (args.ref_input, args.expected_ref_input_sha256),
        }
        for action in selected_actions:
            path, digest = supplied[action]
            assert path is not None and digest is not None
            snapshots[action] = load_request(path.expanduser().resolve(), expected_sha256=digest)

    output_root = reserve_output(args.out)
    staged_requests: dict[str, Path] = {}
    if snapshots:
        requests_root = output_root / "requests"
        requests_root.mkdir()
        staged_requests = {
            action: stage_request(requests_root / PLAN_FILES[action], snapshots[action])
            for action in selected_actions
        }
        make_requests_read_only(requests_root, list(staged_requests.values()))

    base = [args.runtime, "--dir", str(endpoint_path), "--json"]
    descriptor_check = command_json([*base, "describe", "--check"])
    descriptor_digest = verify_descriptor_digest(
        descriptor_check, expected=expected_descriptor
    )
    description = command_json([*base, "describe"])
    require_visible(selected_actions, description)
    doctor = command_json([*base, "doctor"])
    bindings = command_json([*base, "bindings"])
    verify_bindings(
        bindings,
        expected_ref=args.expected_binding_ref,
        expected_checkpoint=expected_checkpoint,
        expected_actions=selected_actions,
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "endpoint": {
            "resolved_path": str(endpoint_path),
            "descriptor_digest": descriptor_digest,
            "plans": plan_facts,
        },
        "doctor": doctor,
        "bindings": bindings,
        "expected_binding_ref": args.expected_binding_ref,
        "expected_checkpoint": expected_checkpoint,
        "selected_actions": sorted(selected_actions),
        "endpoint_observations_status": "pending-runtime-triage-join",
        "automated_status": "runtime-device-observed-and-binding-inspected",
        "human_viewed_listened_status": "pending",
    }
    if not args.inspect_only:
        results = {}
        for action in sorted(selected_actions):
            output = output_root / action
            payload = staged_requests[action]
            verify_staged_request(payload, snapshots[action])
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
                    "--strict-keys",
                ]
            )
            verify_staged_request(payload, snapshots[action])
            result, runtime_outputs = verify_result(action, document)
            results[action] = {
                "request": {
                    "sha256": snapshots[action].sha256,
                    "bytes": len(snapshots[action].raw),
                    "seed": snapshots[action].seed,
                    "staged_path": str(payload),
                },
                "runtime": document,
            }
            results[action]["media"] = probe_media(output, result, runtime_outputs)
        receipt["actions"] = results
        receipt["automated_status"] = "selected-actions-runtime-and-stored-media-green"
    receipt_path = output_root / "h3-production-probe.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
