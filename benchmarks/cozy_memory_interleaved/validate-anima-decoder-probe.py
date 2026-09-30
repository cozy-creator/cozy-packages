"""Strict local probe consumer; missing live receipt/grant/scope data never becomes zero."""

from __future__ import annotations

import math
import re
from typing import Any

from cozy_runtime.internal import output_budget


def validate(
    report: dict[str, Any],
    inventory: dict[str, int],
    interface: dict[str, Any],
    accepted_spec: dict[str, Any],
    reply: dict[str, Any],
    runtime: dict[str, Any],
    scope: dict[str, Any],
) -> None:
    for field, value in {
        "producer_complete": True,
        "runtime_gate_pending": True,
        "scored": False,
        "raw_bits_exact": True,
        "rgb_exact": True,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
    }.items():
        if type(report.get(field)) is not bool or report[field] is not value:
            raise ValueError(f"missing/incorrect {field}")
    if (
        type(report.get("source_profile_decode_calls")) is not int
        or report["source_profile_decode_calls"] != 2
        or type(report.get("profile_regions")) is not int
        or report["profile_regions"] != 1
        or report.get("tile") != 256
        or report.get("stride") != 192
    ):
        raise ValueError("profile inventory changed")
    modes = report.get("modes", [])
    if [mode.get("mode") for mode in modes] != ["scalar", "broadcast"]:
        raise ValueError("mode order/inventory changed")
    for mode in modes:
        if any(
            type(mode.get(key)) is not int or mode[key] < 0
            for key in ["fast_blends", "scalar_blends"]
        ):
            raise ValueError("missing blend path truth")
        if mode["fast_blends"] + mode["scalar_blends"] != 60:
            raise ValueError("full1024 tile/blend inventory changed")
        for field in ["decode_host_wall_ms", "decode_cuda_event_ms"]:
            value = mode.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("missing decoder completion/timing boundary")
        for field in ["raw_bits_sha256", "raw_artifact_sha256", "rgb_sha256"]:
            if not isinstance(mode.get(field), str) or not re.fullmatch(
                "[0-9a-f]{64}", mode[field]
            ):
                raise ValueError("missing exact tensor/RGB digest")
    if (
        modes[0]["raw_bits_sha256"] != modes[1]["raw_bits_sha256"]
        or modes[0]["rgb_sha256"] != modes[1]["rgb_sha256"]
    ):
        raise ValueError("tensor/RGB digest comparison failed")
    if modes[0]["rgb_sha256"] != report.get("baseline_rgb_sha256"):
        raise ValueError("ordinary baseline RGB comparison failed")
    for field in [
        "argument_sha256",
        "argument_bits_sha256",
        "baseline_raw_internal_sha256",
        "baseline_raw_bits_sha256",
    ]:
        if not isinstance(report.get(field), str) or not re.fullmatch(
            "[0-9a-f]{64}", report[field]
        ):
            raise ValueError("missing capture provenance digest")
    if report["baseline_raw_bits_sha256"] != modes[0]["raw_bits_sha256"]:
        raise ValueError("captured baseline raw comparison failed")
    if not any(row.get("device_total_us", 0) > 0 for row in report.get("operators", [])):
        raise ValueError("incomplete CUDA profile")
    bounds = {
        field["name"]: field["asset_bound"]["max_bytes"]
        for field in interface["entrypoints"][0]["result"]["fields"]
        if "asset_bound" in field
    }
    if len(interface["entrypoints"]) != 1 or sum(bounds.values()) != 234 << 20:
        raise ValueError("actual descriptor inventory/budget changed")
    if set(inventory) != set(bounds):
        raise ValueError("missing/extra artifact; exactly2raw/2RGB/one trace required")
    if any(
        type(size) is not int or size <= 0 or size > bounds[name]
        for name, size in inventory.items()
    ):
        raise ValueError("actual artifact size invalid")
    expected_slots = sorted(
        (field["asset_bound"]["max_bytes"], field["asset_bound"]["media_types"][0])
        for field in interface["entrypoints"][0]["result"]["fields"]
        if "asset_bound" in field
    )
    actual_slots = sorted(
        (slot.get("max_bytes"), slot.get("mime_type")) for slot in accepted_spec.get("outputs", [])
    )
    if actual_slots != expected_slots:
        raise ValueError("actual accepted output bindings differ from descriptor")
    effective = output_budget.effective(accepted_spec, reply)
    if effective < sum(bounds.values()) or sum(inventory.values()) > effective:
        raise ValueError("actual accepted output grant too small")
    if type(runtime.get("vae_invokes")) is not int or runtime["vae_invokes"] != 3:
        raise ValueError("baseline+two profile invocations not established")
    for field in ["allocator_ooms", "allocator_retries"]:
        if type(runtime.get(field)) is not int or runtime[field] != 0:
            raise ValueError("OOM/retry/incomplete invocation inventory")
    if runtime.get("ledger_closed") is not True:
        raise ValueError("ledger not closed")
    for field in ["pid", "start_ticks", "memory_high_bytes", "memory_max_bytes", "cpu_cores"]:
        if field not in scope or type(scope[field]) not in (int, float) or scope[field] <= 0:
            raise ValueError("actual identity/scope budget missing")
    if scope.get("nice") != 19 or scope.get("swap_bytes") != 0:
        raise ValueError("scope caps differ")
