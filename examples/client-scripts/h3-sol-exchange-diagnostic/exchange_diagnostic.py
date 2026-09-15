"""Temporary, process-local h3a093 instrumentation; never an attention backend.

Installation occurs in Model.load on every rank without touching CUDA. The first
post-fill parallel call records metadata and synchronizes before unchanged Sol
execution. This run is diagnostic and its latency is not a benchmark.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
from collections.abc import Callable
from typing import Any

import torch
from cozy_runtime import _build_provenance
from cozy_runtime.internal import attention_sol

_INSTALLED = False
_RUNTIME = "d140609bf5fd07f8c67eaf7e87de433f5177f94f"
_PARAMETERS = ("query", "key", "value", "scale", "layout", "site", "group")


def _emit(phase: str, **fields: Any) -> None:
    print(
        json.dumps({"event": "h3a093.first_exchange", "phase": phase, **fields}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
    }


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    reported = False

    @functools.wraps(original)
    def exchange(
        query: Any,
        key: Any,
        value: Any,
        scale: Any,
        layout: Any,
        site: Any,
        group: Any,
    ) -> Any:
        nonlocal reported
        if reported:
            return original(query, key, value, scale, layout, site, group)
        reported = True
        identity = {
            "pid": os.getpid(),
            "rank": torch.distributed.get_rank(group),
            "world": torch.distributed.get_world_size(group),
            "site": site.path,
            "step": layout.step,
            "live_tokens": layout.live_tokens,
            "qkv": {
                name: _tensor_metadata(t)
                for name, t in zip("qkv", (query, key, value), strict=True)
            },
        }
        _emit("entered", **identity)
        facts: dict[str, Any] = {}
        try:
            facts["current_device"] = torch.cuda.current_device()
            facts["allocated_bytes"] = torch.cuda.memory_allocated(query.device)
            facts["reserved_bytes"] = torch.cuda.memory_reserved(query.device)
            free, total = torch.cuda.mem_get_info(query.device)
            facts["driver_free_bytes"], facts["driver_total_bytes"] = free, total
        except Exception as exc:
            _emit("metadata_failed", **identity, **facts, error_type=type(exc).__name__)
            raise
        _emit("before_sync", **identity, **facts)
        try:
            torch.cuda.synchronize(query.device)
        except Exception as exc:
            _emit("sync_failed", **identity, error_type=type(exc).__name__)
            raise
        _emit("sync_passed", **identity, current_device=torch.cuda.current_device())
        try:
            result = original(query, key, value, scale, layout, site, group)
        except Exception as exc:
            _emit("exchange_path_failed", **identity, error_type=type(exc).__name__)
            raise
        _emit("exchange_path_returned", **identity)
        return result

    return exchange


def install() -> None:
    """Only this explicitly captured diagnostic model installs the temporary wrapper."""
    global _INSTALLED
    if _INSTALLED:
        return
    if _build_provenance.COMMIT != _RUNTIME:
        raise RuntimeError("h3a093 diagnostic requires the reviewed d140 Runtime cohort")
    original = attention_sol._execute_parallel
    if (
        original.__module__ != attention_sol.__name__
        or tuple(inspect.signature(original).parameters) != _PARAMETERS
    ):
        raise RuntimeError("h3a093 diagnostic requires the unchanged Sol parallel seam")
    attention_sol._execute_parallel = _wrap(original)
    _INSTALLED = True
