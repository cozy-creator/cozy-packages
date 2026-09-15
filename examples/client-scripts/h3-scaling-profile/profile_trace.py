"""Diagnostic-only annotations around unchanged H3, Diffusers and NCCL calls.

The request marker crosses the existing mirrored component call. Each rank removes
it before model execution, profiles one complete forward, and writes to the same
attempt-owned output prefix. There are no extra per-operation CUDA synchronizations.
"""

from __future__ import annotations

import functools
import json
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.internal import attention_sol
from cozy_runtime.models.minimax_h3.turbo import OVERLAY_KWARG, LoRAFactors
from diffusers.models import attention_dispatch as dispatch

MARKER = "_h3_scaling_profile"
MAX_TRACE_BYTES = 256 << 20


def shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (tuple, list)):
        return [shape(item) for item in value[:4]]
    return None


class ForwardProfile:
    def __init__(self, root: Any, marker: dict[str, Any], attention_kwargs: dict[str, Any]):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.prefix = Path(marker["prefix"])
        self.step = marker["step"]
        self.layout = _ACTIVE_LAYOUT.get()
        if self.layout is None or self.layout.step != self.step:
            raise RuntimeError("profile marker does not name the current denoising step")
        self.heads = root.config.num_attention_heads
        self.head_dim = root.config.attention_head_dim
        self.rows: list[dict[str, Any]] = []
        self.last_split_sizes: list[int] | None = None
        self.stack = ExitStack()
        self.module_spans: dict[int, list[Any]] = defaultdict(list)
        self.cuda = next(root.parameters()).device.type == "cuda"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities, record_shapes=False, profile_memory=False, with_stack=False
        )
        self.profiler.__enter__()
        try:
            self.install(root, attention_kwargs.get(OVERLAY_KWARG))
        except BaseException:
            self.stack.close()
            self.profiler.__exit__(None, None, None)
            raise
        self.started = time.perf_counter()

    @contextmanager
    def region(self, name: str, **facts: Any) -> Any:
        row = {"name": name, **facts}
        started = time.perf_counter()
        with torch.profiler.record_function("h3_profile/" + name):
            try:
                yield row
            finally:
                row["cpu_wall_ms"] = (time.perf_counter() - started) * 1000
                self.rows.append(row)

    def patch(self, owner: Any, name: str, function: Any) -> None:
        original = getattr(owner, name)
        setattr(owner, name, function)
        self.stack.callback(setattr, owner, name, original)

    def timed(self, function: Any, name: str) -> Any:
        @functools.wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            tensors = [
                value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)
            ]
            with self.region(name, inputs=shape(tensors)) as row:
                result = function(*args, **kwargs)
                row["output"] = shape(result)
                return result

        return call

    def exchange(self, function: Any, name: str) -> Any:
        @functools.wraps(function)
        def call(tensor: Any, group: Any, **kwargs: Any) -> Any:
            with self.region(
                name + ".enqueue", input=shape(tensor), backend=str(dist.get_backend(group))
            ) as row:
                self.last_split_sizes = None
                wait = function(tensor, group, **kwargs)
                splits = self.last_split_sizes
                row["splits"] = splits
                row["local_tokens"] = int(kwargs.get("Q_S_LOCAL", tensor.shape[1]))

            def complete() -> Any:
                with self.region(name + ".wait", input=shape(tensor)) as done:
                    result = wait()
                    done["output"] = shape(result)
                    if name == "qkv_exchange":
                        if splits is None or len(splits) != self.world:
                            raise RuntimeError("exchange did not expose its existing size gather")
                        global_tokens = sum(splits)
                        if (
                            not self.layout.live_tokens
                            <= global_tokens
                            < self.layout.live_tokens + self.world
                        ):
                            raise RuntimeError(
                                "exchange has unexplained padding beyond the live document"
                            )
                        expected = (
                            tensor.shape[0],
                            global_tokens,
                            self.heads // self.world,
                            self.head_dim,
                        )
                        done["splits"] = splits
                        done["padding_tokens"] = global_tokens - self.layout.live_tokens
                        if tuple(result.shape) != expected:
                            raise RuntimeError(
                                f"post-exchange shape {tuple(result.shape)} differs from {expected}"
                            )
                    return result

            return complete

        return call

    def install(self, root: Any, overlay: Any) -> None:
        self.patch(
            dispatch,
            "all_to_all_single_any_qkv_async",
            self.exchange(dispatch.all_to_all_single_any_qkv_async, "qkv_exchange"),
        )
        self.patch(
            dispatch,
            "all_to_all_single_any_o_async",
            self.exchange(dispatch.all_to_all_single_any_o_async, "output_exchange"),
        )
        original_gather = dispatch.gather_size_by_comm

        @functools.wraps(original_gather)
        def gather(size: int, group: Any) -> Any:
            with self.region(
                "size_gather", local_size=size, backend=str(dist.get_backend(group))
            ) as row:
                result = original_gather(size, group)
                row["sizes"] = result
                self.last_split_sizes = list(result)
                return result

        self.patch(dispatch, "gather_size_by_comm", gather)
        self.patch(
            dispatch.funcol,
            "all_to_all_single",
            self.timed(dispatch.funcol.all_to_all_single, "collective_all_to_all_enqueue"),
        )
        self.patch(attention_sol, "_native", self.timed(attention_sol._native, "sol_native"))
        self.patch(attention_sol, "_dense", self.timed(attention_sol._dense, "sol_dense_reference"))
        for name in ("kernel_fn", "wrapped_forward_fn"):
            config = dispatch._HUB_KERNELS_REGISTRY[dispatch.AttentionBackendName._FLASH_3_HUB]
            function = getattr(config, name)
            if function is not None:
                self.patch(config, name, self.timed(function, "fa3_" + name))
        self.patch(
            LoRAFactors, "accumulate", self.timed(LoRAFactors.accumulate, "pdd_lora_accumulate")
        )
        for prefix, tree in (("base", root), ("pdd", overlay)):
            if tree is None:
                continue
            for path, module in tree.named_modules():
                kind = type(module).__name__
                linear = hasattr(module, "in_features") and hasattr(module, "out_features")
                if not linear and kind not in {"MiniMaxH3Attention", "TurboHeads"}:
                    continue
                label = prefix + "/" + path

                def before(layer: Any, args: Any, label: str = label, kind: str = kind) -> None:
                    span = self.region("module/" + label, module_type=kind, inputs=shape(args))
                    row = span.__enter__()
                    self.module_spans[id(layer)].append((span, row))

                def after(layer: Any, _args: Any, output: Any) -> None:
                    held = self.module_spans[id(layer)]
                    if held:
                        span, row = held.pop()
                        row["output"] = shape(output)
                        span.__exit__(None, None, None)

                # Append after existing PDD projection hooks, retaining their cost.
                self.stack.callback(module.register_forward_pre_hook(before).remove)
                self.stack.callback(module.register_forward_hook(after, always_call=True).remove)

    def finish(self, succeeded: bool) -> None:
        forward_wall_ms = (time.perf_counter() - self.started) * 1000
        self.stack.close()
        self.profiler.__exit__(None, None, None)
        summaries = [
            {
                "name": event.key,
                "calls": event.count,
                "cpu_self_us": event.self_cpu_time_total,
                "device_self_us": event.self_device_time_total,
            }
            for event in self.profiler.key_averages()
        ]
        report = {
            "rank": self.rank,
            "world": self.world,
            "step": self.step,
            "cuda": self.cuda,
            "device": str(torch.cuda.current_device()) if self.cuda else "cpu",
            "live_tokens": self.layout.live_tokens,
            "global_heads": self.heads,
            "expected_post_exchange_heads": self.heads // self.world,
            "head_dim": self.head_dim,
            "forward_succeeded": succeeded,
            "forward_wall_ms": forward_wall_ms,
            "cuda_self_time_sum_us": sum(row["device_self_us"] for row in summaries),
            "events": sorted(summaries, key=lambda row: row["device_self_us"], reverse=True),
            "spans": self.rows,
            "interpretation": (
                "CPU waits, inclusive module spans and CUDA activity overlap. "
                "Do not add them as independent elapsed time; inspect the raw trace."
            ),
        }
        metadata = Path(f"{self.prefix}.rank-{self.rank}.json")
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Metadata survives an oversized/failed trace export. One forward only.
        trace = Path(f"{self.prefix}.rank-{self.rank}.trace.json")
        self.profiler.export_chrome_trace(str(trace))
        report["trace_bytes"] = trace.stat().st_size
        report["trace_within_bound"] = report["trace_bytes"] <= MAX_TRACE_BYTES
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def install_profile_hooks(root: Any) -> None:
    """Installed only on this diagnostic model, on both leader and follower replicas."""

    def before(module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        values = kwargs.get("attention_kwargs")
        if not isinstance(values, dict) or MARKER not in values:
            return None
        values = dict(values)
        marker = values.pop(MARKER)
        kwargs = {**kwargs, "attention_kwargs": values}
        layout = _ACTIVE_LAYOUT.get()
        if layout is not None and layout.step == marker["step"]:
            if getattr(module, "_h3_profile_active", None) is not None:
                raise RuntimeError("nested H3 diagnostic profile")
            module._h3_profile_active = ForwardProfile(module, marker, values)
        return args, kwargs

    def after(module: Any, _args: Any, output: Any) -> None:
        active = getattr(module, "_h3_profile_active", None)
        if active is not None:
            del module._h3_profile_active
            active.finish(output is not None)

    root.register_forward_pre_hook(before, with_kwargs=True, prepend=True)
    root.register_forward_hook(after, always_call=True)
