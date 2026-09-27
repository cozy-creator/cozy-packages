"""Read-only actual-layer preparation benchmark; no serving weights or cache are mutated."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
import time
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

import msgspec
import torch
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    Outputs,
    Telemetry,
    uses_components,
)
from merge_arithmetic import MergeAdapter, merge_rowwise_fp8

from h3 import FirstLastFrameToVideoInput, H3Model, KeyframeAssets, fl2va

app = App()
_ACTIVE: ContextVar[Capture | None] = ContextVar("h3_lora_merge_probe", default=None)
LIMIT = 1 << 30
ROWS = 4096


class Input(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    duration_s: Literal[5, 15]
    seed: int = 7381
    block: Literal[0, 25, 49] = 0


class Result(msgspec.Struct):
    measurements: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))
    ]


class Captured(Exception):
    pass


def fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    for rows in value.reshape(-1).split(1 << 20):
        digest.update(rows.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def error(actual: Any, expected: Any) -> dict[str, float]:
    squared = reference = maximum = 0.0
    for a, b in zip(actual.split(128), expected.split(128), strict=True):
        delta = a.double() - b.double()
        squared += float(delta.double().square().sum())
        reference += float(b.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {"relative_l2": (squared / max(reference, 1e-30)) ** 0.5, "max_abs": maximum}


def timed(call: Any, repeats: int = 3) -> tuple[Any, dict[str, Any]]:
    samples = []
    result = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        result = call()
        end.record()
        end.synchronize()
        samples.append(
            {"wall_ms": (time.perf_counter() - started) * 1000, "cuda_ms": begin.elapsed_time(end)}
        )
    return result, {
        "samples": samples,
        "median_cuda_ms": statistics.median(x["cuda_ms"] for x in samples),
    }


class Capture:
    def __init__(self, payload: Input, cancel: Any) -> None:
        self.payload, self.cancel = payload, cancel
        self.x: Any = None
        self.observed: Any = None
        self.report: dict[str, Any] = {}

    def observe(self, leaf: Any, args: Any, output: Any) -> None:
        if self.x is not None:
            return
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_contiguous() or not output.is_contiguous():
            raise ValueError("merge probe requires the active LoRA's contiguous floating input")
        self.report["actual_input_shape"] = list(x.shape)
        self.x = x.reshape(-1, x.shape[-1])[:ROWS].detach().clone()
        self.observed = output.reshape(-1, output.shape[-1])[:ROWS].detach().clone()

    def run(self, leaf: Any, terms: Any) -> None:
        if self.x is None:
            raise ValueError("selected leaf was not executed")
        if torch.get_float32_matmul_precision() != "highest":
            raise ValueError("merge prototype requires highest FP32 matmul precision")
        if any(t.a.dtype != torch.float32 or t.b.dtype != torch.float32 for t in terms):
            raise ValueError("the GEMM control requires Runtime's FP32 factor copies")
        # Runtime already incorporated strength*alpha/rank into each term.scale.
        adapters = [MergeAdapter(t.a, t.b, strength=t.scale) for t in terms]
        data, scale = leaf.data, leaf.scale
        sources = (data, scale, *[f for t in terms for f in (t.a, t.b)])
        originals = [fingerprint(x) for x in sources]
        self.report["original_part_sha256"] = originals
        rank = max(t.a.shape[0] for t in terms)
        incoming, outgoing = data.shape[1], data.shape[0]
        held = sum(x.numel() * x.element_size() for x in (self.x, self.observed))
        # One candidate at a time, including the larger BF16 weight control,
        # two live timed outputs, original observations and residual tile scratch.
        required = held + 2 * data.numel() + 4 * scale.numel() + 4 * self.observed.numel()
        required += max(ROWS * (incoming + rank + 2 * outgoing) * 4, 64 << 20)
        if required > LIMIT:
            raise ValueError(f"prototype requires {required} bytes, limit {LIMIT}")
        free, _ = torch.cuda.mem_get_info(data.device)
        reusable = torch.cuda.memory_reserved(data.device) - torch.cuda.memory_allocated(
            data.device
        )
        if required > free + reusable:
            raise ValueError("prototype scratch does not fit current device headroom")
        self.report.update(
            additional_budget_bytes=required,
            weight_shape=list(data.shape),
            encoding=leaf.encoding,
            terms=[
                {"rank": t.a.shape[0], "dtype": str(t.a.dtype), "effective_coefficient": t.scale}
                for t in terms
            ],
            original_scale_unique=int(torch.unique(scale).numel()),
        )
        peaks = []
        for start in range(0, outgoing, 128):
            peaks.extend(data[start : start + 128].float().abs().amax(dim=1).cpu().tolist())
        self.report["original_row_max_codes"] = {
            "min": min(peaks),
            "max": max(peaks),
            "rows_at448": sum(x == 448 for x in peaks),
        }
        indices = torch.linspace(0, outgoing - 1, 16, device=data.device).long().unique()
        x = self.x[:16].double().cpu()
        w = (
            data.index_select(0, indices).double().cpu()
            * scale.reshape(-1).index_select(0, indices).double().cpu()[:, None]
        )
        delta = torch.zeros_like(w)
        oracle = x @ w.T
        for t in terms:
            a = t.a.double().cpu()
            b = t.b.index_select(0, indices).double().cpu()
            delta += (b @ a) * t.scale
            oracle += ((x @ a.T) @ b.T) * t.scale
        update_norm = float(torch.linalg.vector_norm(delta))
        self.report["fp64_oracle_rows"] = indices.cpu().tolist()
        self.report["sampled_update_norm"] = update_norm
        self.report["sampled_update_over_base_norm"] = update_norm / max(
            float(torch.linalg.vector_norm(w)), 1e-30
        )
        previous_peak = torch.cuda.max_memory_allocated(data.device)
        torch.cuda.reset_peak_memory_stats(data.device)
        replay_before = torch.cuda.memory_allocated(data.device)
        replay, timing = timed(lambda: leaf(self.x))
        replay_peak = torch.cuda.max_memory_allocated(data.device)
        self.report["residual_replay"] = {
            "forward": timing,
            "against_captured": error(replay, self.observed),
            "allocator_peak_bytes": replay_peak,
            "allocator_peak_delta_bytes": max(0, replay_peak - replay_before),
        }
        self.observed = replay
        del replay
        self.report["capture_held_bytes"] = held
        self.report["peak_before_microbench_bytes"] = previous_peak
        peaks_observed = [previous_peak, replay_peak]
        self.report["candidates"] = {}
        cases = [
            (method, policy)
            for policy in ("preserve", "grow", "recalibrate")
            for method in ("rank_ordered", "fp32_gemm")
        ]
        cases += [
            ("bf16_merged", "none"),
            ("codec_zero", "preserve"),
            ("codec_zero", "recalibrate"),
        ]
        for method, policy in cases:
            self.cancel()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated(data.device)
            torch.cuda.reset_peak_memory_stats(data.device)
            candidate = torch.empty_like(
                data, dtype=torch.bfloat16 if method == "bf16_merged" else data.dtype
            )
            scales = torch.empty_like(scale)

            def merge(
                method: str = method,
                policy: str = policy,
                candidate: Any = candidate,
                scales: Any = scales,
            ) -> None:
                if method == "rank_ordered":
                    tiles = merge_rowwise_fp8(
                        data,
                        scale,
                        adapters,
                        encoding_spec=leaf.encoding,
                        scale_policy=policy,
                        max_scratch_bytes=64 << 20,
                        tile_rows=128,
                    )
                    assert tiles is not None
                    for tile in tiles:
                        self.cancel()
                        candidate[tile.start : tile.start + len(tile.data)].copy_(tile.data)
                        scales[tile.start : tile.start + len(tile.scale)].copy_(tile.scale)
                    return
                for start in range(0, outgoing, 128):
                    self.cancel()
                    end = min(start + 128, outgoing)
                    merged = data[start:end].float() * scale[start:end].reshape(-1, 1)
                    if method != "codec_zero":
                        for t in terms:
                            merged.add_((t.b[start:end] @ t.a).mul_(t.scale))
                    if not bool(torch.isfinite(merged).all()):
                        raise ValueError("nonfinite merged weight")
                    if method == "bf16_merged":
                        candidate[start:end].copy_(merged)
                        if not bool(torch.isfinite(candidate[start:end]).all()):
                            raise ValueError("BF16 merged weight overflowed")
                        continue
                    peak = merged.abs().amax(dim=1)
                    old = scale[start:end].reshape(-1)
                    if policy == "preserve":
                        overflow = (peak > old * 448) | (old <= 0)
                        if bool(overflow.any()):
                            raise ValueError(
                                f"original grid overflow at tile{start}: {int(overflow.sum())}rows"
                            )
                        row_scale = old
                    elif policy == "grow":
                        if bool((old <= 0).any()):
                            raise ValueError("grow requires positive original scales")
                        row_scale = torch.maximum(old, peak / 448)
                    else:
                        row_scale = (peak / 448).clamp_min_(torch.finfo(torch.float32).tiny)
                        row_scale[peak == 0] = 1
                    codes = (merged / row_scale[:, None]).clamp_(-448, 448).to(data.dtype)
                    candidate[start:end].copy_(codes)
                    scales[start:end].copy_(row_scale.reshape_as(scales[start:end]))
                    # Match the reference helper's per-tile error observations.
                    decoded = codes.float() * row_scale[:, None]
                    if not bool(torch.isfinite(decoded).all()):
                        raise ValueError("reconstructed FP8 weight overflowed")
                    difference = decoded - merged
                    float(difference.double().square().sum())
                    float(merged.double().square().sum())
                    float(difference.abs().max())

            record: dict[str, Any] = {"method": method, "scale_policy": policy}
            try:
                _, record["prepare"] = timed(merge, repeats=1)
                if method == "bf16_merged":
                    output, record["forward"] = timed(
                        lambda candidate=candidate: torch.nn.functional.linear(self.x, candidate)
                    )
                    decoded = candidate.index_select(0, indices).double().cpu()
                    record["activation_route"] = (
                        "BF16 input/weight GEMM; changes base W8A8 activation quantization"
                    )
                else:
                    proxy = SimpleNamespace(data=candidate, scale=scales, out_dtype=leaf.out_dtype)
                    output, record["forward"] = timed(
                        lambda proxy=proxy: leaf._cozy_forward(proxy, self.x)
                    )
                    del proxy
                    decoded = (
                        candidate.index_select(0, indices).double().cpu()
                        * scales.reshape(-1).index_select(0, indices).double().cpu()[:, None]
                    )
                    record["changed_scale_rows"] = int(
                        (scales.reshape(-1) != scale.reshape(-1)).sum()
                    )
                    record["activation_route"] = (
                        "same Runtime per-token FP8 activation quantization"
                    )
                ideal = w if method == "codec_zero" else w + delta
                norm = float(torch.linalg.vector_norm(decoded - ideal))
                record.update(
                    quantization_error_norm=norm,
                    error_over_update_norm=norm / max(update_norm, 1e-30),
                    against_same_shape_residual=error(output, self.observed),
                    against_fp64_oracle=error(output[:16].index_select(1, indices).cpu(), oracle),
                    candidate_data_sha256=fingerprint(candidate),
                )
                if method != "bf16_merged":
                    record["candidate_scale_sha256"] = fingerprint(scales)
                if method == "codec_zero":
                    record["same_original_codes"] = fingerprint(candidate) == originals[0]
                del output
            except ValueError as exc:
                record["refusal"] = str(exc)
            peak = torch.cuda.max_memory_allocated(data.device)
            extra = held + max(0, peak - before)
            record.update(
                allocator_peak_bytes=peak,
                allocator_peak_delta_bytes=max(0, peak - before),
                observed_additional_peak_bytes=extra,
                within_1gib_bound=extra <= LIMIT,
            )
            if extra > LIMIT:
                raise ValueError("observed prototype allocation exceeded its 1GiB bound")
            peaks_observed.append(peak)
            self.report["candidates"][method + "/" + policy] = record
            del merge, candidate, scales
        self.report["observed_whole_probe_peak_bytes"] = max(peaks_observed)
        self.report["allocator_measurement_scope"] = (
            "PyTorch CUDA allocated bytes, excluding driver allocations/cache; peak counters "
            "reset per private candidate, with preceding whole-step peak retained above. "
            "Candidate deltas plus held capture buffers are checked against1GiB."
        )
        self.report["preparation_timing_scope"] = (
            "Rank reference includes helper finite-factor validation; GEMM trusts Runtime's "
            "already-validated FP32 factors. Both include per-row error observations."
        )
        zero = [MergeAdapter(t.a, t.b, strength=0) for t in terms]
        self.report["zero_strength_grafts_original"] = (
            merge_rowwise_fp8(
                data, scale, zero, encoding_spec=leaf.encoding, scale_policy="preserve"
            )
            is None
        )
        self.report["original_bytes_unchanged"] = originals == [fingerprint(x) for x in sources]
        if not self.report["original_bytes_unchanged"]:
            raise ValueError("probe changed source bytes")


class ProbeModel(H3Model, encoded_leaves="accept", fusion="accept"):
    @uses_components("fl2va_dit")
    def sample_fl2va(self, state: Any, *, on_step: Any, cancel: Any, checks: Any) -> Any:
        capture = _ACTIVE.get()
        if capture is None or torch.cuda.device_count() != 1:
            raise ValueError("probe requires one GPU and an active capture")
        root = self.pipe.components["fl2va_dit"]
        target = f"transformer_blocks.{capture.payload.block}.attn.to_q"
        leaf = root.get_submodule(target)
        if (
            getattr(leaf, "provider", None) != "cozy.fp8-rowwise.native-leaf/1"
            or leaf.bias is not None
        ):
            raise ValueError("probe supports one unbiased Runtime rowwise encoded Q projection")
        hooks = [
            h
            for h in leaf._forward_hooks.values()
            if type(h).__module__ == "cozy_runtime.internal.lora" and type(h).__name__ == "_Hook"
        ]
        if len(hooks) != 1:
            raise ValueError("probe requires one Runtime-owned generic LoRA hook")
        active = hooks[0].binding._active.get()
        terms = active.terms.get(hooks[0].site, ()) if active is not None else ()
        if not terms:
            raise ValueError("probe requires active verified nonzero LoRA factors")
        free, _ = torch.cuda.mem_get_info(leaf.data.device)
        if free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < LIMIT:
            raise ValueError("probe needs 1GiB of measured headroom before input capture")
        capture.report.update(target=target, base_snapshot=hooks[0].binding._snapshots["fl2va_dit"])
        handle = leaf.register_forward_hook(capture.observe)

        def step(index: int) -> None:
            on_step(index)
            if index == 0:
                checks.settle()
                with torch.autocast(device_type="cuda", enabled=False):
                    capture.run(leaf, terms)
                raise Captured()

        checks.component("fl2va_dit", root)
        try:
            with checks.forwards(root, "fl2va_dit"):
                return self.pipe.denoise("fl2va", state, on_step=step, cancel=cancel, checks=checks)
        finally:
            handle.remove()


@app.entrypoint
def probe(
    ctx: Context,
    payload: Input,
    assets: KeyframeAssets,
    model: ProbeModel,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    capture = Capture(payload, ctx.raise_if_cancelled)
    token = _ACTIVE.set(capture)
    try:
        try:
            fl2va(
                ctx,
                FirstLastFrameToVideoInput(
                    prompt=payload.prompt,
                    seed=payload.seed,
                    duration_s=payload.duration_s,
                    steps=30,
                ),
                assets,
                model,
                out,
                tel,
            )
        except Captured:
            pass
        else:
            raise RuntimeError("probe unexpectedly generated a full video")
    finally:
        _ACTIVE.reset(token)
    capture.report.update(
        input=msgspec.to_builtins(payload),
        sample_rows=ROWS,
        scope=(
            "One actual projection; separate buffers and same-shape residual replay reference. "
            "Replay-versus-captured measures row-shape effects; FP64 oracle omits base "
            "activation quantization."
        ),
        torch=torch.__version__,
        runtime_version=importlib.metadata.version("cozy-runtime"),
        gpu=torch.cuda.get_device_name(),
        sources=json.loads(Path(__file__).with_name("merge_sources.json").read_text()),
    )
    return Result(
        out.save_bytes(
            json.dumps(capture.report, sort_keys=True).encode(), media_type="application/json"
        )
    )
