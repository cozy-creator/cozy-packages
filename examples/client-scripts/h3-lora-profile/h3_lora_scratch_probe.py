"""Compare exact Runtime scratch implementations on one captured H3 projection."""

from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import textwrap
from pathlib import Path
from typing import Any

import cozy_runtime.internal.lora as lora
import h3_lora_merge_probe as support
import msgspec
import torch
from cozy_runtime.author import App, Context, Outputs, Telemetry
from h3_lora_merge_probe import ProbeModel as ProbeModel

from h3 import FirstLastFrameToVideoInput, KeyframeAssets, fl2va

app = App()
ROWS = 8209
LIMIT = 1 << 30


def load_candidate() -> Any:
    source = Path(__file__).with_name("candidate_apply.py").read_text()
    facts = json.loads(Path(__file__).with_name("scratch_sources.json").read_text())
    if hashlib.sha256(source.encode()).hexdigest() != facts["candidate_apply_sha256"]:
        raise ValueError("candidate source digest changed")
    old = textwrap.dedent(inspect.getsource(lora.LinearAdapters._apply))
    if hashlib.sha256(old.encode()).hexdigest() != facts["old_apply_sha256"]:
        raise ValueError("installed Runtime is not the reviewed old implementation")
    namespace = dict(vars(lora))
    exec(compile(source, "candidate_apply.py", "exec"), namespace)
    return namespace["_apply"]


class Capture(support.Capture):
    def observe(self, leaf: Any, args: Any, output: Any) -> None:
        if self.x is not None:
            return
        x = args[0]
        if not x.is_contiguous() or not output.is_contiguous():
            raise ValueError("scratch probe requires contiguous actual inputs")
        if x.numel() // x.shape[-1] < ROWS:
            raise ValueError("scratch probe requires two full tiles and a partial tail")
        self.report["actual_input_shape"] = list(x.shape)
        self.x = x.reshape(-1, x.shape[-1])[:ROWS].detach().clone()
        self.observed = output.reshape(-1, output.shape[-1])[:ROWS].detach().clone()

    def run(self, leaf: Any, terms: Any) -> None:
        candidate = load_candidate()
        if torch.get_float32_matmul_precision() != "highest":
            raise ValueError("scratch probe requires highest FP32 matmul precision")
        hook = next(iter(leaf._forward_hooks.values()))
        if not isinstance(hook, lora._Hook):
            raise ValueError("original Runtime hook must remain first")
        if any(t.a.dtype != torch.float32 or t.b.dtype != torch.float32 for t in terms):
            raise ValueError("expected verified FP32 factors")
        sources = [leaf.data, leaf.scale, self.x, *[f for t in terms for f in (t.a, t.b)]]
        before_hashes = [support.fingerprint(x) for x in sources]
        rank = max(t.a.shape[0] for t in terms)
        scratch = lora.TILE_ROWS * (leaf.in_features + rank + 2 * leaf.out_features) * 4
        held = sum(t.numel() * t.element_size() for t in (self.x, self.observed))
        # Base plus both parity outputs; fixed scratch and bounded error tiles.
        required = (
            held + 3 * self.observed.numel() * self.observed.element_size() + scratch + (32 << 20)
        )
        if required > LIMIT:
            raise ValueError(f"scratch comparison requires {required} bytes, limit {LIMIT}")
        device = self.x.device
        prior_peak = torch.cuda.max_memory_allocated(device)
        start_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = leaf._cozy_forward(leaf, self.x)
        stacks = {
            "actual": tuple(terms),
            "ordered_negative_stack": (
                *terms,
                *[lora._Term(t.a, t.b, -0.375 * t.scale) for t in terms],
            ),
        }
        self.report.update(
            cases={},
            fixed_tile_rows=lora.TILE_ROWS,
            sample_rows=ROWS,
            calculated_bound_bytes=required,
            capture_held_bytes=held,
            peak_before_probe_bytes=prior_peak,
        )
        for name, selected in stacks.items():
            self.cancel()
            attempt = lora._Attempt({hook.site: selected}, self.cancel)
            token = hook.binding._active.set(attempt)
            try:

                def call(method: Any) -> Any:
                    return method(hook.binding, hook.site, leaf, (self.x,), {}, base.clone())

                old = call(lora.LinearAdapters._apply)
                new = call(candidate)
                equal = torch.equal(old, new)
                record = {
                    "byte_equal": equal,
                    "difference": support.error(new, old),
                    "effective_coefficients": [t.scale for t in selected],
                }
                if name == "actual":
                    record["old_replay_vs_full_capture"] = support.error(old, self.observed)
                del old, new
                if not equal:
                    raise ValueError(f"scratch candidate changed {name} output")
                for _ in range(3):
                    for method in (lora.LinearAdapters._apply, candidate):
                        value = call(method)
                        del value
                torch.cuda.synchronize()
                samples: dict[str, list[float]] = {"old": [], "new": []}
                for index in range(12):
                    order = (("old", lora.LinearAdapters._apply), ("new", candidate))
                    if index % 2:
                        order = tuple(reversed(order))
                    for label, method in order:
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                        value = call(method)
                        end.record()
                        end.synchronize()
                        samples[label].append(begin.elapsed_time(end))
                        del value
                record["cuda_ms"] = samples
                record["median_cuda_ms"] = {k: statistics.median(v) for k, v in samples.items()}
                record["speedup"] = (
                    record["median_cuda_ms"]["old"] / record["median_cuda_ms"]["new"]
                )
                self.report["cases"][name] = record
            finally:
                hook.binding._active.reset(token)
        peak = torch.cuda.max_memory_allocated(device)
        observed = max(0, peak - start_allocated) + held
        self.report.update(
            allocator_peak_bytes=peak,
            observed_additional_peak_bytes=observed,
            original_bytes_unchanged=before_hashes == [support.fingerprint(x) for x in sources],
            matmul_precision=torch.get_float32_matmul_precision(),
        )
        if observed > LIMIT or not self.report["original_bytes_unchanged"]:
            raise ValueError("scratch probe violated its memory or immutable-source bound")


@app.entrypoint
def probe(
    ctx: Context,
    payload: support.Input,
    assets: KeyframeAssets,
    model: ProbeModel,
    out: Outputs,
    tel: Telemetry,
) -> support.Result:
    capture = Capture(payload, ctx.raise_if_cancelled)
    token = support._ACTIVE.set(capture)
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
        except support.Captured:
            pass
        else:
            raise RuntimeError("scratch probe unexpectedly generated a complete video")
    finally:
        support._ACTIVE.reset(token)
    capture.report.update(
        input=msgspec.to_builtins(payload),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        sources=json.loads(Path(__file__).with_name("scratch_sources.json").read_text()),
        scope=(
            "Exact old/new Runtime methods; identical cloned encoded-base output; "
            "Actual factors plus an ordered negative-strength stack; one denoise step only. "
            "Timings include output cloning. Allocator peaks exclude driver/Varena allocations."
        ),
    )
    return support.Result(
        out.save_bytes(
            json.dumps(capture.report, sort_keys=True).encode(), media_type="application/json"
        )
    )
