"""Compare eager/compiled torch.profiler Chrome traces and inspect compiler sources.

Usage: uv run --no-project python scripts/h3-compile-trace-summary.py \
    eager.json compiled.json --out comparison --generated-source torch_compile_debug

Profile the same work in both inputs. Measure inference latency separately without the
profiler. Chrome trace timestamps and durations are microseconds regardless of its
displayTimeUnit. CPU operator durations are inclusive; kernel durations can overlap.
The script reads generated code as text/AST and never imports or executes it.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def union_us(intervals: list[tuple[float, float]]) -> float:
    """Duration covered by at least one interval, without counting overlap twice."""
    end = -math.inf
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def aggregate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for event in events:
        name = event["name"]
        row = rows.setdefault(name, {"name": name, "calls": 0, "total_us": 0.0})
        row["calls"] += 1
        row["total_us"] += event["dur"]
    return sorted(rows.values(), key=lambda row: (-row["total_us"], row["name"]))


def summarize(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        document = json.load(source)
    raw = document if isinstance(document, list) else document["traceEvents"]
    events = []
    for event in raw:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        if not all(math.isfinite(float(event[key])) for key in ("ts", "dur")):
            raise ValueError(f"non-finite timestamp/duration in {path}")
        if event["dur"] < 0:
            raise ValueError(f"negative duration in {path}")
        events.append(event)

    def category(*names: str) -> list[dict[str, Any]]:
        return [e for e in events if set(str(e.get("cat", "")).split(",")) & set(names)]

    kernels = category("kernel", "gpu_kernel")
    cpu = category("cpu_op")
    runtime = category("cuda_runtime", "cuda_driver")
    devices: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for kernel in kernels:
        device = str(kernel.get("args", {}).get("device", f"pid:{kernel.get('pid')}"))
        devices[device].append((kernel["ts"], kernel["ts"] + kernel["dur"]))
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "event_count": len(raw),
        "complete_event_count": len(events),
        "categories": dict(Counter(str(e.get("cat", "")) for e in events)),
        "kernels": aggregate(kernels),
        "kernel_calls": len(kernels),
        "kernel_summed_us": sum(e["dur"] for e in kernels),
        "kernel_devices": {
            device: {
                "union_us": union_us(intervals),
                "first_start_us": min(a for a, _ in intervals),
                "last_end_us": max(b for _, b in intervals),
                "span_us": max(b for _, b in intervals) - min(a for a, _ in intervals),
            }
            for device, intervals in sorted(devices.items())
        },
        "cpu_ops": aggregate(cpu),
        "cpu_op_calls": len(cpu),
        "cuda_api": aggregate(runtime),
        "cuda_launch_api_calls": sum("launch" in e["name"].lower() for e in runtime),
        "gpu_copies_and_memsets": aggregate(category("gpu_memcpy", "gpu_memset")),
        "annotations": aggregate(category("user_annotation")),
        "warnings": [] if kernels else ["No CUDA kernel events: this is not GPU timing evidence."],
    }


def difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a = {row["name"]: row for row in left}
    b = {row["name"]: row for row in right}
    rows = []
    for name in a.keys() | b.keys():
        eager = a.get(name, {"calls": 0, "total_us": 0.0})
        compiled = b.get(name, {"calls": 0, "total_us": 0.0})
        rows.append(
            {
                "name": name,
                "eager_calls": eager["calls"],
                "compiled_calls": compiled["calls"],
                "eager_us": eager["total_us"],
                "compiled_us": compiled["total_us"],
                "eager_minus_compiled_us": eager["total_us"] - compiled["total_us"],
            }
        )
    return sorted(rows, key=lambda row: (-abs(row["eager_minus_compiled_us"]), row["name"]))


def generated_sources(paths: list[Path]) -> list[dict[str, Any]]:
    """Retain exact lowering evidence; a source call count is not an execution count."""
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob("*.py"))
            files.update(p.resolve() for p in path.rglob("ir_*fusion.txt"))
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise FileNotFoundError(path)
    results = []
    for path in sorted(files):
        source = path.read_text()
        origins = [
            {"line": i, "text": line.strip()}
            for i, line in enumerate(source.splitlines(), 1)
            if any(
                marker in line
                for marker in (
                    "Topologically Sorted Source Nodes",
                    "Original ATen",
                    "Source node to ATen",
                )
            )
        ]
        calls: Counter[str] = Counter()
        parse_error = None
        if path.suffix == ".py":
            try:
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Call):
                        name = ast.unparse(node.func)
                        if name.startswith(("extern_kernels.", "torch.ops.", "aten.")):
                            calls[name] += 1
            except SyntaxError as error:
                parse_error = str(error)
        results.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bytes": path.stat().st_size,
                "kernel_names": sorted(set(re.findall(r"\btriton_[A-Za-z0-9_]+", source))),
                "static_operator_call_sites": dict(sorted(calls.items())),
                "origin_comments": origins,
                "parse_error": parse_error,
            }
        )
    return results


def markdown(result: dict[str, Any], top: int) -> str:
    left, right = result["eager"], result["compiled"]
    lines = [
        "# Eager versus compiled profiler comparison",
        "",
        "These are instrumented trace measurements. Summed kernel time counts overlaps;",
        "device union counts each covered interval once; device span includes gaps between",
        "the first and last kernel. None is end-to-end inference latency or compilation time.",
        "CPU operator times are inclusive and overlap their children. Compare identical",
        "profiled work; names alone do not prove that two kernels compute equivalent results.",
        "",
        "| Measurement | Eager | Compiled |",
        "|---|---:|---:|",
        f"| CUDA kernel calls | {left['kernel_calls']} | {right['kernel_calls']} |",
        f"| Sum of CUDA kernel durations, ms | {left['kernel_summed_us'] / 1000:.3f} "
        f"| {right['kernel_summed_us'] / 1000:.3f} |",
        f"| CPU operator calls | {left['cpu_op_calls']} | {right['cpu_op_calls']} |",
        f"| CUDA launch API calls | {left['cuda_launch_api_calls']} "
        f"| {right['cuda_launch_api_calls']} |",
        "",
    ]
    for label, summary in (("Eager", left), ("Compiled", right)):
        for device, timing in summary["kernel_devices"].items():
            lines.append(
                f"- {label} device {device}: "
                f"kernel interval union {timing['union_us'] / 1000:.3f} ms; "
                f"first-to-last kernel span {timing['span_us'] / 1000:.3f} ms."
            )
        for warning in summary["warnings"]:
            lines.append(f"- {label}: {warning}")
    for field, label in (
        ("kernels", "CUDA kernels"),
        ("cpu_ops", "CPU operators (inclusive duration)"),
        ("cuda_api", "CUDA runtime/driver API calls"),
    ):
        lines += [
            "",
            f"## {label}: largest absolute duration changes",
            "",
            "| Name | Eager calls | Compiled calls | Eager ms | Compiled ms "
            "| Eager minus compiled ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in result["differences"][field][:top]:
            name = row["name"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {name} | {row['eager_calls']} | {row['compiled_calls']} | "
                f"{row['eager_us'] / 1000:.3f} | {row['compiled_us'] / 1000:.3f} | "
                f"{row['eager_minus_compiled_us'] / 1000:+.3f} |"
            )
    lines += [
        "",
        "The JSON contains every named event aggregate, raw microsecond totals, source hashes,",
        "and generated-code origin comments. Generated source call-site counts are static",
        "evidence, not executed kernel counts. No inference speedup is estimated from them.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eager", type=Path)
    parser.add_argument("compiled", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output filename prefix")
    parser.add_argument("--generated-source", type=Path, action="append", default=[])
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    eager, compiled = summarize(args.eager), summarize(args.compiled)
    result = {
        "format": "h3.compile-trace-comparison/1",
        "duration_unit": "microseconds",
        "scope": "Complete X events in each supplied trace; no steps normalized or inferred.",
        "eager": eager,
        "compiled": compiled,
        "differences": {
            field: difference(eager[field], compiled[field])
            for field in ("kernels", "cpu_ops", "cuda_api", "gpu_copies_and_memsets")
        },
        "generated_sources": generated_sources(args.generated_source),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{args.out}.json")
    markdown_path = Path(f"{args.out}.md")
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    markdown_path.write_text(markdown(result, args.top))
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
