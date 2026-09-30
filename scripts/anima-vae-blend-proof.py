#!/usr/bin/env python
"""CPU bit proof against source-extracted installed Diffusers blend methods.

Run under the reviewed CPU device guard; no model/package imports or weights.
The full proof compares individual multiply/add results and actual tile traversal.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


class Operations(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Mult, ast.Add)):
            return ast.copy_location(
                ast.Call(
                    ast.Name("record", ast.Load()),
                    [ast.Constant(type(node.op).__name__), node],
                    [],
                ),
                node,
            )
        return node


class Trace:
    def __init__(self) -> None:
        self.enabled = False
        self.events: list[tuple[str, Any]] = []

    def record(self, operator: str, value: Any) -> Any:
        if self.enabled and isinstance(value, torch.Tensor):
            self.events.append((operator, value.detach().clone()))
        return value


def extract(tree: ast.Module, name: str, methods: set[str] | None = None) -> ast.ClassDef:
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    result = copy.deepcopy(node)
    if methods is not None:
        result.bases = []
        result.keywords = []
        result.decorator_list = []
        result.body = [
            node for node in result.body if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        if {node.name for node in result.body if isinstance(node, ast.FunctionDef)} != methods:
            raise RuntimeError("upstream blend/traversal methods changed")
    return result


def load_class(node: ast.ClassDef, trace: Trace, namespace: dict[str, Any]) -> Any:
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), Operations().visit(node)],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace.update(torch=torch, Any=Any, record=trace.record, DecoderOutput=SimpleNamespace)
    exec(compile(module, "<extracted-blend-source>", "exec"), namespace)
    return namespace[node.name]


def exact(a: Any, b: Any, label: str) -> None:
    if a.dtype != b.dtype or a.shape != b.shape:
        raise AssertionError(f"{label}: dtype/shape differ")
    if not torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8)):
        raise AssertionError(f"{label}: bit patterns differ")


def bank(dtype: Any, exhaustive: bool = False, nonfinite: bool = False) -> Any:
    bits = torch.finfo(dtype).bits
    integer = {16: torch.int16, 32: torch.int32, 64: torch.int64}[bits]
    if exhaustive:
        return torch.arange(1 << bits, dtype=torch.int32).to(integer).view(dtype)
    exponent = {torch.bfloat16: 8, torch.float16: 5, torch.float32: 8, torch.float64: 11}[dtype]
    fraction = bits - exponent - 1
    one = ((1 << (exponent - 1)) - 1) << fraction
    infinity = ((1 << exponent) - 1) << fraction
    values = [0, 1, 2, (1 << fraction) - 1, 1 << fraction, one - 1, one, one + 1,
              one + 2, infinity - 1, infinity, infinity + 1, infinity + (1 << (fraction - 1)),
              infinity + (1 << (fraction - 1)) + 3]
    if not nonfinite:
        values = [value for value in values if value < infinity]
    values += [value | (1 << (bits - 1)) for value in values]
    signed = [value if value < (1 << (bits - 1)) else value - (1 << bits) for value in values]
    return torch.tensor(signed, dtype=integer).view(dtype)


def values(dtype: Any, shape: tuple[int, ...], shift: int = 0, nonfinite: bool = False) -> Any:
    count = 1
    for side in shape:
        count *= side
    data = torch.roll(bank(dtype, nonfinite=nonfinite), shift)
    return data.repeat((count + data.numel() - 1) // data.numel())[:count].reshape(shape).clone()


def fixture(dtype: Any, axis: int, side: int, layout: str, channels: int = 3) -> tuple[Any, Any, Any]:
    shape = [1, channels, 1, 9, 9]
    shape[axis] = side
    size = tuple(shape)
    a, b = values(dtype, size, nonfinite=layout == "nonfinite"), values(
        dtype, size, 7, nonfinite=layout == "nonfinite"
    )
    if layout == "outside_nonfinite":
        aa, bb = a.narrow(axis, 0, side - 64), b.narrow(axis, 64, side - 64)
        aa.copy_(values(dtype, tuple(aa.shape), nonfinite=True))
        bb.copy_(values(dtype, tuple(bb.shape), nonfinite=True))
    if layout == "transpose":
        return (a.transpose(3, 4).contiguous().transpose(3, 4),
                b.transpose(3, 4).contiguous().transpose(3, 4), None)
    if layout == "sliced":
        aa, bb = a.repeat_interleave(2, dim=4), b.repeat_interleave(2, dim=4)
        return aa[..., ::2], bb[..., ::2], (aa, bb)
    if layout in ("shared", "same"):
        base = torch.cat((a, b), dim=axis)
        aa = base.narrow(axis, 0, side)
        bb = aa if layout == "same" else base.narrow(axis, 1, side)
        return aa, bb, base
    if layout == "frombuffer":
        count = a.numel()
        raw = bytearray(torch.cat((a.flatten(), b.flatten())).view(torch.uint8).tolist())
        aa = torch.frombuffer(raw, dtype=dtype, count=count).reshape(size)
        bb = torch.frombuffer(raw, dtype=dtype, count=count, offset=a.element_size()).reshape(size)
        assert aa.untyped_storage().data_ptr() != bb.untyped_storage().data_ptr()
        return aa, bb, raw
    if layout == "mixed":
        return a, b.to(torch.float32 if dtype != torch.float32 else torch.float64), None
    if layout == "grad":
        return a.requires_grad_(), b.requires_grad_(), None
    return a, b, None


def compare_holder(a: Any, b: Any, label: str) -> None:
    if isinstance(a, torch.Tensor):
        exact(a, b, label)
    elif isinstance(a, tuple):
        for x, y in zip(a, b, strict=True):
            compare_holder(x, y, label)
    elif a != b:
        raise AssertionError(f"{label}: backing bytes differ")


def blend_case(old: Any, candidate: Any, trace: Trace, axis: int, extent: int,
               first: tuple[Any, Any, Any], second: tuple[Any, Any, Any], label: str) -> dict[str, Any]:
    method = "blend_v" if axis == 3 else "blend_h"
    traces: list[list[tuple[str, Any]]] = []
    errors: list[tuple[str, str] | None] = []
    trace.enabled = True
    for owner, (a, b, _) in [(old, first), (candidate, second)]:
        trace.events = []
        try:
            result = getattr(owner, method)(a, b, extent)
            if result is not b:
                raise AssertionError(f"{label}: destination identity changed")
        except (RuntimeError, TypeError, IndexError, NotImplementedError) as error:
            errors.append((type(error).__name__, str(error)))
        else:
            errors.append(None)
        traces.append(list(trace.events))
    trace.enabled = False
    if errors[0] != errors[1]:
        raise AssertionError(f"{label}: exception behavior differs: {errors}")
    exact(first[0], second[0], f"{label} source/mutation")
    exact(first[1], second[1], f"{label} destination")
    compare_holder(first[2], second[2], f"{label} backing")
    effective = min(first[0].shape[axis], first[1].shape[axis], extent)
    fast = (len(traces[1]) == 3 and len(traces[0]) >= 3 and errors[0] is None
            and traces[0][0][1].ndim != traces[1][0][1].ndim)
    if fast:
        if len(traces[0]) != 3 * effective:
            raise AssertionError(f"{label}: old operation trace changed")
        for index in range(effective):
            for operation in range(3):
                name, tensor = traces[0][3 * index + operation]
                other_name, whole = traces[1][operation]
                if name != other_name:
                    raise AssertionError(f"{label}: operation order changed")
                exact(tensor, whole.select(axis, index), f"{label} product/add {index}:{operation}")
    else:
        if len(traces[0]) != len(traces[1]):
            raise AssertionError(f"{label}: fallback trace differs")
        for index, ((name, tensor), (other_name, other)) in enumerate(zip(*traces, strict=True)):
            if name != other_name:
                raise AssertionError(f"{label}: operation order differs")
            exact(tensor, other, f"{label} fallback product/add {index}")
    return {"case": label, "dtype": str(first[0].dtype), "axis": axis,
            "requested_extent": extent, "effective_extent": effective,
            "old_tensor_ops": len(traces[0]), "candidate_tensor_ops": len(traces[1]),
            "fast_path": fast, "matched_error": errors[0]}


def grid_case(old_type: Any, candidate_type: Any, width: int, height: int, dtype: Any) -> dict[str, Any]:
    snapshots: list[tuple[str, Any]] = []
    new_index = 0
    models = []
    counts = []
    for owner_type in (old_type, candidate_type):
        model = owner_type()
        model.spatial_compression_ratio = 8
        model.tile_sample_min_height = model.tile_sample_min_width = 256
        model.tile_sample_stride_height = model.tile_sample_stride_width = 192
        model.clear_cache = lambda model=model: setattr(model, "_feat_map", [])
        model.post_quant_conv = lambda x: x
        calls: list[int] = []

        def decoder(x: Any, calls: list[int] = calls, **_kwargs: Any) -> Any:
            calls.append(1)
            return (x.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)
                    + x[..., :1, :1] / 32)

        model.decoder = decoder
        for name in ("blend_v", "blend_h"):
            original = getattr(model, name)

            def observe(a: Any, b: Any, extent: int, original: Any = original,
                        name: str = name, is_old: bool = owner_type is old_type) -> Any:
                nonlocal new_index
                result = original(a, b, extent)
                if result is not b:
                    raise AssertionError("grid destination identity changed")
                if is_old:
                    snapshots.append((name, b.clone()))
                else:
                    expected_name, expected = snapshots[new_index]
                    if name != expected_name:
                        raise AssertionError("grid tile/blend order changed")
                    exact(expected, b, f"grid {width}x{height} blend {new_index}")
                    new_index += 1
                return result

            setattr(model, name, observe)
        models.append(model)
        counts.append(calls)
    generator = torch.Generator(device="cpu").manual_seed(width + height)
    latent = torch.randn((1, 3, 1, height // 8, width // 8), generator=generator, dtype=dtype)
    first = models[0].tiled_decode(latent.clone()).sample
    second = models[1].tiled_decode(latent.clone()).sample
    exact(first, second, f"grid {width}x{height} final")
    if new_index != len(snapshots) or len(counts[0]) != len(counts[1]):
        raise AssertionError("grid decode/blend counts changed")
    return {"width": width, "height": height, "dtype": str(dtype),
            "decoder_calls": len(counts[0]), "ordered_blends": new_index, "exact_bits": True}


def negative_controls(old_node: ast.ClassDef, old_type: Any, trace: Trace) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(295)
    a = torch.randn((1, 3, 1, 64, 19), generator=generator, dtype=torch.bfloat16)
    b = torch.randn(a.shape, generator=generator, dtype=torch.bfloat16)
    alpha = torch.tensor([1 - i / 64 for i in range(64)], dtype=a.dtype).reshape(1, 1, 1, 64, 1)
    beta = torch.tensor([i / 64 for i in range(64)], dtype=a.dtype).reshape(1, 1, 1, 64, 1)
    separate = a * alpha + b * beta
    fused = torch.addcmul(a * alpha, b, beta)
    try:
        exact(separate, fused, "negative fused multiply/add")
    except AssertionError:
        fused_rejected = True
    else:
        raise AssertionError("rounding fixture did not reject fused arithmetic")
    wrong_node = copy.deepcopy(old_node)
    wrong_node.name = "WrongOrder"
    changed = False
    for node in ast.walk(wrong_node):
        if (isinstance(node, ast.For) and len(node.body) >= 2
                and isinstance(node.body[0], ast.If) and isinstance(node.body[1], ast.If)
                and "self.blend_v" in ast.unparse(node.body[0])
                and "self.blend_h" in ast.unparse(node.body[1])):
            node.body[0], node.body[1] = node.body[1], node.body[0]
            changed = True
    if not changed:
        raise AssertionError("negative traversal fixture failed to locate actual blend order")
    wrong_type = load_class(wrong_node, trace, {})
    try:
        grid_case(old_type, wrong_type, 1024, 1024, torch.bfloat16)
    except AssertionError as error:
        if "blend order changed" not in str(error):
            raise
        order_rejected = True
    else:
        raise AssertionError("grid fixture did not reject reordered blends")
    return {"fused_arithmetic_rejected": fused_rejected, "changed_traversal_order_rejected": order_rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--upstream-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.read_bytes()
    if hashlib.sha256(upstream).hexdigest() != args.upstream_sha256:
        raise RuntimeError("upstream proof source differs from the reviewed pinned bytes")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = Path(__file__).resolve().parents[1] / "anima/anima/__init__.py"
    trace = Trace()
    old_node = extract(ast.parse(upstream), "AutoencoderKLQwenImage",
                       {"blend_v", "blend_h", "tiled_decode"})
    old_type = load_class(copy.deepcopy(old_node), trace, {})
    candidate_type = load_class(extract(ast.parse(source.read_text()), "_AnimaVae"), trace,
                                {"AutoencoderKLQwenImage": old_type})
    old, candidate = old_type(), candidate_type()
    cases = []
    dtypes = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    for dtype in dtypes:
        for axis in (3, 4):
            for extent in (-1, 0, 1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 256, 257, 512):
                side = max(70, extent)
                label = f"{dtype}:{axis}:extent{extent}"
                cases.append(blend_case(old, candidate, trace, axis, extent,
                                        fixture(dtype, axis, side, "plain"),
                                        fixture(dtype, axis, side, "plain"), label))
            for layout in ("transpose", "sliced", "shared", "same", "frombuffer", "mixed", "grad", "nonfinite", "outside_nonfinite"):
                cases.append(blend_case(old, candidate, trace, axis, 64,
                                        fixture(dtype, axis, 70, layout),
                                        fixture(dtype, axis, 70, layout), f"{dtype}:{axis}:{layout}"))
            for channels in (0, 1, 4):
                cases.append(blend_case(old, candidate, trace, axis, 256,
                                        fixture(dtype, axis, 64, "plain", channels),
                                        fixture(dtype, axis, 64, "plain", channels),
                                        f"{dtype}:{axis}:clamped/channels{channels}"))
            for shape in [(2, 5, 3, 72, 72), (1, 3, 1, 0, 0)]:
                a, b = values(dtype, shape), values(dtype, shape, 7)
                cases.append(blend_case(old, candidate, trace, axis, 64,
                                        (a.clone(), b.clone(), None), (a.clone(), b.clone(), None),
                                        f"{dtype}:{axis}:shape{shape}"))
            first, second = fixture(dtype, axis, 128, "plain"), fixture(dtype, axis, 128, "plain")
            first = first[0], first[1].narrow(axis, 0, 64).clone(), None
            second = second[0], second[1].narrow(axis, 0, 64).clone(), None
            cases.append(blend_case(old, candidate, trace, axis, 256, first, second,
                                    f"{dtype}:{axis}:unequal-clamped-edge"))
            a = values(dtype, (1, 3, 2, 70, 70))
            b = values(dtype, (1, 3, 1, 70, 70), 7)
            cases.append(blend_case(old, candidate, trace, axis, 64,
                                    (a.clone(), b.clone(), None), (a.clone(), b.clone(), None),
                                    f"{dtype}:{axis}:inplace-shape-failure"))
        if dtype in (torch.bfloat16, torch.float16):
            data = bank(dtype, exhaustive=True).reshape(1, 1, 1, 256, 256)
            cases.append(blend_case(old, candidate, trace, 3, 256,
                                    (data.clone(), data.flip(4).clone(), None),
                                    (data.clone(), data.flip(4).clone(), None), f"{dtype}:all65536bits"))
            finite = data.flatten()[torch.isfinite(data.flatten())].reshape(1, 1, 1, 256, -1)
            cases.append(blend_case(old, candidate, trace, 3, 256,
                                    (finite.clone(), finite.flip(4).clone(), None),
                                    (finite.clone(), finite.flip(4).clone(), None),
                                    f"{dtype}:all-finite-bit-patterns"))
    for axis in (3, 4):
        a = torch.arange(1 * 2 * 1 * 70 * 70, dtype=torch.int32).reshape(1, 2, 1, 70, 70)
        b = a.flip(axis).clone()
        cases.append(blend_case(old, candidate, trace, axis, 64,
                                (a.clone(), b.clone(), None), (a.clone(), b.clone(), None),
                                f"integer-fallback:{axis}"))
    meta_a = torch.empty((1, 3, 1, 70, 70), device="meta")
    meta_b = torch.empty_like(meta_a)
    if candidate._blend(meta_a, meta_b, 64, 3) is not False:
        raise AssertionError("unproved meta device selected fast path")
    if old.blend_v(meta_a, meta_b, 64) is not meta_b or candidate.blend_v(meta_a, meta_b, 64) is not meta_b:
        raise AssertionError("meta fallback destination identity changed")
    grids = [grid_case(old_type, candidate_type, w, h, dtype)
             for dtype in dtypes
             for w, h in [(512, 512), (1024, 1024), (1152, 896), (1344, 768),
                          (896, 1152), (768, 1344), (1536, 1536), (1728, 1344),
                          (2016, 1152), (1344, 1728), (1152, 2016)]]
    negatives = negative_controls(old_node, old_type, trace)
    result = {"status": "PASS CPU-only source-body bit proof", "torch": torch.__version__,
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "upstream": str(args.upstream), "upstream_sha256": args.upstream_sha256,
              "candidate": str(source), "candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "proof_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": cases, "grids": grids, "negative_controls": negatives,
              "meta_device_scalar_fallback_identity": True,
              "limits": ["No VAE weights or GPU execution", "No CUDA/backend/compile/autograd-speed equivalence claimed", "Grid decoder is a CPU deterministic stub; actual pinned tiled traversal and blend methods execute"]}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "blend_cases": len(cases),
                      "grid_cases": len(grids), "fast_cases": sum(case["fast_path"] for case in cases)}))


if __name__ == "__main__":
    main()
