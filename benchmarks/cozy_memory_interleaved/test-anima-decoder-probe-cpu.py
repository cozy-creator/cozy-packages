"""Source-extracted CPU contracts; no application/model imports or CUDA checks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import runpy
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from contextlib import nullcontext

import torch
from safetensors.torch import load, save
from cozy_runtime.internal import output_budget


ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/fidika/.cozy/outputs/comfy-cozy-memory-20260929")
OUT = BASE / "analysis-anima-blend-private-prep-20260930"
PACKAGE = (
    ROOT
    / "benchmarks/cozy_memory_interleaved/cohorts/r20-anima-blend-decoder-probe-20260930/packages/anima"
)
SOURCE = PACKAGE / "anima/diagnostic_decode.py"
PINNED = Path(
    "/home/fidika/.cozy/machine/root/var/lib/cozy/installs/installations/d8a19dc0055fd789/venv/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py"
)


def node(tree: ast.Module, name: str) -> Any:
    return copy.deepcopy(next(x for x in tree.body if getattr(x, "name", None) == name))


def execute(nodes: list[Any], namespace: dict[str, Any]) -> None:
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    exec(compile(module, "<actual-source-cpu-fixture>", "exec"), namespace)


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    text = SOURCE.read_text()
    tree = ast.parse(text)
    exports = {
        "image": 64 << 20,
        "argument": 1 << 20,
        "scalar_raw": 16 << 20,
        "broadcast_raw": 16 << 20,
        "scalar_rgb": 4 << 20,
        "broadcast_rgb": 4 << 20,
        "report": 1 << 20,
        "trace": 128 << 20,
    }
    names = (
        "_conv_num",
        "_conv_idx",
        "_feat_map",
        "_enc_conv_num",
        "_enc_conv_idx",
        "_enc_feat_map",
    )
    diag_names = (
        "_diag_mode",
        "_diag_profile",
        "_diag_capture",
        "_diag_argument",
        "_diag_baseline",
        "_diag_fast",
        "_diag_scalar",
    )
    ns: dict[str, Any] = {
        "torch": torch,
        "Any": Any,
        "cast": cast,
        "json": json,
        "save_tensors": save,
        "load_tensors": load,
        "ARG_LIMIT": 1 << 20,
        "RAW_LIMIT": 16 << 20,
        "EXPORT_LIMITS": exports,
        "CACHE_NAMES": names,
        "DIAG_NAMES": diag_names,
        "DecoderOutput": SimpleNamespace,
        "nullcontext": nullcontext,
        "hashlib": hashlib,
    }
    retry_tree = ast.parse((OUT / "sources/_retry.py").read_text())
    execute([node(retry_tree, "retry_state")], ns)
    ns["cozy_author"] = SimpleNamespace(retry_state=ns["retry_state"])
    functions = [
        "_require_native_cuda",
        "_host_tensor_bytes",
        "_drain_native",
        "_rgb",
        "_exact_raw",
        "_raw_bit_digest",
        "_profiled_blend",
        "_initialize_diagnostic",
        "_declare_diagnostic_retry_state",
        "_save_bounded",
    ]
    execute([node(tree, name) for name in functions], ns)
    checks: list[str] = []
    for constant in ("FROZEN_PAYLOAD", "FROZEN_CHECKPOINT", "BASELINE_RGB"):
        assignment = next(
            x
            for x in tree.body
            if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == constant for t in x.targets)
        )
        execute([copy.deepcopy(assignment)], ns)
    execute([node(tree, "_require_frozen_probe")], ns)
    frozen = ns["FROZEN_PAYLOAD"]
    manifest = json.loads(
        (BASE / "analysis-startup-store-read-qualification/control-input-manifest.json").read_text()
    )
    frozen_run = next(
        row
        for row in manifest["suites"]["grouped6"]
        if row["model"] == "Anima" and row["seed"] == 1006
    )
    assert frozen == frozen_run["payload"] and ns["FROZEN_CHECKPOINT"] == frozen_run["checkpoint"]
    ns["_require_frozen_probe"](SimpleNamespace(**frozen), frozen_run["checkpoint"])
    qualified_author = ast.parse(
        Path(manifest["packages"]["anima"]["path"]).joinpath("anima/__init__.py").read_text()
    )
    ns.update(Enum=Enum, IntEnum=IntEnum)
    execute([node(qualified_author, "AspectRatio"), node(qualified_author, "Megapixels")], ns)
    enum_payload = {
        **frozen,
        "aspect_ratio": ns["AspectRatio"].SQUARE,
        "megapixels": ns["Megapixels"].MP1,
    }
    ns["_require_frozen_probe"](SimpleNamespace(**enum_payload), frozen_run["checkpoint"])
    for field in frozen:
        bad = dict(frozen)
        value = bad[field]
        bad[field] = value + " changed" if isinstance(value, str) else value + 1
        try:
            ns["_require_frozen_probe"](SimpleNamespace(**bad), frozen_run["checkpoint"])
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"changed frozen {field} accepted")
    try:
        ns["_require_frozen_probe"](SimpleNamespace(**frozen), "sha256:" + "0" * 64)
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed checkpoint accepted")
    checks.append(
        "all twelve original frozen request fields and exact checkpoint guard; each changed field refuses"
    )
    for dtype, integer in [(torch.bfloat16, torch.int16), (torch.float32, torch.int32)]:
        bits = (
            [0, -32768, 1, 0x7FC1] if dtype == torch.bfloat16 else [0, -2147483648, 1, 0x7FC12345]
        )
        tensor = torch.tensor(bits, dtype=integer).view(dtype)
        data = ns["_host_tensor_bytes"](tensor, "decoded")
        decoded = load(data)["tensor"]
        assert torch.equal(tensor.view(torch.uint8), decoded.view(torch.uint8))
        assert decoded.dtype == dtype and decoded.shape == tensor.shape
        assert (
            ns["_raw_bit_digest"](data)
            == hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        )
    checks.append("BF16/FP32 signedzero/subnormal/NaN-payload safetensors exact roundtrip")

    class Spy:
        device = SimpleNamespace(type="cuda")
        dtype = torch.bfloat16
        shape = (1, 16, 1, 128, 128)

        def __init__(self, contiguous: bool = True, conj: bool = False, neg: bool = False) -> None:
            self.contiguous, self.conj, self.neg, self.calls = contiguous, conj, neg, 0

        def is_contiguous(self) -> bool:
            return self.contiguous

        def is_conj(self) -> bool:
            return self.conj

        def is_neg(self) -> bool:
            return self.neg

        def stride(self) -> tuple[int, ...]:
            return (262144, 16384, 16384, 128, 1)

        def to(self, **kwargs: Any) -> Any:
            self.calls += 1
            assert kwargs == {"device": "cpu", "dtype": self.dtype, "non_blocking": False}
            return torch.zeros(self.shape, dtype=self.dtype)

    for spy in (Spy(False), Spy(conj=True), Spy(neg=True)):
        try:
            ns["_drain_native"](spy, "argument")
        except RuntimeError:
            assert spy.calls == 0
        else:
            raise AssertionError("unsafe CUDA-like spy reached drain")
    good = Spy()
    captured = ns["_drain_native"](good, "argument")
    assert good.calls == 1 and len(captured) < exports["argument"]
    checks.append("CUDA-like contiguous/conj/neg guards precede any .to; safe same-dtype drain")

    old_tree = ast.parse(PINNED.read_text())
    old_class = node(old_tree, "AutoencoderKLQwenImage")
    old_class.bases = []
    old_class.keywords = []
    old_class.decorator_list = []
    selected = {"decode", "_decode", "tiled_decode", "blend_v", "blend_h"}
    old_class.body = [
        x for x in old_class.body if isinstance(x, ast.FunctionDef) and x.name in selected
    ]
    for method in old_class.body:
        method.decorator_list = []
    execute([old_class], ns)
    helper_tree = ast.parse((PACKAGE / "anima/__init__.py").read_text())
    execute([node(helper_tree, "_AnimaVae"), node(tree, "_DiagnosticVae")], ns)
    cls = ns["_DiagnosticVae"]
    vae = cls()
    ns["_initialize_diagnostic"](vae)
    vae.spatial_compression_ratio = 8
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = 8
    vae.use_tiling = True
    vae.use_slicing = False
    vae.post_quant_conv = lambda x: x
    vae.decoder = lambda x, **kwargs: x.repeat_interleave(8, 3).repeat_interleave(8, 4)

    def clear() -> None:
        vae._conv_num = 1
        vae._conv_idx = [0]
        vae._feat_map = [None]
        vae._enc_conv_num = 1
        vae._enc_conv_idx = [0]
        vae._enc_feat_map = [None]

    vae.clear_cache = clear
    clear()
    ns["_declare_diagnostic_retry_state"](vae)
    save_state, restore = vars(vae)["_cozy_retry_state"]
    assert vars(vae)["_cozy_retry_hooks"] == vars(vae)["_cozy_retry_forwards"] == ()
    initial = save_state()
    generator = torch.Generator(device="cpu").manual_seed(1006)
    z = torch.randn((1, 3, 1, 4, 4), generator=generator, dtype=torch.bfloat16)
    global_rng, request_rng = torch.get_rng_state(), generator.get_state()
    scalar = vae.decode(z, return_dict=False)[0]
    assert vae._diag_argument == vae._diag_baseline == b"" and vae._diag_mode == "scalar"
    broadcast = vae.decode(z, return_dict=False, blend_mode="broadcast")[0]
    assert torch.equal(
        scalar.contiguous().view(torch.uint8), broadcast.contiguous().view(torch.uint8)
    )
    assert torch.equal(global_rng, torch.get_rng_state()) and torch.equal(
        request_rng, generator.get_state()
    )
    checks.append(
        "actual pinned tiny decode/tiled loops scalar/broadcast; warm/off no capture; global/request RNG unchanged"
    )
    # Establish the real full-request layout instead of assuming that the final
    # crop is noncontiguous. Edge tiles truncate before the row concatenation.
    # This invokes upstream's actual loop without constructing upstream weights.
    vae.tile_sample_min_height = vae.tile_sample_min_width = 256
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = 192
    full_argument = torch.zeros((1, 16, 1, 128, 128), dtype=torch.bfloat16)
    vae.decoder = lambda x, **kwargs: x[:, :3].repeat_interleave(8, 3).repeat_interleave(8, 4)
    full_scalar = vae.decode(full_argument, return_dict=False, blend_mode="scalar")[0]
    scalar_paths = (vae._diag_fast, vae._diag_scalar)
    full_broadcast = vae.decode(full_argument, return_dict=False, blend_mode="broadcast")[0]
    broadcast_paths = (vae._diag_fast, vae._diag_scalar)
    assert tuple(full_scalar.shape) == tuple(full_broadcast.shape) == (1, 3, 1, 1024, 1024)
    assert full_scalar.is_contiguous() and full_broadcast.is_contiguous()
    assert full_scalar.stride() == full_broadcast.stride()
    assert torch.equal(
        full_scalar.contiguous().view(torch.uint8), full_broadcast.contiguous().view(torch.uint8)
    )
    assert scalar_paths == (0, 60) and broadcast_paths == (60, 0)
    full_profiled = vae.decode(
        full_argument, return_dict=False, blend_mode="broadcast", profile_decode=True
    )[0]
    assert torch.equal(full_broadcast.view(torch.uint8), full_profiled.view(torch.uint8))
    assert vae._diag_mode == "scalar" and vae._diag_profile is False
    layout_fact = {
        "shape": list(full_scalar.shape),
        "stride": list(full_scalar.stride()),
        "storage_bytes": full_scalar.untyped_storage().nbytes(),
        "logical_bytes": full_scalar.numel() * full_scalar.element_size(),
        "contiguous": True,
        "scalar_paths": list(scalar_paths),
        "broadcast_paths": list(broadcast_paths),
        "noncontiguous_cuda_drain_guard_refuses_before_copy": True,
        "weights_constructed": False,
        "boundary": "actual pinned tiled_decode with shape-preserving CPU decoder double",
    }
    unsafe_output = Spy(False)
    unsafe_output.shape = tuple(full_scalar.shape)
    try:
        ns["_drain_native"](unsafe_output, "decoded")
    except RuntimeError:
        assert unsafe_output.calls == 0
    else:
        raise AssertionError("unsafe full-shape CUDA-like result reached drain")
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = 8
    vae.decoder = lambda x, **kwargs: x.repeat_interleave(8, 3).repeat_interleave(8, 4)
    checks.append(
        "full1024 pinned traversal proves contiguous edge-truncated result; unsafe CUDA layouts still refuse before copy"
    )
    restore(initial)
    for name in names:
        assert vars(vae)[name] is initial[name][0]
    snapshot = save_state()
    old_ids = {name: id(vars(vae)[name]) for name in names}
    for name in names:
        vars(vae)[name] = [object()] if isinstance(vars(vae)[name], list) else 99
    vae._diag_mode = "broadcast"
    vae._diag_argument = b"failed"
    restore(snapshot)
    assert all(id(vars(vae)[name]) == old_ids[name] for name in names)
    assert vae._diag_mode == "scalar" and vae._diag_argument == b""
    checks.append("six cache bindings/list identity and diagnostic immutable bindings restore")
    sentinel = RuntimeError("exact original decoder failure")
    normal_decoder = vae.decoder

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise sentinel

    vae.decoder = failed
    try:
        vae.decode(z, return_dict=False, blend_mode="broadcast", profile_decode=True)
    except RuntimeError as error:
        assert error is sentinel and vae._diag_mode == "scalar" and vae._diag_profile is False
    else:
        raise AssertionError("decoder failure swallowed")
    restore(snapshot)
    vae.decoder = normal_decoder
    checks.append("failed replay original exception and finally mode restoration")
    vae._diag_capture = True
    ns["_require_native_cuda"] = lambda tensor, role: None
    original_drain = ns["_drain_native"]
    drains: list[str] = []

    def capture_failed(tensor: Any, role: str) -> bytes:
        drains.append(role)
        if role == "decoded":
            raise sentinel
        return b"actual argument bytes"

    ns["_drain_native"] = capture_failed
    capture_snapshot = save_state()
    try:
        vae.decode(z, return_dict=False, blend_mode="scalar")
    except RuntimeError as error:
        assert error is sentinel and vae._diag_mode == "scalar"
    else:
        raise AssertionError("capture failure swallowed")
    restore(capture_snapshot)
    assert vae._diag_argument == capture_snapshot["_diag_argument"][0]
    ns["_drain_native"] = lambda tensor, role: (
        b"actual argument bytes" if role == "argument" else b"actual decoded bytes"
    )
    vae.decode(z, return_dict=False, blend_mode="scalar")
    assert (
        vae._diag_argument == b"actual argument bytes"
        and vae._diag_baseline == b"actual decoded bytes"
    )
    ns["_drain_native"] = original_drain
    vae._diag_capture = False
    vae.decode(z, return_dict=False, blend_mode="scalar")
    vae.decoder = failed
    second_snapshot = save_state()
    try:
        vae.decode(z, return_dict=False, blend_mode="broadcast", profile_decode=True)
    except RuntimeError as error:
        assert error is sentinel and vae._diag_mode == "scalar" and vae._diag_profile is False
    else:
        raise AssertionError("second replay failure swallowed")
    restore(second_snapshot)
    vae.decoder = normal_decoder
    checks.append("capture success/failure retry and second replay original exception/state")

    interface = json.loads((OUT / "static-diagnostic-interface.json").read_text())
    entry = interface["entrypoints"]
    assert len(entry) == 1 and entry[0]["name"] == "generate"
    fields = {
        f["name"]: f["asset_bound"]["max_bytes"]
        for f in entry[0]["result"]["fields"]
        if "asset_bound" in f
    }
    assert fields == exports and sum(fields.values()) == 234 << 20
    spec = {
        "outputs": [
            {
                "mime_type": field["asset_bound"]["media_types"][0],
                "max_bytes": field["asset_bound"]["max_bytes"],
            }
            for field in entry[0]["result"]["fields"]
            if "asset_bound" in field
        ]
    }
    assert output_budget.admitted(spec) == 234 << 20
    assert output_budget.intermediate(spec) == 256 << 20
    assert output_budget.effective(spec, {"max_output_bytes": 256 << 20}) == 234 << 20
    assert output_budget.effective(spec, {"max_output_bytes": 4 << 20}) < sum(fields.values())
    checks.append(
        "actual AST eight exports sum234MiB; real Runtime6660 output-budget admitted/intermediate/effective gates"
    )

    class Sink:
        def __init__(self) -> None:
            self.saved = 0

        def save_bytes(self, data: bytes, **kwargs: Any) -> bytes:
            self.saved += 1
            return data

    sink = Sink()
    try:
        ns["_save_bounded"](sink, b"x" * ((1 << 20) + 1), "argument")
    except RuntimeError:
        assert sink.saved == 0
    else:
        raise AssertionError("oversize artifact silently saved")
    checks.append("individual bound overflow rejected before save")

    class OversizeTrace:
        def __len__(self) -> int:
            return (128 << 20) + 1

    try:
        ns["_save_bounded"](sink, OversizeTrace(), "trace")
    except RuntimeError:
        assert sink.saved == 0
    else:
        raise AssertionError("oversize combined trace silently saved")
    validator = runpy.run_path(
        str(ROOT / "benchmarks/cozy_memory_interleaved/validate-anima-decoder-probe.py")
    )["validate"]
    report = {
        "producer_complete": True,
        "runtime_gate_pending": True,
        "scored": False,
        "raw_bits_exact": True,
        "rgb_exact": True,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "source_profile_decode_calls": 2,
        "profile_regions": 1,
        "modes": [
            {
                "mode": "scalar",
                "fast_blends": 0,
                "scalar_blends": 60,
                "decode_host_wall_ms": 1.0,
                "decode_cuda_event_ms": 1.0,
                "raw_bits_sha256": "0" * 64,
                "raw_artifact_sha256": "0" * 64,
                "rgb_sha256": "1" * 64,
            },
            {
                "mode": "broadcast",
                "fast_blends": 60,
                "scalar_blends": 0,
                "decode_host_wall_ms": 1.0,
                "decode_cuda_event_ms": 1.0,
                "raw_bits_sha256": "0" * 64,
                "raw_artifact_sha256": "0" * 64,
                "rgb_sha256": "1" * 64,
            },
        ],
        "argument_sha256": "0" * 64,
        "argument_bits_sha256": "0" * 64,
        "baseline_raw_internal_sha256": "0" * 64,
        "baseline_raw_bits_sha256": "0" * 64,
        "baseline_rgb_sha256": "1" * 64,
        "operators": [{"device_total_us": 1}],
        "tile": 256,
        "stride": 192,
    }
    inventory = dict.fromkeys(exports, 1)
    runtime = {"vae_invokes": 3, "allocator_ooms": 0, "allocator_retries": 0, "ledger_closed": True}
    scope = {
        "pid": 1,
        "start_ticks": 1,
        "memory_high_bytes": 12 << 30,
        "memory_max_bytes": 16 << 30,
        "cpu_cores": 4,
        "nice": 19,
        "swap_bytes": 0,
    }
    validator(report, inventory, interface, spec, {"max_output_bytes": 256 << 20}, runtime, scope)
    negatives = 0
    for changed_report, changed_inventory, changed_reply, changed_runtime, changed_scope in [
        ({}, inventory, {"max_output_bytes": 256 << 20}, runtime, scope),
        (
            {**report, "producer_complete": False},
            inventory,
            {"max_output_bytes": 256 << 20},
            runtime,
            scope,
        ),
        (
            report,
            {key: size for key, size in inventory.items() if key != "trace"},
            {"max_output_bytes": 256 << 20},
            runtime,
            scope,
        ),
        (report, {**inventory, "third_raw": 1}, {"max_output_bytes": 256 << 20}, runtime, scope),
        (report, inventory, {"max_output_bytes": 4 << 20}, runtime, scope),
        (report, inventory, {"max_output_bytes": 256 << 20}, {}, scope),
        (
            report,
            inventory,
            {"max_output_bytes": 256 << 20},
            {**runtime, "allocator_retries": 1},
            scope,
        ),
        (report, inventory, {"max_output_bytes": 256 << 20}, runtime, {}),
        (
            {**report, "baseline_rgb_sha256": "wrong"},
            inventory,
            {"max_output_bytes": 256 << 20},
            runtime,
            scope,
        ),
        (
            {**report, "argument_sha256": "missing"},
            inventory,
            {"max_output_bytes": 256 << 20},
            runtime,
            scope,
        ),
        (
            {
                **report,
                "modes": [
                    {**report["modes"][0], "decode_cuda_event_ms": float("nan")},
                    report["modes"][1],
                ],
            },
            inventory,
            {"max_output_bytes": 256 << 20},
            runtime,
            scope,
        ),
    ]:
        try:
            validator(
                changed_report,
                changed_inventory,
                interface,
                spec,
                changed_reply,
                changed_runtime,
                changed_scope,
            )
        except ValueError:
            negatives += 1
        else:
            raise AssertionError("malformed/incomplete proof accepted")
    assert negatives == 11
    checks.append(
        "trace overflow and11missing/malformed/digest/timing/live-grant/scope consumer negatives rejected"
    )
    source_fn = node(tree, "_generate_baseline")
    qualified_tree = ast.parse(
        Path(
            json.loads(
                (
                    BASE / "analysis-startup-store-read-qualification/control-input-manifest.json"
                ).read_text()
            )["packages"]["anima"]["path"]
        )
        .joinpath("anima/__init__.py")
        .read_text()
    )
    expected = node(qualified_tree, "generate")
    assert [ast.dump(x, include_attributes=False) for x in source_fn.body] == [
        ast.dump(x, include_attributes=False) for x in expected.body
    ]
    checks.append("full qualified generation body source facts exact")
    old_pipeline = node(qualified_tree, "AnimaPipeline")
    old_constructor = next(
        x for x in old_pipeline.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"
    )
    new_pipeline = node(tree, "DiagnosticPipeline")
    new_constructor = next(
        x for x in new_pipeline.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"
    )
    expected_body = copy.deepcopy(old_constructor.body)

    class ConstructorProjection(ast.NodeTransformer):
        def visit_Name(self, current: ast.Name) -> ast.AST:
            if current.id == "AutoencoderKLQwenImage":
                current.id = "_DiagnosticVae"
            return current

        def visit_Expr(self, current: ast.Expr) -> Any:
            if (
                isinstance(current.value, ast.Call)
                and isinstance(current.value.func, ast.Name)
                and current.value.func.id == "_declare_vae_retry_state"
            ):
                return ast.parse(
                    "_initialize_diagnostic(vae)\n_declare_diagnostic_retry_state(vae)"
                ).body
            return self.generic_visit(current)

    projected: list[ast.stmt] = []
    for statement in expected_body:
        transformed = ConstructorProjection().visit(statement)
        projected.extend(transformed if isinstance(transformed, list) else [transformed])
    assert [ast.dump(x, include_attributes=False) for x in projected] == [
        ast.dump(x, include_attributes=False) for x in new_constructor.body
    ]
    old_contract = node(qualified_tree, "_declare_vae_retry_state")
    new_contract = node(tree, "_declare_diagnostic_retry_state")
    for name in ("save", "restore"):
        first = next(
            x for x in old_contract.body if isinstance(x, ast.FunctionDef) and x.name == name
        )
        second = next(
            x for x in new_contract.body if isinstance(x, ast.FunctionDef) and x.name == name
        )
        assert ast.dump(first, include_attributes=False) == ast.dump(
            second, include_attributes=False
        )
    profiled_helper = node(tree, "_profiled_blend")
    original_helper = node(helper_tree, "_AnimaVae")
    original_blend = next(
        x for x in original_helper.body if isinstance(x, ast.FunctionDef) and x.name == "_blend"
    )

    class RemoveProfileRanges(ast.NodeTransformer):
        def visit_With(self, current: ast.With) -> Any:
            assert len(current.items) == 1
            call = current.items[0].context_expr
            assert (
                isinstance(call, ast.Call)
                and ast.unparse(call.func) == "torch.profiler.record_function"
            )
            assert len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
            return [self.visit(x) for x in current.body]

    stripped = RemoveProfileRanges().visit(profiled_helper)
    assert [ast.dump(x, include_attributes=False) for x in stripped.body] == [
        ast.dump(x, include_attributes=False) for x in original_blend.body
    ]
    checks.append(
        "profiled helper inverse body exact; full1024 scoped helper output bits/counters unchanged; no CUDA profiler activation"
    )
    base_interface = json.loads((OUT / "static-base-interface.json").read_text())
    assert (
        base_interface["entrypoints"][0]["request"]["fields"] == entry[0]["request"]["fields"][:-1]
    )
    assert entry[0]["request"]["fields"][-1] == {
        "default": False,
        "name": "probe",
        "type": "bool",
        "wire": "optional",
    }
    original_uses = base_interface["entrypoints"][0]["models"][0]["component_use"]
    new_uses = entry[0]["models"][0]["component_use"]
    assert all(new_uses[name] == components for name, components in original_uses.items())
    assert not node(tree, "DiagnosticModel").keywords
    assert (
        base_interface["entrypoints"][0]["models"][0]["encoded_leaves"]
        == entry[0]["models"][0]["encoded_leaves"]
        == "accept"
    )
    memory_tree = ast.parse((OUT / "sources/memory.py").read_text())
    scalar_node = next(
        x
        for x in memory_tree.body
        if isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_SCALAR" for t in x.targets)
    )
    ns.update(bool=bool, int=int, float=float, str=str, type=type)
    execute([scalar_node, node(memory_tree, "_leaves")], ns)
    old_flags = tuple(
        x
        for x in ns["_leaves"](((), {"blend_mode": "scalar", "return_dict": False}))
        if isinstance(x, ns["_SCALAR"])
    )
    new_flags = tuple(
        x
        for x in ns["_leaves"](((), {"blend_mode": "broadcast", "return_dict": False}))
        if isinstance(x, ns["_SCALAR"])
    )
    assert old_flags != new_flags and "scalar" in old_flags and "broadcast" in new_flags
    checks.append(
        "constructor inverse projection, exact old callbacks, inherited scopes/defaults and real mode scalar leaves"
    )
    result = {
        "status": "CPU contract fixtures pass; no GPU/application/model import",
        "checks": checks,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "inventory_bytes": sum(fields.values()),
        "full_decode_layout": layout_fact,
        "gpu_probe_source_gate": "pending root and independent source review; actual GPU layout still requires the unchanged guard",
        "actual_worker_grant_still_required": True,
        "remaining": [
            "accepted Worker/interface/output-grant and actual scope resource limits",
            "source/independent review before any activation",
        ],
    }
    (OUT / "DIAGNOSTIC-CPU-CONTRACTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(checks),
                "inventory_MiB": sum(fields.values()) >> 20,
            }
        )
    )


if __name__ == "__main__":
    main()
