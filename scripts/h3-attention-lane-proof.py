#!/usr/bin/env python3
"""Run attention-lane through real Runtime and TensorFS over a tiny H3-shaped pruned fixture.

Proves, for every lane the package declares, that the produced header is the source header
plus one canonical `execution` config; that the config is byte-identical to the design
document typed out here (attention-quantization.md §2, h3a-012) and that the runtime's own
contract reader (cr-109) admits both at the same digest; that every tensor entry — object
refs included — and every other config is byte-identical to the source's with zero source
reads; that a retry replays the receipts; and that the typed refusals arm — a produced lane
as a source (it already carries `execution`) and a BF16 source (its bytes are not the route
the contracts name).
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import tensorfs
from cozy_runtime.author import UnsupportedInput, WeightsSink, canonical_json
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt
from cozy_runtime.derive.quantization import h3_quantization_plan, prepare_quantization
from cozy_runtime.internal.execution_contract import read as read_contract
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from h3_tables import job
from h3_tables.model_config import dual_adaln_pruned_config, parse_production_config
from h3_tables.source import TARGET_COMPONENT, H3FullTransformer, full_targets

#: attention-quantization.md §2 (decision #707) and h3a-012's same-class kernel, verbatim.
#: The package assets are these documents' canonical bytes and nothing else.
DESIGN: dict[str, dict[str, Any]] = {
    "sm90-attn8": {
        "device": "sm90",
        "activations": "bf16",
        "class": "quantized",
        "weights": {"route": "encoded_gemm"},
        "attention": {
            "distribution": "sageattention",
            "version": "2.2.0",
            "entry": "sageattn_qk_int8_pv_fp8_cuda_sm90",
            "kwargs": {
                "tensor_layout": "NHD",
                "qk_quant_gran": "per_thread",
                "pv_accum_dtype": "fp32+fp32",
                "smooth_k": True,
                "is_causal": False,
            },
            "scheme": (
                "qk int8 per-thread (Q64/16, K128/128) K-mean-smoothed; v fp8-e4m3 "
                "per-channel; pv fp32+fp32"
            ),
        },
    },
    "sm90-fa3": {
        "device": "sm90",
        "activations": "bf16",
        "class": "same",
        "weights": {"route": "encoded_gemm"},
        "attention": {
            "distribution": "flash-attn3",
            "version": "1",
            "revision": "7cb368cf8278b583132eb72cbf312d54586df2e2",
            "variant": "torch-stable-abi29-cu130-x86_64-linux",
            "entry": "flash_attn_func",
            "kwargs": {
                "causal": False,
                "q_descale": None,
                "k_descale": None,
                "v_descale": None,
                "num_splits": 1,
                "deterministic": False,
            },
            "scheme": (
                "bf16 q/k/v with no descale (the bf16 path, not fp8); fp32 softmax and PV "
                "accumulation; full non-causal attention, no window, no softcap"
            ),
        },
    },
}


def fixture(store: Any, encoding: str) -> tuple[str, int]:
    """A five-component AdaLN-pruned checkpoint whose plan rows are stored as `encoding`."""
    sections = parse_production_config(job._asset("model-config.json"))
    tables = job._table_additions(sections)
    plan = prepare_quantization(h3_quantization_plan())
    config = dual_adaln_pruned_config(
        sections, job._production_plan("fl2va"), job._production_plan("ref2va")
    )
    plain = {
        "logical_dtype": "bf16",
        "shape": [2],
        "encoding": job.PLAIN_SPEC,
        "parts": {"value": {"dtype": "bf16", "shape": [2]}},
    }
    targets: dict[str, Any] = {}
    values: dict[tuple[str, str, str], bytes] = {}
    order: list[tuple[str, str]] = []
    for component in full_targets():
        add: dict[str, Any] = {}
        task = next((t for t, c in TARGET_COMPONENT.items() if c == component), None)
        if task is None:
            add[f"{component}.w"] = plain
            values[component, f"{component}.w", "value"] = b"\x00\x3f\x80\x3f"
        else:
            for key in tables[task]:
                add[key] = plain
                values[component, key, "value"] = b"\x00\x3f\x80\x3f"
            for tensor in plan.tensors:
                if encoding == "fp8-rowwise/1":
                    add[tensor.key] = {
                        "logical_dtype": "bf16",
                        "shape": [2, 4],
                        "encoding": job.FP8_SPEC,
                        "parts": {
                            "data": {"dtype": "f8_e4m3fn", "shape": [2, 4]},
                            "scale": {"dtype": "f32", "shape": [2]},
                        },
                    }
                    values[component, tensor.key, "data"] = bytes(range(8))
                    values[component, tensor.key, "scale"] = struct.pack("<2f", 1, 2)
                else:
                    add[tensor.key] = plain
                    values[component, tensor.key, "value"] = b"\x00\x3f\x80\x3f"
        order += [(component, key) for key in add]
        targets[component] = {"drop": [], "add": add}
    transaction = tensorfs.object_id(f"attention-lane-source-{encoding}".encode())
    writer = store.begin_derived(
        transaction,
        1,
        {},
        targets,
        {"model": {"kind": "add"}},
        order,
        1 << 20,
        work_fingerprint=tensorfs.object_id(b"attention-lane fixture"),
    )
    for (component, key, role), data in values.items():
        writer.add_part(component, key, role, io.BytesIO(data))
    writer.add_config("model", io.BytesIO(config))
    receipt = writer.commit()
    store.derived_adopt(transaction, f"source-{encoding}")
    return "sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"]


def invoke(
    store: Any, root: Path, source: str, length: int, request_id: str
) -> tuple[job.AttentionLaneResult, dict[str, dict[str, Any]]]:
    bounds = {lane: job.MAX_FULL_BYTES for lane in job.LANE_CONTRACT}
    model = _derive_model(H3FullTransformer, source)
    host = WeightsTransactionHost(
        store=store,
        owner_scope="attention-lane-proof",
        request_id=request_id,
        invocation_spec_digest=tensorfs.object_id(b"attention-lane invocation"),
        work_fingerprint=tensorfs.object_id(b"attention-lane work"),
        writer_session_id=1,
        allowed_sources={source: length},
        output_bounds=bounds,
    )
    attempt = fake_attempt(request_id, spool=root / f"spool-{request_id}")
    sink = WeightsSink(attempt, {"pruned": model}, bounds, host.open, host.structure)
    result = job.attention_lane(job.ProductionRequest(), model, sink)
    facts = {lane: store.derived_lookup(host.transaction_id(lane))["receipt"] for lane in bounds}
    return result, facts


def refusal(run: Any) -> str:
    try:
        run()
    except UnsupportedInput as exc:
        return exc.code
    raise AssertionError("attention-lane accepted a source it must refuse")


def header(store: Any, manifest: str) -> Any:
    return tensorfs.parse_header(bytes(store.manifest(manifest)["header"]))


def main() -> None:
    assets = {name: job._asset(f"execution.{name}.json") for name in DESIGN}
    for name, document in DESIGN.items():
        assert assets[name] == canonical_json.encode(document), f"{name} is not the design bytes"
    with tempfile.TemporaryDirectory(prefix="h3-attention-lane-proof-") as temporary:
        root = Path(temporary)
        store = tensorfs.Store.ensure(root / "store")
        source, length = fixture(store, "fp8-rowwise/1")
        result, facts = invoke(store, root, source, length, "attn8-1")
        before = header(store, source)
        inherited = {name: bytes(raw) for name, raw in before["configs"].items()}
        dit_rows = len(before["components"]["fl2va_dit"])
        assert result.inherited_tensors == dit_rows * 2 + 3
        produced: dict[str, str] = {}
        contracts: dict[str, str] = {}
        for lane in result.lanes:
            asset = assets[lane.contract]
            receipt = facts[lane.lane]
            manifest = "sha256:" + receipt["manifest"]["sha256"]
            after = header(store, manifest)
            assert after["components"] == before["components"], f"{lane.lane} changed a tensor"
            configs = {name: bytes(raw) for name, raw in after["configs"].items()}
            assert configs == {**inherited, "execution": asset}, f"{lane.lane} is not source + 1"
            contract = read_contract(configs)
            expected = read_contract({"execution": canonical_json.encode(DESIGN[lane.contract])})
            assert contract is not None and expected is not None
            assert contract.document() == expected.document()
            assert contract.digest() == expected.digest()
            assert contract.weights.route == "encoded_gemm" and contract.device == "sm90"
            assert lane.execution_config_digest == canonical_json.digest_bytes(asset)
            assert not lane.replayed
            assert set(receipt["inherit_observation"].values()) == {0}, receipt
            produced[lane.lane] = manifest
            contracts[lane.lane] = contract.digest()
        assert sorted(produced) == sorted(job.LANE_CONTRACT)
        assert len(set(produced.values())) == len(produced), "the lanes share one manifest"
        replayed, again = invoke(store, root, source, length, "attn8-1")
        assert all(lane.replayed for lane in replayed.lanes)
        assert {lane: "sha256:" + row["manifest"]["sha256"] for lane, row in again.items()} == (
            produced
        )
        bound = refusal(
            lambda: invoke(
                store,
                root,
                produced["fp8-attn8-adaln-pruned"],
                facts["fp8-attn8-adaln-pruned"]["manifest"]["length"],
                "attn8-2",
            )
        )
        bf16, bf16_length = fixture(store, "plain/1")
        plain = refusal(lambda: invoke(store, root, bf16, bf16_length, "attn8-3"))
        assert (bound, plain) == ("attention_lane_source", "attention_lane_source")
        print(
            json.dumps(
                {
                    "source": source,
                    "produced": produced,
                    "inherited_tensors": result.inherited_tensors,
                    "execution_config_digests": {
                        lane.lane: lane.execution_config_digest for lane in result.lanes
                    },
                    "execution_contract_digests": contracts,
                    "refusals": {"already_bound": bound, "bf16_source": plain},
                }
            )
        )


if __name__ == "__main__":
    main()
