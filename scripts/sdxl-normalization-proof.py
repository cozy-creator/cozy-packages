#!/usr/bin/env python3
"""Check the real constructor mapping, bounded native byte transforms and process restart."""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tensorfs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl"))

from cozy_runtime.author import (  # noqa: E402
    UnsupportedInput,
    WeightsReceipt,
    WeightsSink,
    WeightsTransaction,
    canonical_json,
)
from cozy_runtime.author._model import _derive_model  # noqa: E402
from cozy_runtime.author.fakes import fake_attempt, fake_context, fake_telemetry  # noqa: E402
from cozy_runtime.derive.quantization import QuantizationSource  # noqa: E402
from cozy_runtime.internal.weights_sink import (  # noqa: E402
    WeightsTransactionHost,
    protocol_receipt,
)

from sdxl import normalization as norm  # noqa: E402
from sdxl_normalization_plan import routes_for  # noqa: E402


def tiny_plan() -> norm.NormalizationPlan:
    tensor = norm.SourceTensor
    route = norm.TensorRoute
    return norm.NormalizationPlan(
        plain=norm.PLAN.plain,
        source={
            "unet": {"raw.weight": tensor("f16", (4,))},
            "vae": {"raw.attention": tensor("f16", (2, 2, 1, 1))},
            "text_encoder": {
                "embedding": tensor("f16", (2, 2)),
                "text_model.embeddings.position_ids": tensor("f16", (1, 4)),
            },
            "text_encoder_2": {
                "qkv": tensor("f16", (6, 2)),
                "projection": tensor("f16", (2, 3)),
                "logit_scale": tensor("f16", ()),
            },
        },
        targets=(
            route("unet", "weight", "raw.weight", (4,), "graft", 0),
            route("text_encoder", "embedding", "embedding", (2, 2), "graft", 0),
            route("vae", "attention", "raw.attention", (2, 2), "read", 0),
            route("text_encoder_2", "q", "qkv", (2, 2), "read", 0),
            route("text_encoder_2", "k", "qkv", (2, 2), "read", 4),
            route("text_encoder_2", "v", "qkv", (2, 2), "read", 8),
            route("text_encoder_2", "projection", "projection", (3, 2), "transpose", 0),
        ),
        configs={"model": {"fixture": True}},
    )


def source(
    store: Any, plan: norm.NormalizationPlan, *, bad_positions: bool = False
) -> tuple[str, int, dict[str, bytes]]:
    targets: dict[str, Any] = {}
    values: dict[str, bytes] = {}
    order = []
    for component, rows in plan.source.items():
        additions = {}
        for key, spec in rows.items():
            size = math.prod(spec.shape)
            if key == "text_model.embeddings.position_ids":
                positions = np.arange(size, dtype="<f2" if spec.dtype == "f16" else "<i8")
                value = (positions[::-1] if bad_positions else positions).tobytes()
            else:
                # Include signed zero and a NaN payload: normalization must preserve bits.
                value = (np.arange(size, dtype="<u2") + 0x7DFA).tobytes()
            values[component + "/" + key] = value
            additions[key] = {
                "logical_dtype": spec.dtype,
                "shape": list(spec.shape),
                "encoding": plan.plain,
                "parts": {"value": {"dtype": spec.dtype, "shape": list(spec.shape)}},
            }
            order.append((component, key))
        targets[component] = {"drop": [], "add": additions}
    identity = tensorfs.object_id(b"normalization-source")
    writer = store.begin_derived(
        identity,
        1,
        {},
        targets,
        {},
        order,
        4096,
        work_fingerprint=tensorfs.object_id(b"normalization source bytes"),
    )
    for identity_key, value in values.items():
        component, key = identity_key.split("/", 1)
        writer.add_part(component, key, "value", io.BytesIO(value))
    receipt = writer.commit()
    store.derived_adopt(identity, "normalization-source")
    return "sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"], values


def child(root: Path, mode: str) -> None:
    data = json.loads((root / "input.json").read_text())
    store = tensorfs.Store.open(root / "store")
    plan = tiny_plan()
    model = _derive_model(QuantizationSource, data["manifest"])
    receipts: list[Any] = []
    reads: list[str] = []

    def checkpoint(_row: Any) -> None:
        if mode == "interrupt":
            os._exit(73)  # No Python cleanup; the native checkpoint is already durable.

    host = WeightsTransactionHost(
        store=store,
        owner_scope="normalization-proof",
        request_id="normalize",
        invocation_spec_digest=tensorfs.object_id(b"normalization invocation"),
        work_fingerprint=tensorfs.object_id(b"normalization implementation"),
        writer_session_id={"interrupt": 1, "resume": 2, "replay": 3}[mode],
        allowed_sources={data["manifest"]: data["length"]},
        output_bounds={"model": norm.MAX_NEW_BYTES},
        record_checkpoint=checkpoint,
        record_receipt=receipts.append,
    )
    transaction = host.transaction_id("model")
    prior = store.derived_lookup(transaction)
    if mode != "interrupt" and prior.get("state") == "open":
        store.derived_fence(transaction, prior["writer_session_id"])
    attempt = fake_attempt("normalize", spool=root / ("spool-" + mode))
    sink = WeightsSink(
        attempt, {"source": model}, {"model": norm.MAX_NEW_BYTES}, host.open, host.structure
    )
    original = norm._bytes

    def observe(
        transaction: WeightsTransaction, route: norm.TensorRoute, spec: norm.SourceTensor
    ) -> bytes:
        reads.append(route.component + "/" + route.key)
        return original(transaction, route, spec)

    norm._bytes = observe
    try:
        result = norm._normalize(model, sink, fake_context(), fake_telemetry(attempt), plan)
    finally:
        norm._bytes = original
    receipt = receipts[-1]
    (root / (mode + ".json")).write_text(
        json.dumps(
            {
                "manifest": result.manifest.digest,
                "length": result.manifest.length,
                "transaction": transaction,
                "replayed": receipt.replayed,
                "transformed_roles": reads,
            }
        )
    )


def component_equivalence(
    root: Path, store: Any, plan: norm.NormalizationPlan,
    manifest: str, length: int, reference: str,
) -> None:
    def sink(
        name: str, sources: dict[str, tuple[str, int]], maximum: int,
    ) -> tuple[WeightsSink, dict[str, QuantizationSource], Any]:
        host = WeightsTransactionHost(
            store=store, owner_scope="normalization-components", request_id=name,
            invocation_spec_digest=tensorfs.object_id(b"component normalization proof"),
            work_fingerprint=tensorfs.object_id(name.encode()), writer_session_id=1,
            allowed_sources={digest: size for digest, size in sources.values()},
            output_bounds={"model": maximum},
        )
        models = {
            key: _derive_model(QuantizationSource, value[0]) for key, value in sources.items()
        }
        attempt = fake_attempt(name, spool=root / ("spool-" + name))
        weights = WeightsSink(
            attempt, models, {"model": maximum}, host.open, host.structure,
            source_config=host.config,
        )
        return weights, models, attempt

    parts: dict[str, tuple[str, int]] = {}
    for component in norm._COMPONENTS:
        weights, models, attempt = sink(
            "component-" + component, {"source": (manifest, length)}, norm.MAX_NEW_BYTES,
        )
        result = norm._normalize(
            models["source"], weights, fake_context(), fake_telemetry(attempt),
            norm._component_plan(plan, component),
        )
        parts[component] = (result.manifest.digest, result.manifest.length)
    weights, models, _ = sink("component-assembly", parts, 0)
    result = norm._assemble_normalized(models, weights, fake_context(), plan)
    assert result.manifest.digest == reference

    # A valid component artifact cannot be accepted under another component slot.
    wrong = {**parts, "vae": parts["unet"]}
    weights, models, _ = sink("component-wrong-slot", wrong, 0)
    try:
        norm._assemble_normalized(models, weights, fake_context(), plan)
    except UnsupportedInput:
        pass
    else:
        raise AssertionError("assembly accepted a component in the wrong slot")


def full_metadata_receipt_limits() -> None:
    """The real structural plan must fit both native intent and protocol receipt.

    Commit identities below are bounded metadata placeholders. Real native commit
    and exact output equality are exercised separately by the small byte fixture.
    """
    ref = {"sha256": "0" * 64, "length": 164}
    with tempfile.TemporaryDirectory(prefix="sdxl-receipt-metadata-") as temporary:
        store = tensorfs.Store.init(Path(temporary) / "store")
        for component in norm._COMPONENTS:
            plan = norm._component_plan(norm.PLAN, component)
            targets = norm._targets(plan)
            declaration = store.derived_declaration(
                {"source": ("sha256:" + "0" * 64, 164)},
                {name: WeightsTransactionHost._target(target) for name, target in targets.items()},
                {name: {"kind": "add"} for name in plan.configs},
                [(route.component, route.key) for route in plan.targets],
                norm.MAX_NEW_BYTES, work_fingerprint="sha256:" + "1" * 64,
            )
            assert len(declaration) <= 1 << 20
            # Each bounded non-graft role can add at most one16MiB object; treating
            # inline roles as objects overestimates the receipt rather than hiding it.
            added = [
                {"sha256": f"{index:064x}", "length": math.prod(route.shape) * 2}
                for index, route in enumerate(plan.targets)
                if route.kind != "graft"
            ]
            native = canonical_json.encode({
                "added_objects": added, "declaration": canonical_json.decode(declaration),
                "header": ref, "manifest": ref,
                "inherit_observation": {
                    "bytes": 6937666560, "hashes": 0, "objects": 2641, "reads": 0,
                },
                "sources": [{"alias": "source", "components": [component],
                             "header": ref, "manifest": ref}],
                "transaction_id": "sha256:" + "2" * 64,
            })
            _, wrapped, _ = protocol_receipt(
                WeightsReceipt(
                    "model", "sha256:" + "2" * 64,
                    canonical_json.digest(canonical_json.decode(native)), native,
                ),
                owner_scope="cozy-local-client", request_id="job-" + "3" * 24,
                invocation_spec_digest="sha256:" + "4" * 64,
            )
            assert len(wrapped) <= 1 << 20, (component, len(declaration), len(wrapped))


def main() -> None:
    full_metadata_receipt_limits()
    assert routes_for(norm.PLAN) == norm.PLAN.targets
    assert len(norm.PLAN.targets) == 2641
    assert norm.PLAN.plain == next(d for alias, d in tensorfs.seed_digests() if alias == "plain/1")
    for dtype in (np.dtype("<f2"), np.dtype("<i8")):
        valid_positions = np.arange(77, dtype=dtype)
        norm._validate_position_ids(valid_positions.tobytes(), dtype, 77)
        for bad_positions in (valid_positions[::-1], np.full(77, -1, dtype=dtype)):
            try:
                norm._validate_position_ids(bad_positions.tobytes(), dtype, 77)
            except UnsupportedInput:
                pass
            else:
                raise AssertionError("changed position IDs accepted")
    for value in (0.5, float("nan"), float("inf")):
        bad_positions = np.arange(77, dtype="<f2")
        bad_positions[1] = value
        try:
            norm._validate_position_ids(bad_positions.tobytes(), np.dtype("<f2"), 77)
        except UnsupportedInput:
            pass
        else:
            raise AssertionError("noninteger or nonfinite position ID accepted")
    with tempfile.TemporaryDirectory(prefix="sdxl-normalization-") as temporary:
        root = Path(temporary)
        store = tensorfs.Store.init(root / "store")
        plan = tiny_plan()
        manifest, length, values = source(store, plan)
        (root / "input.json").write_text(json.dumps({"manifest": manifest, "length": length}))
        for mode in ("interrupt", "resume", "replay"):
            result = subprocess.run(
                [sys.executable, __file__, "child", str(root), mode], capture_output=True, text=True
            )
            assert result.returncode == (73 if mode == "interrupt" else 0), result.stderr
        resumed = json.loads((root / "resume.json").read_text())
        replayed = json.loads((root / "replay.json").read_text())
        assert resumed["manifest"] == replayed["manifest"]
        component_equivalence(root, store, plan, manifest, length, resumed["manifest"])
        assert len(resumed["transformed_roles"]) == 4
        assert "vae/attention" not in resumed["transformed_roles"]
        assert replayed["replayed"] and replayed["transformed_roles"] == []
        produced_header = store.manifest(resumed["manifest"])["header"]
        assert produced_header is not None, "normalized model has no CozyTensors header"
        produced = tensorfs.parse_header(produced_header)
        expected_order = [(r.component, r.key) for r in plan.targets]
        assert [
            (c, k) for c, rows in produced["components"].items() for k in rows
        ] == expected_order
        assert produced["configs"]["model"] == canonical_json.encode(plan.configs["model"])
        original_header = store.manifest(manifest)["header"]
        assert original_header is not None, "model fixture has no CozyTensors header"
        original = tensorfs.parse_header(original_header)
        assert (
            produced["components"]["unet"]["weight"] == original["components"]["unet"]["raw.weight"]
        )
        probe_id = tensorfs.object_id(b"normalization readback")
        probe = store.begin_derived(
            probe_id,
            1,
            {"result": (resumed["manifest"], resumed["length"])},
            {
                c: {"source": "result", "source_component": c, "drop": [], "add": {}}
                for c in produced["components"]
            },
            {},
            expected_order,
            0,
            work_fingerprint=tensorfs.object_id(b"normalization readback"),
        )
        for route in plan.targets:
            raw = values[route.component + "/" + route.source_key]
            expected = raw
            if route.kind == "read":
                expected = raw[route.offset * 2 : route.offset * 2 + math.prod(route.shape) * 2]
            elif route.kind == "transpose":
                expected = np.frombuffer(raw, dtype="<u2").reshape(2, 3).T.copy().tobytes()
            actual = bytearray(len(expected))
            probe.source_read_into("result", route.component, route.key, "value", 0, actual)
            assert actual == expected, route.key
        probe.fence()
        store.derived_abandon(probe_id)
        host = WeightsTransactionHost(
            store=store,
            owner_scope="normalization-proof",
            request_id="refusal",
            invocation_spec_digest=tensorfs.object_id(b"refusal invocation"),
            work_fingerprint=tensorfs.object_id(b"refusal implementation"),
            writer_session_id=1,
            allowed_sources={manifest: length},
            output_bounds={"model": norm.MAX_NEW_BYTES},
        )
        observed = host.structure(manifest)
        for bad in (
            replace(observed, tensors=observed.tensors[:-1]),
            replace(observed, configs=("unexpected",)),
            replace(
                observed,
                tensors=(replace(observed.tensors[0], encoding="wrong"), *observed.tensors[1:]),
            ),
        ):
            try:
                norm._validate(bad, plan)
            except UnsupportedInput:
                pass
            else:
                raise AssertionError("unreviewed source entered normalization")
        bad_store = tensorfs.Store.init(root / "bad-positions")
        bad_manifest, bad_length, _ = source(bad_store, plan, bad_positions=True)
        bad_model = _derive_model(QuantizationSource, bad_manifest)
        bad_host = WeightsTransactionHost(
            store=bad_store,
            owner_scope="normalization-proof",
            request_id="bad-positions",
            invocation_spec_digest=tensorfs.object_id(b"bad positions invocation"),
            work_fingerprint=tensorfs.object_id(b"bad positions implementation"),
            writer_session_id=1,
            allowed_sources={bad_manifest: bad_length},
            output_bounds={"model": norm.MAX_NEW_BYTES},
        )
        bad_attempt = fake_attempt("bad-positions", spool=root / "bad-spool")
        bad_sink = WeightsSink(
            bad_attempt,
            {"source": bad_model},
            {"model": norm.MAX_NEW_BYTES},
            bad_host.open,
            bad_host.structure,
        )
        try:
            norm._normalize(bad_model, bad_sink, fake_context(), fake_telemetry(bad_attempt), plan)
        except UnsupportedInput as error:
            assert "position IDs" in str(error)
        else:
            raise AssertionError("changed position-ID values were discarded")
        assert bad_store.derived_lookup(bad_host.transaction_id("model"))["state"] != "committed"
    print(
        json.dumps(
            {
                "constructor_destinations": 2641,
                "real_process_exit": 73,
                "completed_role_not_repeated": True,
                "replayed_role_reads": 0,
                "graft_identity_preserved": True,
                    "component_assembly_equals_monolithic": True,
                    "wrong_component_slot_refused": True,
                "split_transpose_reshape_bits_exact": True,
                "unreviewed_sources_refused": True,
                "changed_position_ids_refused": True,
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(Path(sys.argv[2]), sys.argv[3])
    else:
        main()
