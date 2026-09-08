#!/usr/bin/env python3
"""Run H3 row repair through real Runtime and TensorFS, including interrupted resume."""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import tensorfs
from cozy_runtime.author import Model, WeightsSink
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.internal.weights_sink import WeightsTransactionHost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "client-scripts"))
import h3_checkpoint_repair as repair  # noqa: E402

ENCODINGS = repair.ENCODINGS


class Checkpoint(Model[object]):
    def load(self, loader: Any) -> None:
        del loader


def fixture(
    store: Any, variant: str
) -> tuple[str, int, dict[tuple[str, str, str], bytes], list[tuple[str, str]]]:
    targets: dict[str, Any] = {}
    values: dict[tuple[str, str, str], bytes] = {}
    order: list[tuple[str, str]] = []
    keys = sorted(repair.FC1_KEYS)
    keys = keys[1:] + keys[:1]  # Deliberately nonlexical within each component.
    for component in reversed(repair.COMPONENTS):
        add: dict[str, Any] = {}
        for i, key in enumerate(keys):
            # Mixed quantized/plain rows prove selection is per tensor, not per lane.
            plain = variant == "plain" or i == 0
            if plain:
                parts = {"value": {"dtype": "bf16", "shape": [4, 32]}}
                payload = {"value": b"".join(bytes([row + 48]) * 64 for row in range(4))}
                encoding = ENCODINGS["plain"]
            else:
                parts = {"data": {"dtype": "f8_e4m3fn", "shape": [4, 32]}}
                payload = {"data": b"".join(bytes([row + 48]) * 32 for row in range(4))}
                if variant == "fp8":
                    parts["scale"] = {"dtype": "f32", "shape": [4]}
                    payload["scale"] = struct.pack("<4f", 1, 2, 4, 8)
                else:
                    parts["scale"] = {"dtype": "u8", "shape": [4, 1]}
                    payload["scale"] = bytes([126, 127, 128, 129])
                encoding = ENCODINGS[variant]
            add[key] = {
                "logical_dtype": "bf16",
                "shape": [4, 32],
                "encoding": encoding,
                "parts": parts,
            }
            for role, data in payload.items():
                values[component, key, role] = data
            order.append((component, key))
        add["untouched"] = {
            "logical_dtype": "bf16",
            "shape": [4],
            "encoding": ENCODINGS["plain"],
            "parts": {"value": {"dtype": "bf16", "shape": [4]}},
        }
        values[component, "untouched", "value"] = b"abcdefgh"
        order.append((component, "untouched"))
        targets[component] = {"drop": [], "add": add}
    transaction = tensorfs.object_id(("source-" + variant).encode())
    writer = store.begin_derived(
        transaction,
        1,
        {},
        targets,
        {"model": {"kind": "add"}},
        order,
        1 << 20,
        work_fingerprint=tensorfs.object_id(b"repair source fixture"),
    )
    for (component, key, role), data in values.items():
        writer.add_part(component, key, role, io.BytesIO(data))
    writer.add_config("model", io.BytesIO(b'{"preserve":"config"}'))
    receipt = writer.commit()
    store.derived_adopt(transaction, "source-" + variant)
    return "sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"], values, order


def exercise(root: Path, variant: str) -> None:
    store = tensorfs.Store.ensure(root)
    source, length, values, source_order = fixture(store, variant)
    original = tensorfs.parse_header(bytes(store.manifest(source)["header"]))
    model = _derive_model(Checkpoint, source)
    # Ordinary new roots are refused; explicitly admit this tiny fixture only in
    # this diagnostic process. Production keeps its four immutable source IDs.
    try:
        repair.repair(model, None, None, ENCODINGS)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown source entered the one-time migration")
    repair.SOURCES[source] = variant
    checkpoints: list[Any] = []

    def invoke(epoch: int, interrupt: bool = False) -> Any:
        receipts: list[Any] = []
        reads: list[int] = []
        read_swapped = repair._read_swapped

        def observed_read(
            transaction: Any, component: str, key: str, role: str, length: int
        ) -> bytearray:
            reads.append(length)
            return read_swapped(transaction, component, key, role, length)

        def checkpoint(row: Any) -> None:
            checkpoints.append(row)
            if interrupt:
                raise RuntimeError("injected interruption after a native checkpoint")

        host = WeightsTransactionHost(
            store=store,
            owner_scope="repair-proof",
            request_id="repair-" + variant,
            invocation_spec_digest=tensorfs.object_id(b"repair invocation"),
            work_fingerprint=tensorfs.object_id(("repair-work-" + variant).encode()),
            writer_session_id=epoch,
            allowed_sources={source: length},
            output_bounds={"checkpoint": repair.MAX_NEW_BYTES},
            record_checkpoint=checkpoint,
            record_receipt=receipts.append,
        )
        structure_order = [(row.component, row.key) for row in host.structure(source).tensors]
        assert structure_order == source_order, "source structure changed construction order"
        attempt = fake_attempt("repair-" + variant, spool=root / f"spool-{epoch}")
        sink = WeightsSink(
            attempt,
            {"source": model},
            {"checkpoint": repair.MAX_NEW_BYTES},
            host.open,
            host.structure,
        )
        repair._read_swapped = observed_read
        try:
            result = repair.main(source=model, artifacts=sink, tel=fake_telemetry(attempt))
        finally:
            repair._read_swapped = read_swapped
        return result, receipts[-1], reads

    try:
        invoke(2, interrupt=True)
    except Exception as error:
        assert "injected interruption" in str(error), error
    else:
        raise AssertionError("native checkpoint interruption did not fire")
    assert len(checkpoints) == 1
    resumed, resumed_receipt, resumed_reads = invoke(3)
    changed_roles = sum(key in repair.FC1_KEYS for _, key, _ in values)
    assert 0 < len(resumed_reads) < changed_roles, "completed parts were read again"
    replayed, replayed_receipt, replayed_reads = invoke(4)
    assert replayed_receipt.replayed and not replayed_reads
    assert replayed == resumed
    facts = store.derived_lookup(resumed_receipt.weights_transaction_id)["receipt"]
    manifest = "sha256:" + facts["manifest"]["sha256"]
    produced = tensorfs.parse_header(bytes(store.manifest(manifest)["header"]))
    assert produced["configs"] == original["configs"]
    before_plan = tensorfs.plan(bytes(store.manifest(source)["header"]), source_order)
    after_plan = tensorfs.plan(bytes(store.manifest(manifest)["header"]), source_order)
    assert before_plan.order == after_plan.order, "repair changed construction traversal"
    assert [
        (component, key) for component, rows in produced["components"].items() for key in rows
    ] == source_order
    for component in repair.COMPONENTS:
        assert (
            produced["components"][component]["untouched"]
            == original["components"][component]["untouched"]
        )
    order = [(component, key) for component, rows in produced["components"].items() for key in rows]
    probe_id = tensorfs.object_id(("inspect-" + variant).encode())
    probe = store.begin_derived(
        probe_id,
        1,
        {"fixed": (manifest, facts["manifest"]["length"])},
        {
            component: {"source": "fixed", "source_component": component, "drop": [], "add": {}}
            for component in produced["components"]
        },
        {},
        order,
        0,
        work_fingerprint=tensorfs.object_id(b"repair inspection"),
    )
    for (component, key, role), data in values.items():
        actual = bytearray(len(data))
        probe.source_read_into("fixed", component, key, role, 0, actual)
        half = len(data) // 2
        expected = data if key == "untouched" else data[half:] + data[:half]
        assert actual == expected, ("repair bytes mismatch", variant, component, key, role)
    probe.fence()
    store.derived_abandon(probe_id)
    # Corrected output cannot enter the migration and accidentally undo the fix.
    try:
        repair.repair(_derive_model(Checkpoint, manifest), None, None, ENCODINGS)
    except ValueError:
        pass
    else:
        raise AssertionError("corrected checkpoint could be swapped twice")
    print(
        json.dumps(
            {
                "encoding": variant,
                "repaired_tensors": 104,
                "replayed_parts": changed_roles - len(resumed_reads),
                "verified_roles": len(values),
                "replay_bytes_read": sum(replayed_reads),
                "unchanged_tensor_refs_and_configs": True,
                "nonlexical_order_preserved": True,
            }
        )
    )
    del repair.SOURCES[source]


def main() -> None:
    if "--red-identity" in sys.argv:

        def identity(
            transaction: Any, component: str, key: str, role: str, length: int
        ) -> bytearray:
            raw = bytearray(length)
            transaction.source_read_into("source", component, key, role, 0, raw)
            return raw

        repair._read_swapped = identity
        with tempfile.TemporaryDirectory(prefix="h3-repair-red-") as temporary:
            exercise(Path(temporary), "plain")
        raise AssertionError("identity reader unexpectedly passed the row-permutation oracle")
    negative = subprocess.run(
        [sys.executable, __file__, "--red-identity"], capture_output=True, text=True, check=False
    )
    assert negative.returncode != 0 and "repair bytes mismatch" in negative.stderr, negative.stderr
    print("identity-copy control REFUSED by exact native output comparison", flush=True)
    # Cross multiple read chunks even with tiny tensors; production retains32MiB reads.
    repair.READ_BYTES = 37
    assert importlib.util.find_spec("torch") is None and "torch" not in sys.modules
    assert set(ENCODINGS.values()) <= {digest for _, digest in tensorfs.seed_digests()}
    for name, width in repair.DTYPE_BYTES.items():
        assert tensorfs.dtypes()[name] == width
    with tempfile.TemporaryDirectory(prefix="h3-repair-proof-") as temporary:
        for variant in ENCODINGS:
            exercise(Path(temporary) / variant, variant)


if __name__ == "__main__":
    main()
