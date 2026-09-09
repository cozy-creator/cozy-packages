#!/usr/bin/env python3
"""Both table sets through the real Runtime WeightsSink and a real TensorFS store, on CPU.

    minimax-h3-tools/.venv/bin/python scripts/h3-turbo-store-proof.py

A tiny H3-shaped `full`, an AdaLN-pruned `pruned` and two PDD-shaped adapters are minted in
a temporary store; `job._retable` — the exact orchestration the `retable` job runs — derives
the four outputs from them. The committed headers then prove what the tables are and what
moved: the turbo rows equal the fused kernel run directly over the same values, the launch rows
equal the unfused kernel, the adapter slice rides the turbo bank as the adapter's own stored
objects, and every non-table row of both checkpoints is `pruned`'s own object.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import tensorfs
import torch
from cozy_runtime.author import WeightsSink
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_context, fake_telemetry
from cozy_runtime.internal.weights_sink import WeightsTransactionHost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
from h3_tables import job  # noqa: E402
from h3_tables.kernel import (  # noqa: E402
    H3Topology,
    LowRankAdapter,
    adapter_shapes,
    precompute_tables,
    source_shapes,
    table_bytes,
    table_shapes,
)
from h3_tables.plans import Task  # noqa: E402
from h3_tables.source import (  # noqa: E402
    ADAPTER_ALPHA,
    ADAPTER_RANK,
    H3FullTransformer,
    H3TurboAdapter,
)

TINY = H3Topology(8, 3, 8, 12, 4)
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")
SLOT_BYTES = 1 << 24
SHARED = ("text_encoder", "video_vae", "audio_vae")


def fail(what: str) -> None:
    raise SystemExit(f"h3-turbo-store-proof: {what}")


def plain_spec() -> str:
    return next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")


def mint(store: Any, suffix: str, values: dict[str, dict[str, torch.Tensor]]) -> tuple[str, int]:
    plain = plain_spec()
    targets = {}
    for component, rows in values.items():
        add = {}
        for key, value in rows.items():
            dtype = "f32" if value.dtype == torch.float32 else "bf16"
            add[key] = {
                "logical_dtype": dtype,
                "shape": list(value.shape),
                "encoding": plain,
                "parts": {"value": {"dtype": dtype, "shape": list(value.shape)}},
            }
        targets[component] = {"drop": [], "add": add}
    order = [(component, key) for component, rows in values.items() for key in rows]
    writer = store.begin_derived(
        "sha256:" + suffix * 32,
        1,
        {},
        targets,
        {},
        order,
        SLOT_BYTES,
        work_fingerprint="sha256:" + "50" * 32,
    )
    for component, rows in values.items():
        for key, value in rows.items():
            writer.add_part(
                component,
                key,
                "value",
                io.BytesIO(value.contiguous().view(torch.uint8).numpy().tobytes()),
            )
    manifest = writer.commit()["manifest"]
    return "sha256:" + manifest["sha256"], manifest["length"]


def header(store: Any, manifest: str) -> Any:
    return tensorfs.parse_header(bytes(store.manifest(manifest)["header"]))


def bodies(store: Any, manifest: str) -> dict[tuple[str, str], str]:
    return {
        (component, key): json.dumps(
            tensor["parts"]["value"].get("segments", tensor["parts"]["value"].get("inline")),
            sort_keys=True,
            default=str,
        )
        for component, tensors in header(store, manifest)["components"].items()
        for key, tensor in tensors.items()
    }


def role_bytes(store: Any, head: Any, component: str, key: str) -> bytes:
    part = head["components"][component][key]["parts"]["value"]
    if "inline" in part:
        return bytes(part["inline"])
    return b"".join(
        bytes(store.document("sha256:" + segment["sha256"], segment["length"]))
        for segment in part["segments"]
    )


def fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    torch.manual_seed(2026)
    shared = {component: {f"{component}.w": torch.randn(3)} for component in SHARED}
    launch = job._plans(job.LAUNCH_SET)
    full: dict[str, Any] = dict(shared)
    pruned: dict[str, Any] = dict(shared)
    adapters: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        component = f"{task}_dit"
        full[component] = {"proj_in.weight": torch.randn(2, 2).bfloat16()}
        full[component].update(
            {
                key: (torch.randn(shape) * 0.05).to(dtype)
                for key, (dtype, shape) in source_shapes(TINY).items()
            }
        )
        pruned[component] = {"proj_in.weight": full[component]["proj_in.weight"]}
        pruned[component].update(
            {
                key: torch.zeros(shape, dtype=torch.bfloat16)
                for key, shape in table_shapes(TINY, launch[task]).items()
            }
        )
        rows = {"proj_out.weight": torch.randn(2, 8, 8).bfloat16()}
        rows.update(
            {
                key: (torch.randn(shape) * 0.2).to(dtype)
                for key, (dtype, shape) in adapter_shapes(TINY, ADAPTER_RANK).items()
            }
        )
        rows["transformer_blocks.0.attn.to_q.lora_down"] = torch.randn(ADAPTER_RANK, 8).bfloat16()
        adapters[task] = {"model": rows}
    return full, pruned, adapters


def direct_tables(
    full: dict[str, Any], adapter: dict[str, Any] | None, task: Task, plan: Any
) -> dict[str, torch.Tensor]:
    component = f"{task}_dit"
    out: dict[str, torch.Tensor] = {}
    precompute_tables(
        plan=plan,
        topology=TINY,
        read=lambda key, _d, _s: full[component][key],
        write=lambda key, value: out.__setitem__(key, value.clone()),
        progress=lambda _done, _total: None,
        device=torch.device("cpu"),
        adapter=(
            LowRankAdapter(
                ADAPTER_RANK,
                ADAPTER_ALPHA / ADAPTER_RANK,
                lambda key, _d, _s: adapter["model"][key],
            )
            if adapter is not None
            else None
        ),
    )
    return out


def main() -> None:
    full_values, pruned_values, adapter_values = fixtures()
    plans = {table_set.name: job._plans(table_set) for table_set in job.TABLE_SETS}
    with tempfile.TemporaryDirectory(prefix="h3-turbo-store-proof-") as root:
        store = tensorfs.Store.init(root)
        full_id, full_len = mint(store, "61", full_values)
        pruned_id, pruned_len = mint(store, "62", pruned_values)
        adapter_ids = {
            task: mint(store, suffix, adapter_values[task])
            for task, suffix in zip(TASKS, ("63", "64"), strict=True)
        }
        full_model = _derive_model(H3FullTransformer, full_id)
        pruned_model = _derive_model(H3FullTransformer, pruned_id)
        adapter_models: dict[Task, H3TurboAdapter] = {
            task: _derive_model(H3TurboAdapter, adapter_ids[task][0]) for task in TASKS
        }
        models = {
            "full": full_model,
            "pruned": pruned_model,
            "fl2va_adapter": adapter_models["fl2va"],
            "ref2va_adapter": adapter_models["ref2va"],
        }
        slots = {output.name: SLOT_BYTES for output in job.RETABLE_OUTPUTS}
        allowed = {full_id: full_len, pruned_id: pruned_len}
        allowed.update({identity: length for identity, length in adapter_ids.values()})
        attempt = fake_attempt("h3-turbo-store-proof", spool=Path(root) / "spool")
        ctx = fake_context()
        tel = fake_telemetry(attempt, ctx)
        pruned_order = tuple(
            (component, key) for component, rows in pruned_values.items() for key in rows
        )
        work = job.RetableWork(
            topologies={task: TINY for task in TASKS},
            plans=plans,
            configs={name: json.dumps({"table_set": name}).encode() for name in plans},
            order=pruned_order,
            device=torch.device("cpu"),
        )
        receipts: list[Any] = []

        def run(session: int) -> Any:
            host = WeightsTransactionHost(
                store=store,
                owner_scope="h3-turbo-store-proof",
                request_id="turbo-store-proof",
                invocation_spec_digest="sha256:" + "11" * 32,
                work_fingerprint="sha256:" + "22" * 32,
                writer_session_id=session,
                allowed_sources=allowed,
                output_bounds=slots,
                record_receipt=receipts.append,
            )
            sink = WeightsSink(attempt, models, slots, host.open, host.structure)
            return job._retable(
                ctx,
                tel,
                sink,
                work,
                full=full_model,
                pruned=pruned_model,
                adapters=adapter_models,
            )

        result = run(1)
        by_name = {row.table_set: row for row in result.table_sets}
        per_task = {name: table_bytes(TINY, plans[name]["fl2va"]) for name in plans}
        modulation = sum(
            value.numel() * value.element_size()
            for key, value in full_values["fl2va_dit"].items()
            if key != "proj_in.weight"
        )
        slice_bytes = sum(
            value.numel() * value.element_size()
            for key, value in adapter_values["fl2va"]["model"].items()
            if key in adapter_shapes(TINY, ADAPTER_RANK)
        )
        if (
            any(row.replayed for row in result.table_sets)
            or by_name["launch"].table_bytes_this_run != 2 * per_task["launch"]
            or by_name["turbo"].table_bytes_this_run != 2 * per_task["turbo"]
            or result.source_bytes_read_this_run != 4 * modulation + 2 * slice_bytes
        ):
            fail(f"first attempt result is {result}")
        print(
            f"  computed: launch {by_name['launch'].table_bytes_this_run} B, turbo "
            f"{by_name['turbo'].table_bytes_this_run} B of tables; read "
            f"{result.source_bytes_read_this_run} B"
        )

        manifests = {
            receipt.output_slot: "sha256:"
            + json.loads(receipt.tensorfs_receipt)["manifest"]["sha256"]
            for receipt in receipts
        }
        if set(manifests) != set(slots):
            fail(f"committed {sorted(manifests)}")
        pruned_bodies = bodies(store, pruned_id)
        adapter_bodies = {task: bodies(store, adapter_ids[task][0]) for task in TASKS}
        for table_set in job.TABLE_SETS:
            head = header(store, manifests[table_set.checkpoint])
            checkpoint_bodies = bodies(store, manifests[table_set.checkpoint])
            if set(head["components"]) != {*SHARED, "fl2va_dit", "ref2va_dit"}:
                fail(f"{table_set.checkpoint} components are {sorted(head['components'])}")
            for identity, body in pruned_bodies.items():
                if identity[1].endswith(".table"):
                    continue
                if checkpoint_bodies.get(identity) != body:
                    fail(f"{table_set.checkpoint} rewrote the inherited row {identity}")
            bank_head = header(store, manifests[table_set.bank])
            wanted = {"fl2va_dit", "ref2va_dit"}
            if table_set.adapted:
                wanted |= {"fl2va_adapter", "ref2va_adapter"}
            if set(bank_head["components"]) != wanted:
                fail(f"{table_set.bank} components are {sorted(bank_head['components'])}")
            for task in TASKS:
                plan = plans[table_set.name][task]
                expected = direct_tables(
                    full_values, adapter_values[task] if table_set.adapted else None, task, plan
                )
                component = f"{task}_dit"
                if set(bank_head["components"][component]) != set(expected):
                    fail(f"{table_set.bank} {component} does not hold exactly the table rows")
                for key, value in expected.items():
                    for manifest_head in (head, bank_head):
                        raw = bytearray(role_bytes(store, manifest_head, component, key))
                        stored = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(value.shape)
                        if not torch.equal(stored, value):
                            fail(f"{table_set.name} {component}/{key} bytes differ from the kernel")
                if table_set.adapted:
                    alias = f"{task}_adapter"
                    slice_keys = set(adapter_shapes(TINY, ADAPTER_RANK))
                    carried = sorted(bank_head["components"][alias])
                    if set(carried) != slice_keys:
                        fail(f"{table_set.bank} {alias} carries {carried}")
                    bank_bodies = bodies(store, manifests[table_set.bank])
                    for key in slice_keys:
                        if bank_bodies[(alias, key)] != adapter_bodies[task][("model", key)]:
                            fail(f"{alias}/{key} is not the adapter's own stored object")
            print(
                f"  {table_set.name}: checkpoint inherits every non-table row of pruned by object; "
                f"bank holds {sorted(bank_head['components'])}; tables equal the direct kernel"
                + (" with the adapter fused" if table_set.adapted else "")
            )

        replay = run(2)
        if not all(row.replayed for row in replay.table_sets) or replay.source_bytes_read_this_run:
            fail(f"replay result is {replay}")
        if {row.table_set: row.tensorfs_receipt_digest for row in replay.table_sets} != {
            row.table_set: row.tensorfs_receipt_digest for row in result.table_sets
        }:
            fail("replay returned different receipts")
        print("  replay: all four outputs replayed, zero bytes read")
    print("H3 TURBO STORE PROOF PASS sets=2 outputs=4 fused=reference-exact inherited=by-object")


if __name__ == "__main__":
    main()
