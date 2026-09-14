#!/usr/bin/env python3
"""Real CPU math and native checkpoints for interrupted H3 table computation."""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import msgspec
import tensorfs
import torch
from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    ModelArtifact,
    ObjectRef,
    Telemetry,
    WeightsOutput,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._model import _derive_model
from h3_tables.adaln_operations import PLAIN, _compute_into, _plan
from h3_tables.kernel import (
    H3Topology,
    precompute_tables,
    removed_keys,
    source_shapes,
    table_shapes,
)
from h3_tables.source import H3FullTransformer
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from native_execution_fixture import NativeExecution

TOPOLOGY = H3Topology(8, 3, 8, 12, 4)
PLAN = _plan("fl2va")
COMPONENT = "fl2va_dit"


@invocable(memoize=True)
async def mini_tables(ctx: Context, *, source: H3FullTransformer, tel: Telemetry) -> ModelArtifact:
    tables = table_shapes(TOPOLOGY, PLAN)
    with ctx.tensorfs_source(source) as native_source:
        selected = native_source.inspect().source
    with ctx.output("model").open(
        Derivation(
            sources={"source": selected},
            targets={
                COMPONENT: Target(
                    "source",
                    COMPONENT,
                    drop=removed_keys(TOPOLOGY),
                    add={
                        key: Tensor("bf16", shape, PLAIN, {"value": Part("bf16", shape)})
                        for key, shape in tables.items()
                    },
                )
            },
            configs={"adaln": Config("add")},
            order=tuple((COMPONENT, key) for key in tables),
        )
    ) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        _compute_into(ctx, tel, transaction, "fl2va", PLAN, TOPOLOGY)
        transaction.add_config("adaln", b"{}")
        return ctx.adopt_model(transaction.commit())


def main() -> None:
    torch.manual_seed(71)
    values = {
        key: (torch.randn(shape) * 0.03).to(dtype)
        for key, (dtype, shape) in source_shapes(TOPOLOGY).items()
    }
    full: dict[str, torch.Tensor] = {}
    reads: list[str] = []

    def read(key: str, _dtype: torch.dtype, _shape: tuple[int, ...]) -> torch.Tensor:
        reads.append(key)
        return values[key]

    precompute_tables(
        plan=PLAN,
        topology=TOPOLOGY,
        read=read,
        write=lambda key, value: full.__setitem__(key, value),
        progress=lambda _done, _total: None,
        device=torch.device("cpu"),
    )
    resumed: dict[str, torch.Tensor] = {}
    reads.clear()
    done = frozenset({"transformer_blocks.0.adaln_proj.table", "norm_out.table"})
    precompute_tables(
        plan=PLAN,
        topology=TOPOLOGY,
        read=read,
        write=lambda key, value: resumed.__setitem__(key, value),
        progress=lambda _done, _total: None,
        device=torch.device("cpu"),
        completed=done,
    )
    assert not any(key.startswith(("transformer_blocks.0.", "norm_out.")) for key in reads)
    assert set(resumed) == set(full) - done
    assert all(torch.equal(value, full[key]) for key, value in resumed.items())
    reads.clear()
    precompute_tables(
        plan=PLAN,
        topology=TOPOLOGY,
        read=read,
        write=lambda _key, _value: None,
        progress=lambda _done, _total: None,
        device=torch.device("cpu"),
        completed=frozenset(full),
    )
    assert reads == []

    app = App()
    app.job(mini_tables, weights=(WeightsOutput("model", 1 << 20),))
    describe(app)
    with tempfile.TemporaryDirectory(prefix="h3-adaln-native-resume-") as area:
        root = Path(area)
        store = tensorfs.Store.init(root / "store")
        source_id = "sha256:" + "41" * 32
        additions = {}
        for key, value in values.items():
            dtype = "f32" if value.dtype == torch.float32 else "bf16"
            additions[key] = {
                "logical_dtype": dtype,
                "shape": list(value.shape),
                "encoding": PLAIN,
                "parts": {"value": {"dtype": dtype, "shape": list(value.shape)}},
            }
        writer = store.begin_derived(
            source_id,
            1,
            {},
            {COMPONENT: {"drop": [], "add": additions}},
            {},
            [(COMPONENT, key) for key in values],
            1 << 20,
            work_fingerprint="sha256:" + "42" * 32,
        )
        for key, value in values.items():
            writer.add_part(
                COMPONENT,
                key,
                "value",
                io.BytesIO(value.contiguous().view(torch.uint8).numpy().tobytes()),
            )
        receipt = writer.commit()
        manifest = receipt["manifest"]
        source = ModelArtifact(
            "source",
            "model",
            ObjectRef("sha256:" + manifest["sha256"], manifest["length"]),
            "sha256:"
            + __import__("hashlib")
            .sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
            .hexdigest(),
        )

        def run(name: str, epoch: int, stop: bool) -> Any:
            with NativeExecution(
                store, root, name, {"source": source}, {"model": 1 << 20}, epoch=epoch
            ) as execution:
                return attempt(
                    app.get("mini_tables"),
                    {"source": msgspec.to_builtins(source)},
                    Invocation(
                        name,
                        execution.spool,
                        time.monotonic() + 60,
                        models={"source": _derive_model(H3FullTransformer, source.manifest.digest)},
                        tensorfs_output=execution.client.open_output,
                        tensorfs_source=execution.client.source,
                        tensorfs_adopt=execution.client.adopt_model,
                        cancel=lambda: stop and execution.checkpointed,
                    ),
                )

        first, outcome, _ = run("interrupted", 1, True)
        assert first is None and outcome.terminal == "canceled", outcome
        result, outcome, record = run("interrupted", 2, False)
        assert outcome.terminal == "succeeded" and result is not None, outcome
        clean, outcome, _ = run("clean", 1, False)
        assert outcome.terminal == "succeeded" and clean is not None, outcome
        assert result.result.manifest == clean.result.manifest
        metrics = {
            row["name"]: row["value"] for row in record.ring.rows() if row["kind"] == "metric"
        }
        assert metrics["h3.adaln.reused_tables"] == 1
        expected = sum(
            value.numel() * value.element_size()
            for key, value in values.items()
            if not key.startswith("transformer_blocks.0.")
        )
        assert metrics["h3.adaln.source_bytes"] == expected
        replay, outcome, _ = run("interrupted", 3, False)
        assert outcome.terminal == "succeeded" and replay.result == result.result, outcome
        print(
            json.dumps(
                {
                    "kernel_remaining_tables": len(resumed),
                    "completed_kernel_reads": 0,
                    "native_checkpoint_reused_tables": 1,
                    "resumed_manifest_equals_clean": True,
                    "resumed_source_bytes": expected,
                    "receipt_replay_identical": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
