#!/usr/bin/env python3
"""Restamp a real native checkpoint without changing its tensors or explicit row order."""

from __future__ import annotations

import copy
import hashlib
import io
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import tensorfs
import torch
from cozy_runtime.author import Invocation, ModelArtifact, ObjectRef, attempt, canonical_json
from cozy_runtime.author._model import _derive_model
from cozy_runtime.models.minimax_h3.table_layout import TableLayout
from h3_tables import job
from h3_tables.kernel import H3Topology, table_shapes
from h3_tables.plans import TASKS
from h3_tables.source import H3FullTransformer

from native_execution_fixture import NativeExecution


def mint(store: Any, config: dict[str, Any]) -> ModelArtifact:
    values = {name: {"weight": torch.tensor([2.0])} for name in config}
    for task in TASKS:
        component = f"{task}_dit"
        layout = TableLayout.parse(config[component]["cozy_h3"]["table_keys"])
        plan = job._production_plan(task)
        plan = replace(plan, timesteps=layout.timesteps, block_rows=layout.block_keys)
        values[component] = {
            key: torch.full(shape, 0.125, dtype=torch.bfloat16)
            for key, shape in table_shapes(H3Topology.from_config(config[component]), plan).items()
        }
    targets = {
        component: {
            "drop": [],
            "add": {
                key: {
                    "logical_dtype": "bf16" if value.dtype == torch.bfloat16 else "f32",
                    "shape": list(value.shape),
                    "encoding": job.PLAIN_SPEC,
                    "parts": {
                        "value": {
                            "dtype": "bf16" if value.dtype == torch.bfloat16 else "f32",
                            "shape": list(value.shape),
                        }
                    },
                }
                for key, value in rows.items()
            },
        }
        for component, rows in values.items()
    }
    writer = store.begin_derived(
        "sha256:" + hashlib.sha256(b"restamp-source").hexdigest(),
        1,
        {},
        targets,
        {"model": {"kind": "add"}},
        [(component, key) for component, rows in values.items() for key in rows],
        16 << 20,
        work_fingerprint="sha256:" + "12" * 32,
    )
    for component, rows in values.items():
        for key, value in rows.items():
            writer.add_part(
                component, key, "value", io.BytesIO(value.view(torch.uint8).numpy().tobytes())
            )
    writer.add_config("model", io.BytesIO(canonical_json.encode(config)))
    receipt = writer.commit()
    manifest = receipt["manifest"]
    return ModelArtifact(
        "source",
        "model",
        ObjectRef("sha256:" + manifest["sha256"], manifest["length"]),
        canonical_json.digest(receipt),
    )


def main() -> None:
    config = copy.deepcopy(canonical_json.decode(job._asset("model-config.json")))
    # Tiny real tables at the release feature widths; row meanings are explicitly
    # checkpoint-owned, so metadata-only restamp need not invent unused schedule rows.
    for task in TASKS:
        config[f"{task}_dit"]["cozy_h3"]["table_keys"] = {
            "final_normalization": [{"index": 0, "timestep": (0.0).hex()}],
            "block_modulation": [
                {"index": 0, "timestep": (0.0).hex(), "modality": "video", "modality_tag": 0}
            ],
        }
    with tempfile.TemporaryDirectory(prefix="h3-restamp-native-") as directory:
        root = Path(directory)
        store = tensorfs.Store.init(root / "store")
        source = mint(store, config)
        with NativeExecution(
            store, root, "restamp", {"lane": source}, {"restamped": 128 << 10}
        ) as execution:
            result, outcome, _ = attempt(
                job.app.get("restamp"),
                {},
                Invocation(
                    "restamp",
                    root / "attempt",
                    time.monotonic() + 60,
                    models={"lane": _derive_model(H3FullTransformer, source.manifest.digest)},
                    tensorfs_output=execution.client.open_output,
                    tensorfs_source=execution.client.source,
                    tensorfs_adopt=execution.client.adopt_model,
                ),
            )
        assert outcome.terminal == "succeeded" and result is not None, outcome
        before_raw = store.manifest(source.manifest.digest)["header"]
        after_raw = store.manifest(result.result.manifest.digest)["header"]
        assert before_raw is not None and after_raw is not None
        before = tensorfs.parse_header(bytes(before_raw))
        after = tensorfs.parse_header(bytes(after_raw))
        assert before == after
        print("Native restamp preserves every tensor, config and row order without casting")


if __name__ == "__main__":
    main()
