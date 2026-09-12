#!/usr/bin/env python3
"""Prove selective cast fanout, exact native inheritance, interruption and replay on CPU."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import tensorfs
from cozy_runtime.author import Invocation, attempt, canonical_json, script_app
from cozy_runtime.author._model import _derive_model
from cozy_runtime.internal.weights_sink import WeightsTransactionHost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
sys.path.insert(0, str(ROOT / "examples" / "client-scripts"))
import h3_shared_vae_repair as repair  # noqa: E402
from h3_tables import lanes  # noqa: E402
from h3_tables.plans import TASKS, parse_declared_plan  # noqa: E402
from h3_tables.source import H3FullTransformer  # noqa: E402

SHAPES = {
    "encoder.conv_in.weight": (8, 8, 2, 2, 2),
    "decoder.proj_in.weight": (128, 24),
    "decoder.transformer_blocks.0.attn.to_q.weight": (128, 128),
    "decoder.norm_out.weight": (128,),
    "post_quant_conv.weight": (24, 24, 1, 1, 1),
}


def fixture(store: Any, lane: str) -> tuple[str, int]:
    specs = dict(tensorfs.seed_digests())
    values: dict[tuple[str, str, str], bytes] = {}
    targets: dict[str, Any] = {}
    for index, component in enumerate(lanes.COMPONENTS):
        rng = np.random.default_rng(719 + index)
        additions: dict[str, Any] = {}
        shapes = SHAPES if component == "video_vae" else {"weight": (128, 64)}
        if component.endswith("_dit") and lane != "bf16-full":
            shapes = {"norm_out.table": (2, 16), **shapes}
        for key, shape in shapes.items():
            if (
                component.endswith("_dit")
                and key == "weight"
                and lane in {"fp8-pruned", "mxfp8-pruned"}
            ):
                rowwise = lane == "fp8-pruned"
                parts = {
                    "data": {"dtype": "f8_e4m3fn", "shape": list(shape)},
                    "scale": {
                        "dtype": "f32" if rowwise else "u8",
                        "shape": [128, 1] if rowwise else [128, 2],
                    },
                }
                values[component, key, "data"] = b"\x38" * 8192
                values[component, key, "scale"] = (
                    np.ones(128, dtype="<f4").tobytes() if rowwise else b"\x7f" * 256
                )
                dtype, encoding = "bf16", specs["fp8-rowwise/1" if rowwise else "mxfp8/1"]
            else:
                dtype, encoding = "f32", specs["plain/1"]
                parts = {"value": {"dtype": dtype, "shape": list(shape)}}
                values[component, key, "value"] = (
                    (rng.standard_normal(shape) * 0.02).astype("<f4").tobytes()
                )
            additions[key] = {
                "logical_dtype": dtype,
                "shape": list(shape),
                "encoding": encoding,
                "parts": parts,
            }
        targets[component] = {"drop": [], "add": additions}
    config = canonical_json.decode(
        files("h3_tables").joinpath("assets", "model-config.json").read_bytes()
    )
    for task in TASKS:
        stamp = config[f"{task}_dit"]["cozy_h3"]
        if lane == "bf16-full":
            config[f"{task}_dit"]["cozy_h3"] = {"task": task, "modulation": "full"}
        elif lane == "mxfp8-pruned":
            plan = parse_declared_plan(
                files("h3_tables").joinpath("assets", f"timestep-plan.{task}.json").read_bytes()
            )
            del stamp["table_keys"]
            stamp["timestep_plan_digest"] = plan.digest
    transaction = tensorfs.object_id(("fixture-" + lane).encode())
    writer = store.begin_derived(
        transaction,
        1,
        {},
        targets,
        {"model": {"kind": "add"}, "untouched": {"kind": "add"}},
        [(component, key) for component, target in targets.items() for key in target["add"]],
        1 << 20,
        work_fingerprint=tensorfs.object_id(b"fixture"),
    )
    for (component, key, role), data in values.items():
        writer.add_part(component, key, role, io.BytesIO(data))
    writer.add_config("model", io.BytesIO(canonical_json.encode(config)))
    writer.add_config("untouched", io.BytesIO(b'{"keep":"exact"}'))
    receipt = writer.commit()
    store.derived_adopt(transaction, "fixture-" + lane)
    return "sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"]


def header(store: Any, manifest: str) -> Any:
    raw = store.manifest(manifest)["header"]
    assert raw is not None
    return tensorfs.parse_header(raw)


def main() -> None:
    app = script_app("h3_shared_vae_repair")
    with tempfile.TemporaryDirectory(prefix="h3-shared-vae-native-") as directory:
        root = Path(directory)
        store = tensorfs.Store.ensure(root / "store")
        sources = {lane: fixture(store, lane) for lane in repair.SOURCES}
        models = {
            lane.replace("-", "_"): _derive_model(H3FullTransformer, value[0])
            for lane, value in sources.items()
        }
        original = {lane: header(store, value[0]) for lane, value in sources.items()}
        repair.SOURCES = {lane: value[0] for lane, value in sources.items()}
        reads: list[str] = []
        original_read = lanes._read_f32

        def read(*args: Any, **kwargs: Any) -> Any:
            assert kwargs["source"] == "shared" and kwargs["component"] == "video_vae"
            reads.append(kwargs["key"])
            return original_read(*args, **kwargs)

        lanes._read_f32 = read

        def invoke(epoch: int, *, interrupt: bool = False) -> Any:
            checkpoints: list[Any] = []

            def checkpoint(value: Any) -> None:
                checkpoints.append(value)
                if interrupt and len(checkpoints) == 4:
                    raise RuntimeError("injected interruption after one shared tensor")

            host = WeightsTransactionHost(
                store=store,
                owner_scope="shared-vae-proof",
                request_id="repair",
                invocation_spec_digest=tensorfs.object_id(b"invocation"),
                work_fingerprint=tensorfs.object_id(b"same repair work"),
                writer_session_id=epoch,
                allowed_sources=dict(sources.values()),
                output_bounds={name: repair.MAX_NEW_BYTES for name in models},
                record_checkpoint=checkpoint,
            )
            return attempt(
                app.get("main"),
                {},
                Invocation(
                    "repair",
                    root / f"attempt-{epoch}",
                    time.monotonic() + 60,
                    models=models,
                    weights=host.open,
                    weights_source_structure=host.structure,
                    weights_source_config=host.config,
                ),
            )

        result, outcome, _ = invoke(1, interrupt=True)
        assert result is None and outcome.terminal == "failed", outcome
        assert reads == ["decoder.proj_in.weight"], reads
        reads.clear()
        result, outcome, _ = invoke(2)
        assert outcome.terminal == "succeeded" and result is not None, outcome
        assert len(reads) == 2 and "decoder.proj_in.weight" not in reads, reads
        fixed = result.result.value
        reads.clear()
        replayed, outcome, _ = invoke(3)
        assert outcome.terminal == "succeeded" and replayed is not None, outcome
        assert replayed.result.value == fixed and reads == [], (replayed, reads)

        shared_vae = None
        for lane, artifact in fixed.items():
            produced = header(store, artifact.manifest.digest)
            before = original[lane]
            assert list(produced["components"]) == list(before["components"])
            for component in lanes.COMPONENTS:
                assert list(produced["components"][component]) == list(
                    before["components"][component]
                )
                if component != "video_vae":
                    assert produced["components"][component] == before["components"][component], (
                        lane,
                        component,
                    )
            vae = produced["components"]["video_vae"]
            assert shared_vae is None or shared_vae == vae, "VAE ObjectRefs diverged across lanes"
            shared_vae = vae
            for key, row in vae.items():
                expected = "f16" if lanes.decode_operand(key, SHAPES[key]) else "f32"
                assert row["logical"]["logical_dtype"] == expected, (key, row)
                if expected == "f32":
                    assert row == before["components"]["video_vae"][key]
            assert produced["configs"]["untouched"] == before["configs"]["untouched"]
            config = canonical_json.decode(produced["configs"]["model"])
            old_config = canonical_json.decode(before["configs"]["model"])
            if lane != "mxfp8-pruned":
                assert config == old_config, lane
            else:
                assert all("table_keys" in config[f"{task}_dit"]["cozy_h3"] for task in TASKS)
        print(
            json.dumps(
                {
                    "native_outputs": 4,
                    "unique_cast_tensors": 3,
                    "resumed_cast_tensors": 2,
                    "replay_source_reads": 0,
                    "all_dit_and_quantized_object_refs_preserved": True,
                    "shared_vae_object_refs_equal": True,
                    "only_mxfp8_table_metadata_upgraded": True,
                    "construction_order_preserved": True,
                }
            )
        )


if __name__ == "__main__":
    main()
