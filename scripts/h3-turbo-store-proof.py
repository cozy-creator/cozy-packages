#!/usr/bin/env python3
"""Native PDD overlay construction, cancellation, resume, and source-object identity."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import time
from collections.abc import Mapping
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
    WeightsSink,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from h3_tables.adaln_operations import PLAIN
from h3_tables.kernel import H3Topology, adapter_shapes, source_shapes
from h3_tables.plans import Task
from h3_tables.source import H3FullTransformer
from h3_tables.turbo import RANK, TASKS, _produce, overlay_shapes, turbo_plan

if len(sys.argv) > 1:
    from diffusers import MiniMaxH3Transformer3DModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
    from official import canonical_timestep_plan
    from turbo import TurboOverlay, TurboSchedule


CONFIG = {
    "hidden_size": 8,
    "num_attention_heads": 2,
    "attention_head_dim": 4,
    "ffn_dim": 12,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "in_channels": 3,
    "patch_size": [1, 1, 1],
    "audio_in_channels": 2,
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 12,
    "time_embed_dim": 4,
    "rope_freq_dim": 1,
}
TOPOLOGY = H3Topology(8, 2, 8, 12, 4)
CONFIGS = {f"{task}_dit": CONFIG for task in TASKS}


@invocable(memoize=True)
async def mini_turbo(
    ctx: Context,
    *,
    full: H3FullTransformer,
    fl2va_adapter: H3FullTransformer,
    ref2va_adapter: H3FullTransformer,
    weights: WeightsSink,
    tel: Telemetry,
) -> ModelArtifact:
    return _produce(
        ctx,
        tel,
        weights,
        full=full,
        adapters={"fl2va": fl2va_adapter, "ref2va": ref2va_adapter},
        configs=CONFIGS,
        topologies=dict.fromkeys(TASKS, TOPOLOGY),
    )


def mint(store: Any, name: str, values: dict[str, dict[str, torch.Tensor]]) -> ModelArtifact:
    targets = {}
    for component, rows in values.items():
        additions = {}
        for key, value in rows.items():
            dtype = "bf16" if value.dtype == torch.bfloat16 else "f32"
            additions[key] = {
                "logical_dtype": dtype,
                "shape": list(value.shape),
                "encoding": PLAIN,
                "parts": {"value": {"dtype": dtype, "shape": list(value.shape)}},
            }
        targets[component] = {"drop": [], "add": additions}
    writer = store.begin_derived(
        "sha256:" + __import__("hashlib").sha256(name.encode()).hexdigest(),
        1,
        {},
        targets,
        {},
        [(component, key) for component, rows in values.items() for key in rows],
        1 << 24,
        work_fingerprint="sha256:" + "22" * 32,
    )
    for component, rows in values.items():
        for key, value in rows.items():
            writer.add_part(
                component,
                key,
                "value",
                io.BytesIO(value.contiguous().view(torch.uint8).numpy().tobytes()),
            )
    receipt = writer.commit()
    manifest = receipt["manifest"]
    return ModelArtifact(
        name,
        "model",
        ObjectRef("sha256:" + manifest["sha256"], manifest["length"]),
        "sha256:" + "33" * 32,
    )


def header(store: Any, model: ModelArtifact) -> Any:
    return tensorfs.parse_header(bytes(store.manifest(model.manifest.digest)["header"]))


def reference_check(
    reference_path: str,
    store: Any,
    actual: Any,
    full: dict[str, dict[str, torch.Tensor]],
    adapters: Mapping[Task, dict[str, dict[str, torch.Tensor]]],
) -> None:
    """Compare emitted bytes with the pinned upstream implementation over Diffusers."""
    spec = importlib.util.spec_from_file_location("upstream_pdd", reference_path)
    assert spec is not None and spec.loader is not None
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    def read(component: str, key: str) -> torch.Tensor:
        row = actual["components"][component][key]
        part = row["parts"]["value"]
        raw = (
            bytes(part["inline"])
            if "inline" in part
            else b"".join(
                bytes(store.document("sha256:" + seg["sha256"], seg["length"]))
                for seg in part["segments"]
            )
        )
        dtype = torch.bfloat16 if row["logical"]["logical_dtype"] == "bf16" else torch.float32
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(row["logical"]["shape"])

    for task in TASKS:
        model = MiniMaxH3Transformer3DModel(**CONFIG).to(torch.bfloat16)
        model.time_embedder.float()
        model.proj_out.float()
        model.audio_proj_out.float()
        model.load_state_dict(full[f"{task}_dit"], strict=False)
        upstream.add_lora(
            model, upstream.DEFAULT_PDD_CONFIG["lora_targets"].split(","), RANK, float(RANK)
        )
        upstream.attach_parallel_decoder(model, 32)
        incompatible = model.load_state_dict(adapters[task]["adapter"], strict=False)
        assert not incompatible.unexpected_keys
        plan = turbo_plan(task)
        serving_plan = canonical_timestep_plan("fl2va_turbo" if task == "fl2va" else "ref2va_turbo")
        (schedule,) = serving_plan.schedules
        overlay = TurboOverlay.from_official_config(
            CONFIG,
            rank=RANK,
            alpha=float(RANK),
            schedule=TurboSchedule(schedule.video_timesteps, schedule.audio_timesteps),
            table_timesteps=plan.timesteps,
            table_block_keys=plan.block_rows,
            block_table_dtype=torch.bfloat16,
            final_table_dtype=torch.bfloat16,
        )
        overlay.load_state_dict(
            {key: read(f"{task}_turbo", key) for key in actual["components"][f"{task}_turbo"]},
            strict=True,
        )
        with torch.inference_mode():
            temb = model.time_embedder(model.time_proj(torch.tensor(plan.timesteps)))
            rows = torch.tensor([index * 3 + tag for index, tag in plan.block_rows])
            for index, block in enumerate(model.transformer_blocks):
                expected = torch.stack(block.adaln_proj(temb), dim=1).index_select(0, rows)
                observed = read(f"{task}_turbo", f"transformer_blocks.{index}.adaln_proj.table")
                assert torch.equal(expected, observed), (task, index, "modulation")
            expected = model.norm_out.linear(torch.nn.functional.silu(temb).bfloat16()).reshape(
                len(plan.timesteps), 2, -1
            )
            assert torch.equal(expected, read(f"{task}_turbo", "norm_out.table"))
            for name, shift in (("proj_out", 12.0), ("audio_proj_out", 3.0)):
                head = getattr(model, name)
                inputs = torch.randn(3, head.in_features)
                for index in range(8):
                    head.set_plan(
                        upstream.pdd_sampling_plan(
                            upstream.pdd_time_grid(shift, 32).diff(), index * 4, 4
                        ).float()
                    )
                    observed = torch.nn.functional.linear(
                        inputs,
                        read(f"{task}_turbo", f"{name}.weight")[index],
                        read(f"{task}_turbo", f"{name}.bias")[index],
                    )
                    torch.testing.assert_close(head(inputs), observed, rtol=1e-6, atol=1e-7)
    print("Stored tables and all eight heads match pinned upstream PDD over Diffusers")


def main() -> None:
    torch.manual_seed(41)
    full_values = {
        f"{task}_dit": {
            key: (torch.randn(shape) * 0.02).to(dtype)
            for key, (dtype, shape) in source_shapes(TOPOLOGY).items()
        }
        for task in TASKS
    }
    adapters = {}
    for task in TASKS:
        shapes = {
            key: ((32, *shape[1:]) if dtype == "f32" else shape)
            for key, (dtype, shape) in overlay_shapes(CONFIG, turbo_plan(task)).items()
            if not key.endswith(".table")
        }
        shapes.update({key: shape for key, (_, shape) in adapter_shapes(TOPOLOGY, RANK).items()})
        adapters[task] = {
            "adapter": {
                key: (torch.randn(shape) * 0.03).bfloat16() for key, shape in shapes.items()
            }
        }
    app = App()
    app.job(mini_turbo, weights=(WeightsOutput("model", 1 << 24),))
    describe(app)
    with tempfile.TemporaryDirectory(prefix="h3-turbo-native-") as area:
        root = Path(area)
        store = tensorfs.Store.init(root / "store")
        sources = {
            "full": mint(store, "full", full_values),
            **{f"{task}_adapter": mint(store, task, adapters[task]) for task in TASKS},
        }

        def run(name: str, epoch: int, cancel: bool = False) -> Any:
            checkpoints: list[Any] = []
            host = WeightsTransactionHost(
                store=store,
                owner_scope="turbo-proof",
                request_id=name,
                invocation_spec_digest="sha256:" + "44" * 32,
                work_fingerprint="sha256:" + "55" * 32,
                writer_session_id=epoch,
                allowed_sources={
                    value.manifest.digest: value.manifest.length for value in sources.values()
                },
                output_bounds={"model": 1 << 24},
                record_checkpoint=checkpoints.append,
            )
            return attempt(
                app.get("mini_turbo"),
                {name: msgspec.to_builtins(value) for name, value in sources.items()},
                Invocation(
                    name,
                    root / f"{name}-{epoch}",
                    time.monotonic() + 60,
                    models={
                        name: _derive_model(H3FullTransformer, value.manifest.digest)
                        for name, value in sources.items()
                    },
                    weights=host.open,
                    weights_source_structure=host.structure,
                    weights_source_config=host.config,
                    cancel=lambda: cancel and bool(checkpoints),
                ),
            )

        stopped, outcome, _ = run("resume", 1, True)
        assert stopped is None and outcome.terminal == "canceled", outcome
        result, outcome, _ = run("resume", 2)
        assert outcome.terminal == "succeeded" and result is not None, outcome
        clean, outcome, _ = run("clean", 1)
        assert outcome.terminal == "succeeded" and clean is not None, outcome
        assert result.result.manifest == clean.result.manifest
        replay, outcome, _ = run("resume", 3)
        assert outcome.terminal == "succeeded" and replay.result == result.result, outcome
        actual = header(store, result.result)
        if len(sys.argv) > 1:
            reference_check(sys.argv[1], store, actual, full_values, adapters)
        assert set(actual["components"]) == {
            "fl2va_turbo",
            "ref2va_turbo",
        }
        for task in TASKS:
            original = header(store, sources[f"{task}_adapter"])["components"]["adapter"]
            overlay = actual["components"][f"{task}_turbo"]
            assert set(overlay) == set(overlay_shapes(CONFIG, turbo_plan(task)))
            for key, tensor in overlay.items():
                if key.endswith((".lora_down", ".lora_up")):
                    assert tensor == original[key]
                assert "adaln_proj.linear" not in key
        print(
            json.dumps(
                {
                    "adapter_only_checkpoint": True,
                    "factor_objects_inherited": True,
                    "resume_equals_clean": True,
                    "replay_identical": True,
                    "no_adapter_modulation_copies": True,
                }
            )
        )


if __name__ == "__main__":
    main()
