#!/usr/bin/env python3
"""Native H3 projection identities and bank/body binding refusals on small real tensors."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import tensorfs
import torch
from cozy_runtime.author import (
    Context,
    ModelArtifact,
    ObjectRef,
    UnsupportedInput,
    canonical_json,
)
from cozy_runtime.author._model import _derive_model
from h3_tables.adaln_operations import (
    PLAIN,
    _bank_config,
    _bindings,
    _plan,
    _project,
    _require_bank_binding,
)
from h3_tables.kernel import H3Topology, precompute_tables, source_shapes, table_shapes
from h3_tables.source import H3FullTransformer, inspection
from tensorfs.derived import Config, Derivation, Part, PartSource, Target, Tensor

from native_execution_fixture import NativeExecution

TOPOLOGY = H3Topology(8, 2, 8, 12, 4)
COMPONENT = "fl2va_dit"


def native(
    store: Any, name: str, values: dict[str, torch.Tensor], config: dict[str, bytes] | None = None
) -> ModelArtifact:
    tx = "sha256:" + hashlib.sha256(name.encode()).hexdigest()
    tensors = {}
    for key, value in values.items():
        dtype = "f32" if value.dtype == torch.float32 else "bf16"
        tensors[key] = {
            "logical_dtype": dtype,
            "shape": list(value.shape),
            "encoding": PLAIN,
            "parts": {"value": {"dtype": dtype, "shape": list(value.shape)}},
        }
    configs = {key: {"kind": "add"} for key in (config or {})}
    writer = store.begin_derived(
        tx,
        1,
        {},
        {COMPONENT: {"drop": [], "add": tensors}},
        configs,
        [(COMPONENT, key) for key in tensors],
        1 << 20,
        work_fingerprint="sha256:" + "51" * 32,
    )
    for key, value in values.items():
        writer.add_part(
            COMPONENT,
            key,
            "value",
            io.BytesIO(value.contiguous().view(torch.uint8).numpy().tobytes()),
        )
    for key, raw in (config or {}).items():
        writer.add_config(key, io.BytesIO(raw))
    receipt = writer.commit()
    manifest = receipt["manifest"]
    return ModelArtifact(
        name,
        "model",
        ObjectRef("sha256:" + manifest["sha256"], manifest["length"]),
        canonical_json.digest(receipt),
    )


def bound_context(
    stack: ExitStack,
    store: Any,
    root: Path,
    name: str,
    artifacts: dict[str, ModelArtifact],
    outputs: dict[str, int],
) -> tuple[Context, dict[str, H3FullTransformer]]:
    execution = stack.enter_context(NativeExecution(store, root, name, artifacts, outputs))
    models = {
        key: _derive_model(H3FullTransformer, artifact.manifest.digest)
        for key, artifact in artifacts.items()
    }
    return execution.context(), models


def main() -> None:
    torch.manual_seed(99)
    values = {
        key: (torch.randn(shape) * 0.03).to(dtype)
        for key, (dtype, shape) in source_shapes(TOPOLOGY).items()
    }
    with tempfile.TemporaryDirectory(prefix="h3-adaln-bindings-") as area, ExitStack() as stack:
        root = Path(area)
        store = tensorfs.Store.init(root / "store")
        original = native(
            store, "source", {**values, "body.weight": torch.zeros((16, 32), dtype=torch.bfloat16)}
        )
        changed = native(
            store,
            "body-changed",
            {**values, "body.weight": torch.ones((16, 32), dtype=torch.bfloat16)},
        )
        projections = []
        for name, artifact in (("project-source", original), ("project-body-changed", changed)):
            ctx, models = bound_context(
                stack, store, root, name, {"source": artifact}, {"model": 1 << 20}
            )
            projections.append(_project(ctx, models["source"], "fl2va", "model", TOPOLOGY))
        assert projections[0].manifest == projections[1].manifest
        for index, key in enumerate(
            (
                "time_embedder.linear_1.weight",
                "transformer_blocks.0.adaln_proj.linear.weight",
                "norm_out.linear.weight",
            )
        ):
            edited = {name: value.clone() for name, value in values.items()}
            edited[key].flatten()[0] += 0.25
            source = native(store, f"generator-change-{index}", edited)
            ctx, models = bound_context(
                stack,
                store,
                root,
                f"projection-change-{index}",
                {"source": source},
                {"model": 1 << 20},
            )
            result = _project(ctx, models["source"], "fl2va", "model", TOPOLOGY)
            assert result.manifest != projections[0].manifest

        tables: dict[str, torch.Tensor] = {}
        precompute_tables(
            plan=_plan("fl2va"),
            topology=TOPOLOGY,
            read=lambda key, _dtype, _shape: values[key],
            write=lambda key, value: tables.__setitem__(key, value),
            progress=lambda _done, _total: None,
            device=torch.device("cpu"),
        )
        bank = native(
            store, "bank", tables, {"adaln": _bank_config("fl2va", projections[0].manifest.digest)}
        )
        wrong = native(
            store, "wrong-bank", tables, {"adaln": _bank_config("fl2va", result.manifest.digest)}
        )
        bindings = canonical_json.encode(
            {
                "fl2va_dit": {
                    "cozy_h3": {"generating_projection_digest": projections[0].manifest.digest}
                },
                "ref2va_dit": {
                    "cozy_h3": {"generating_projection_digest": projections[0].manifest.digest}
                },
            }
        )
        body = native(
            store,
            "pruned-body",
            {"body.weight": torch.zeros((16, 32), dtype=torch.bfloat16)},
            {"model": bindings},
        )
        outputs = {"bf16": 0, "fp8": 1024, "mxfp8": 1024}
        ctx, models = bound_context(
            stack, store, root, "attach", {"body": body, "bank": bank, "wrong": wrong}, outputs
        )
        expected = _bindings(inspection(ctx, models["body"]))["fl2va"]
        _require_bank_binding(ctx, models["bank"], "fl2va", expected, TOPOLOGY)
        try:
            _require_bank_binding(ctx, models["wrong"], "fl2va", expected, TOPOLOGY)
        except UnsupportedInput as error:
            assert error.code == "adaln_binding"
        else:
            raise AssertionError("same-shape/same-plan bank from another generator was accepted")
        tensors = {
            key: Tensor(
                "bf16",
                shape,
                PLAIN,
                {"value": Part("bf16", shape, source=PartSource("bank", COMPONENT, key, "value"))},
            )
            for key, shape in table_shapes(TOPOLOGY, _plan("fl2va")).items()
        }
        rowwise = next(
            digest for alias, digest in tensorfs.seed_digests() if alias == "fp8-rowwise/1"
        )
        mxfp8 = next(digest for alias, digest in tensorfs.seed_digests() if alias == "mxfp8/1")
        table_parts = []
        for label, encoding in (("bf16", None), ("fp8", rowwise), ("mxfp8", mxfp8)):
            additions = dict(tensors)
            payloads = {}
            if encoding is not None:
                payloads = {
                    "data": bytes(512),
                    "scale": torch.ones(16).numpy().tobytes()
                    if label == "fp8"
                    else bytes([127] * 16),
                }
                parts = {"data": Part("f8_e4m3fn", (16, 32))}
                parts["scale"] = Part("f32", (16,)) if label == "fp8" else Part("u8", (16, 1))
                additions["body.weight"] = Tensor("bf16", (16, 32), encoding, parts)
            with ctx.output(label).open(
                Derivation(
                    sources={
                        name: inspection(ctx, models[name]).source for name in ("body", "bank")
                    },
                    targets={
                        COMPONENT: Target(
                            "body",
                            COMPONENT,
                            drop=("body.weight",) if encoding else (),
                            add=additions,
                        )
                    },
                    configs={"model": Config("copy", "body", "model")},
                    order=((COMPONENT, "body.weight"), *((COMPONENT, key) for key in tensors)),
                )
            ) as transaction:
                for role, payload in payloads.items():
                    transaction.add_part(COMPONENT, "body.weight", role, payload)
                receipt = ctx.adopt_model(transaction.commit())
            header_bytes = store.manifest(receipt.manifest.digest)["header"]
            assert header_bytes is not None, "produced model has no CozyTensors header"
            header = tensorfs.parse_header(header_bytes)
            table_parts.append({key: header["components"][COMPONENT][key] for key in tensors})
        assert table_parts[0] == table_parts[1] == table_parts[2]
        print(
            json.dumps(
                {
                    "unrelated_body_change_keeps_projection": True,
                    "generating_group_changes_invalidate": 3,
                    "wrong_generator_bank_refused": True,
                    "bf16_fp8_mxfp8_share_exact_tables": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
