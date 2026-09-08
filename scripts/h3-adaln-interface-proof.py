#!/usr/bin/env python3
"""Generated caller and real broker sequencing; numerical/native proofs are separate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import msgspec
from cozy_runtime.author import (
    Invocation,
    ModelArtifact,
    ObjectRef,
    attempt,
    canonical_json,
    describe,
)
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.internal import interface_wheel, package_interface
from cozy_runtime.internal.discovery import Discovered
from h3_tables import job
from h3_tables.adaln_operations import Selection


def artifact(name: str, digit: str) -> ModelArtifact:
    return ModelArtifact(
        name, "model", ObjectRef("sha256:" + digit * 64, 100), "sha256:" + digit * 64
    )


def main() -> None:
    surfaces = describe(job.app)
    parent = next(surface for surface in surfaces if surface.name == "precompute-adaln")
    assert not parent.model_bindings and not parent.weights_outputs
    declared = {surface.name: len(surface.weights_outputs) for surface in surfaces}
    assert declared["select-adaln-weights"] == 1
    assert declared["apply-adaln"] == 3
    assert declared["retable-adaln"] == 1
    discovered = Discovered(
        job.app, "h3_tables.job:app", Path(job.__file__).parents[2], job, surfaces, {}
    )
    interface = package_interface.canonical_bytes(package_interface.build(discovered))
    generated = interface_wheel.generate(interface)
    caller = generated["h3_tables/operations/__init__.py"].decode()
    assert "def precompute_adaln(" in caller and "ModelArtifact" in caller
    assert "torch" not in caller and "WeightsSink" not in caller
    bindings = {
        ("", surface.fn.__module__, surface.fn.__name__): _CallType(
            "sha256:" + "f" * 64,
            surface.fn.__module__,
            surface.fn.__name__,
            cast(type[msgspec.Struct], surface.payload_type),
            cast(type[msgspec.Struct], surface.result_type),
        )
        for surface in surfaces
        if surface.invocable
    }
    original = artifact("original", "1")
    projected_fl, projected_ref = artifact("projected-fl", "2"), artifact("projected-ref", "3")
    bank_fl, bank_ref, pruned = (
        artifact("bank-fl", "4"),
        artifact("bank-ref", "5"),
        artifact("pruned", "6"),
    )
    answers = [
        Selection(projected_fl, False),
        Selection(projected_ref, False),
        bank_fl,
        bank_ref,
        pruned,
    ]
    calls: list[dict[str, Any]] = []

    def exchange(kind: str, frame: dict[str, Any]) -> dict[str, Any]:
        if kind == "child_call":
            calls.append(frame)
        return {
            "ok": True,
            "state": "succeeded",
            "result": canonical_json.encode(
                msgspec.to_builtins(answers[frame["call_index"]])
            ).decode(),
        }

    with tempfile.TemporaryDirectory(prefix="h3-adaln-interface-") as area:
        root = Path(area)
        for path, data in generated.items():
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from h3_tables.operations import precompute_adaln; "
                "assert 'torch' not in sys.modules; assert callable(precompute_adaln)",
                str(root),
            ],
            check=True,
        )
        result, outcome, _ = attempt(
            job.app.get("precompute-adaln"),
            {"model": msgspec.to_builtins(original), "timesteps": 50},
            Invocation(
                "parent",
                root / "spool",
                time.monotonic() + 20,
                calls=_Broker("parent", bindings, exchange),
            ),
        )
        assert outcome.terminal == "succeeded" and result is not None, outcome
        assert result.result == pruned
        assert len(calls) == 5
        assert [call["export"] for call in calls] == [
            "select_adaln_weights",
            "select_adaln_weights",
            "compute_adaln_tables",
            "compute_adaln_tables",
            "apply_adaln",
        ]
        payloads = [json.loads(call["payload"]) for call in calls]
        assert payloads[2]["source"] == msgspec.to_builtins(projected_fl)
        assert payloads[3]["source"] == msgspec.to_builtins(projected_ref)
        assert payloads[4]["source"] == msgspec.to_builtins(original)
        assert payloads[4]["fl2va"] == msgspec.to_builtins(bank_fl)
        assert payloads[4]["ref2va"] == msgspec.to_builtins(bank_ref)
        print(
            json.dumps(
                {
                    "parent_model_bindings": 0,
                    "generated_caller_imports_torch": False,
                    "managed_sibling_calls": 5,
                    "per_task_projection_inputs": True,
                }
            )
        )


if __name__ == "__main__":
    main()
