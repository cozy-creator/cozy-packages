#!/usr/bin/env python3
"""Run the ordinary H3 helper through its caller overlay and admitted broker.

Child replies are fixed routing controls. Native table computation and model quality
are qualified by the separate binding/resume/numerical drivers, not by this proof.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any, cast

import msgspec
from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    ModelArtifact,
    ObjectRef,
    attempt,
    canonical_json,
    describe,
)
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.internal import (
    installed_interfaces,
    interface_wheel,
    package_environment,
    package_interface,
)
from cozy_runtime.internal.discovery import Discovered

# The isolated child resolves the actual caller wheel before importing the model tools.
if "--overlay-root" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--overlay-root") + 1])

from h3_tables import job
from h3_tables.adaln_operations import Selection
from h3_tables.operations import precompute_adaln

app = App()


class Request(msgspec.Struct):
    model: ModelArtifact
    timesteps: int = 50


@app.job
async def prepare(ctx: Context, payload: Request) -> ModelArtifact:
    ctx.raise_if_cancelled()
    return await precompute_adaln(model=payload.model, timesteps=payload.timesteps)


def artifact(name: str, digit: str) -> ModelArtifact:
    return ModelArtifact(
        name, "model", ObjectRef("sha256:" + digit * 64, 100), "sha256:" + digit * 64
    )


def run_overlay(root: Path) -> None:
    assert Path(job.__file__).is_relative_to(root)
    assert precompute_adaln.__module__ == "h3_tables.operations"
    surfaces = describe(job.app)
    assert "precompute-adaln" not in {surface.name for surface in surfaces}
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

    result, outcome, _ = attempt(
        app.get("prepare"),
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
    assert payloads[4] == {
        "source": msgspec.to_builtins(original),
        "fl2va": msgspec.to_builtins(bank_fl),
        "ref2va": msgspec.to_builtins(bank_ref),
    }
    calls.clear()
    _, refused, _ = attempt(
        app.get("prepare"),
        {"model": msgspec.to_builtins(original), "timesteps": 37},
        Invocation(
            "invalid",
            root / "invalid",
            time.monotonic() + 20,
            calls=_Broker("invalid", bindings, exchange),
        ),
    )
    assert refused.code == "adaln_plan" and not calls, refused
    print(
        json.dumps(
            {
                "helper_from_overlay": True,
                "precompute_job_registered": False,
                "managed_sibling_calls": 5,
                "per_task_projection_inputs": True,
                "invalid_plan_refused_before_calls": True,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--implementation-wheel", type=Path)
    mode.add_argument("--overlay-root", type=Path)
    args = parser.parse_args()
    if args.overlay_root is not None:
        run_overlay(args.overlay_root)
        return
    surfaces = describe(job.app)
    assert "precompute-adaln" not in {surface.name for surface in surfaces}
    discovered = Discovered(
        job.app, "h3_tables.job:app", Path(job.__file__).parents[2], job, surfaces, {}
    )
    interface = package_interface.canonical_bytes(package_interface.build(discovered))
    implementation = args.implementation_wheel.read_bytes()
    with zipfile.ZipFile(io.BytesIO(implementation)) as source:
        metadata = BytesParser().parsebytes(
            source.read(
                next(name for name in source.namelist() if name.endswith(".dist-info/METADATA"))
            )
        )
    _, wheel = interface_wheel.build(
        interface,
        distribution="minimax-h3-tools",
        version=str(metadata["Version"]),
        implementation_wheel=implementation,
        implementation_filename=args.implementation_wheel.name,
    )
    with tempfile.TemporaryDirectory(prefix="h3-adaln-overlay-") as area:
        root = Path(area)
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            archive.extractall(root)
            preserved = archive.read(f"minimax_h3_tools-{metadata['Version']}.dist-info/METADATA")
            assert b"Requires-Dist: torch" in preserved
            assert archive.read("h3_tables/assets/timestep-plan.fl2va.json")
        exports = installed_interfaces.read(
            package_environment.InstalledEnvironment(
                root, root, Path(sys.executable), b"", "", "sha256:" + "a" * 64, True
            )
        )
        assert "precompute_adaln" not in {row["export"] for row in exports}
        assert {"select_adaln_weights", "compute_adaln_tables", "apply_adaln", "retable_adaln"} <= {
            row["export"] for row in exports
        }
        subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--overlay-root", str(root)],
            check=True,
        )


if __name__ == "__main__":
    main()
