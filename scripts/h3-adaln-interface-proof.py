#!/usr/bin/env python3
"""Run the installed H3 AdaLN helper through Runtime's call broker.

Child replies are fixed routing controls. Native table computation and model quality
are qualified by the separate binding/resume/numerical drivers, not by this proof.
"""

from __future__ import annotations

import json
import tempfile
import time
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


def run(root: Path) -> None:
    assert "site-packages" in Path(job.__file__).parts, job.__file__
    assert precompute_adaln.__module__ == "h3_tables.operations"
    surfaces = describe(job.app)
    assert "precompute-adaln" not in {surface.name for surface in surfaces}
    bindings = {
        (surface.fn.__module__, surface.fn.__name__): _CallType(
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
                "helper_from_installed_wheel": True,
                "precompute_job_registered": False,
                "managed_sibling_calls": 5,
                "per_task_projection_inputs": True,
                "invalid_plan_refused_before_calls": True,
            }
        )
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="h3-adaln-helper-") as area:
        run(Path(area))


if __name__ == "__main__":
    main()
