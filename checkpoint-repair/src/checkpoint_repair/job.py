"""Ordinary CPU job repairing the original four H3 checkpoint layouts."""

from __future__ import annotations

from typing import Any

import msgspec
from cozy_runtime.author import App, Model, Telemetry, WeightsOutput, WeightsSink

from .repair import ENCODINGS, MAX_NEW_BYTES, RepairResult, repair

app = App()


class Checkpoint(Model[object]):
    """Granted source metadata and bytes; no inference model is constructed."""

    def load(self, loader: Any) -> None:
        del loader


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


@app.job(name="h3-swiglu", weights=(WeightsOutput("checkpoint", max_new_bytes=MAX_NEW_BYTES),))
def h3_swiglu(
    payload: Request, source: Checkpoint, artifacts: WeightsSink, tel: Telemetry
) -> RepairResult:
    """Repair the recorded FC1 rows without downloading source files or requantizing."""
    del payload
    return repair(source, artifacts, tel, ENCODINGS)
