#!/usr/bin/env python3
"""Exercise actual four-lane sequencing with interrupted computation and receipt replay.

This is a CPU orchestration proof. Recording collaborators replace expensive table
and quantization work and native storage; no GPU, payload custody, or numerical claim.
The callable still parses its real assets and builds its real output declarations.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from cozy_runtime.author import (
    WeightsReceipt,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
)
from cozy_runtime.derive.quantization import QuantizationStats
from h3_tables import job


class Interrupted(RuntimeError):
    pass


class Transaction:
    def __init__(self, owner: Recorder, slot: str) -> None:
        self.owner, self.slot = owner, slot
        self.replayed = slot in owner.committed
        prior = owner.committed.get(slot)
        self.receipt = replace(prior, replayed=True) if prior else None
        self.configured = False

    def __enter__(self) -> Transaction:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def add_config(self, name: str, raw: bytes) -> None:
        assert not self.replayed and name == "model" and raw
        self.configured = True

    def commit(self) -> WeightsReceipt:
        assert not self.replayed and self.configured
        assert self.slot not in self.owner.committed
        receipt = WeightsReceipt(self.slot, self.slot, self.slot, b"")
        self.owner.committed[self.slot] = receipt
        self.owner.events.append("commit:" + self.slot)
        return receipt


class Recorder:
    def __init__(self) -> None:
        self.committed: dict[str, WeightsReceipt] = {}
        self.events: list[str] = []
        self.fail_at = ""

    def structure(self, _: Any) -> WeightsSource:
        return WeightsSource(
            (),
            tuple(
                WeightsSourceTensor(
                    component, key, "f32", (1,), (WeightsSourcePart("value", "f32", (1,)),)
                )
                for component, target in job._full_targets().items()
                for key in target.drop
            ),
        )

    def open(self, slot: str, **_: Any) -> Transaction:
        return Transaction(self, slot)

    def derive(self, slot: str, **_: Any) -> WeightsReceipt:
        transaction = self.open(slot)
        if transaction.receipt is not None:
            return transaction.receipt
        transaction.configured = True
        return transaction.commit()

    def tables(
        self, task: str, _ctx: Any, _source: Any, active: dict[str, Any], _tel: Any, _range: Any
    ) -> int:
        event = "tables:" + task
        self.events.append(event)
        assert not self.committed.keys() & active.keys()
        if self.fail_at == event:
            raise Interrupted(event)
        return 1

    def quantize(self, transaction: Transaction, *_: Any, **kwargs: Any) -> QuantizationStats:
        event = "quantize:" + transaction.slot
        self.events.append(event + ":" + kwargs["target_component"])
        if self.fail_at == event:
            raise Interrupted(event)
        return QuantizationStats(1, 1, 1, 2, 0.03125)


class Telemetry:
    def stage(self, *_: Any, **__: Any) -> Any:
        return nullcontext()

    def progress(self, *_: Any, **__: Any) -> None:
        pass

    def metric(self, *_: Any, **__: Any) -> None:
        pass

    def log(self, *_: Any, **__: Any) -> None:
        pass


def invoke(recorder: Recorder, fail_at: str = "") -> Any:
    recorder.fail_at = fail_at
    original_tables = job._write_tables
    original_quantize = job.quantize_component_into
    original_cuda = job.torch.cuda.is_available
    original_tf32 = job.torch.backends.cuda.matmul.allow_tf32
    original_precision = job.torch.get_float32_matmul_precision()
    try:
        # Explicit computation seams only; all stage control stays in four_lane.
        module: Any = job
        module._write_tables = recorder.tables
        module.quantize_component_into = recorder.quantize
        module.torch.cuda.is_available = lambda: True
        source = job.H3FullTransformer.for_test()
        return module.four_lane(
            None, job.ProductionRequest(), source, source, recorder, Telemetry()
        )
    finally:
        job._write_tables = original_tables
        job.quantize_component_into = original_quantize
        job.torch.cuda.is_available = original_cuda
        job.torch.set_float32_matmul_precision(original_precision)
        job.torch.backends.cuda.matmul.allow_tf32 = original_tf32


def main() -> None:
    slots = ("bf16-full", "bf16-adaln-pruned", "fp8-adaln-pruned", "mxfp8-adaln-pruned")
    for failure, retained in (
        ("tables:ref2va", slots[:1]),
        ("quantize:fp8-adaln-pruned", slots[:2]),
        ("quantize:mxfp8-adaln-pruned", slots[:3]),
    ):
        recorder = Recorder()
        try:
            invoke(recorder, failure)
        except Interrupted:
            pass
        else:
            raise AssertionError("injected computation failure did not fire")
        assert tuple(recorder.committed) == retained, (failure, tuple(recorder.committed))

        recorder.events.clear()
        result = invoke(recorder)
        assert tuple(recorder.committed) == slots
        assert result.replayed_outputs == len(retained)
        assert {(row.output_slot, row.component) for row in result.weight_fidelity_this_run} == {
            (slot, component)
            for slot in set(slots[2:]) - set(retained)
            for component in ("fl2va_dit", "ref2va_dit")
        }
        for row in result.weight_fidelity_this_run:
            assert row.stats.saturated_elements == 2
            assert row.stats.worst_relative_frobenius == 0.03125
        assert not any(event == "commit:" + slot for slot in retained for event in recorder.events)
        for slot in retained:
            assert not any(event.startswith("quantize:" + slot) for event in recorder.events)

        recorder.events.clear()
        result = invoke(recorder)
        assert result.replayed_outputs == 4
        assert result.weight_fidelity_this_run == []
        assert result.source_bytes_read_this_run == result.quantized_keys_this_run == 0
        assert recorder.events == []
    print(
        "H3 stage commits PASS table/FP8/MXFP8 interruption, "
        "completed receipt replay, full no-work replay"
    )


if __name__ == "__main__":
    main()
