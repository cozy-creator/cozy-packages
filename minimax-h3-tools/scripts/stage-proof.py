#!/usr/bin/env python3
"""Exercise actual lane sequencing with interrupted computation and receipt replay.

This is a CPU orchestration proof. Recording collaborators replace expensive table
and quantization work and native storage; no GPU, payload custody, or numerical claim.
The callable still parses its real assets and builds its real output declarations.
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import Any

import torch
from cozy_runtime.author._calls import _export
from cozy_runtime.derive.quantization import (
    QuantizationStats,
    prepare_quantization,
)
from h3_tables import job, lanes
from h3_tables.quantization import h3_quantization_plan
from h3_tables.source import TARGET_COMPONENT, H3FullTransformer, full_targets
from tensorfs.derived import (
    Part,
    Source,
    SourceCapability,
    SourceInspection,
    Tensor,
)


class Interrupted(RuntimeError):
    pass


class Transaction:
    def __init__(self, owner: Recorder, slot: str) -> None:
        self.owner, self.slot = owner, slot
        self.replayed = slot in owner.committed
        prior = owner.committed.get(slot)
        self.receipt = dict(prior) if prior else None
        self.configured = False

    def __enter__(self) -> Transaction:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def add_config(self, name: str, raw: bytes) -> None:
        assert not self.replayed and name == "model" and raw
        self.configured = True

    def commit(self) -> dict[str, Any]:
        if self.receipt is not None:
            return self.receipt
        assert not self.replayed and self.configured
        assert self.slot not in self.owner.committed
        receipt = {"transaction_id": self.slot, "manifest": {"sha256": "11" * 32, "length": 1}}
        self.owner.committed[self.slot] = receipt
        self.owner.events.append("commit:" + self.slot)
        return receipt


class Recorder:
    def __init__(self) -> None:
        self.committed: dict[str, dict[str, Any]] = {}
        self.events: list[str] = []
        self.fail_at = ""

    def tensorfs_source(self, source: Any) -> SourceCapability:
        """Native geometry values only; this stage test makes no byte custody claim."""

        def inspect(_components: Any, _configs: Any) -> SourceInspection:
            plan = prepare_quantization(h3_quantization_plan())
            components = {
                component: {
                    key: Tensor("f32", (1,), job.PLAIN_SPEC, {"value": Part("f32", (1,))})
                    for key in target.drop
                }
                for component, target in full_targets().items()
            }
            for component in TARGET_COMPONENT.values():
                components[component].update(
                    {
                        tensor.key: Tensor(
                            tensor.logical_dtype,
                            tensor.shape,
                            job.PLAIN_SPEC,
                            {"value": Part(tensor.logical_dtype, tensor.shape)},
                        )
                        for tensor in plan.tensors
                    }
                )
            components["video_vae"].update(
                {
                    key: Tensor("f32", shape, job.PLAIN_SPEC, {"value": Part("f32", shape)})
                    for key, shape in (
                        ("decoder.proj_in.weight", (2048, 24)),
                        ("decoder.transformer_blocks.0.attn.to_q.weight", (2048, 2048)),
                        ("post_quant_conv.weight", (24, 24, 1, 1, 1)),
                        ("decoder.norm_out.weight", (2048,)),
                    )
                }
            )
            return SourceInspection(Source(source.checkpoint_ref, 1), components, {})

        return SourceCapability(source.checkpoint_ref, 1, inspect)

    def output(self, slot: str) -> Any:
        owner = self

        class Output:
            def open(self, _definition: Any) -> Transaction:
                return Transaction(owner, slot)

        return Output()

    def adopt_model(self, receipt: dict[str, Any]) -> None:
        assert receipt in self.committed.values()

    def tables(
        self, task: str, _ctx: Any, _source: Any, active: dict[str, Any], _tel: Any, _range: Any
    ) -> tuple[int, int, int]:
        event = "tables:" + task
        self.events.append(event)
        assert not self.committed.keys() & active.keys()
        if self.fail_at == event:
            raise Interrupted(event)
        return 1, 1, 0

    def quantize(self, transaction: Transaction, *_: Any, **kwargs: Any) -> QuantizationStats:
        event = "quantize:" + transaction.slot
        self.events.append(event + ":" + kwargs["target_component"])
        if self.fail_at == event:
            raise Interrupted(event)
        return QuantizationStats(
            encoded_keys=1,
            reused_keys=0,
            source_bytes_read=1,
            new_bytes_written=1,
            saturated_elements=2,
            worst_relative_frobenius=0.03125,
        )

    def cast(self, transaction: Transaction, *_: Any, **kwargs: Any) -> Any:
        """The video VAE cast, recorded rather than run.

        Same seam as `tables` and `quantize`: this proof sequences lanes and never claims a
        byte. The cast's own numerics are h3a-027's `h3-vae-precision-proof.py`.
        """
        event = "cast:" + transaction.slot
        self.events.append(event + ":" + kwargs["target_component"])
        if self.fail_at == event:
            raise Interrupted(event)
        return lanes.CastStats(
            converted_keys=3,
            reused_keys=0,
            source_bytes_read=1,
            new_bytes_written=1,
            worst_relative_frobenius=0.0,
        )


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
    module: Any = job
    original_tables = job._write_tables
    original_quantize = module.quantize_component_into
    original_cast = module.write_cast
    original_cuda = torch.cuda.is_available
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    try:
        # Explicit computation seams only; all stage control stays in the lanes job.
        module._write_tables = recorder.tables
        module.quantize_component_into = recorder.quantize
        module.write_cast = recorder.cast
        module.torch.cuda.is_available = lambda: True
        source = H3FullTransformer.for_test()
        implementation = _export(job.lanes).implementation
        return asyncio.run(
            implementation(
                recorder, source=source, lanes=None, max_relative_frobenius=None, tel=Telemetry()
            )
        )
    finally:
        job._write_tables = original_tables
        module.quantize_component_into = original_quantize
        module.write_cast = original_cast
        torch.cuda.is_available = original_cuda
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


def main() -> None:
    slots = tuple(lanes.LANES)
    for failure, retained in (
        ("tables:ref2va", slots[:1]),
        ("quantize:fp8-pruned", slots[:2]),
        ("quantize:mxfp8-pruned", slots[:3]),
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
        assert [row.lane for row in result.lanes] == list(slots)
        # Every lane treats the video VAE, because the producer normalises it rather than
        # each lane authoring it; only the encoding lanes also treat the two DiTs.
        casts = {event.split(":")[1] for event in recorder.events if event.startswith("cast:")}
        assert casts == set(slots) - set(retained), casts
        encoded = {
            tuple(event.split(":")[1:])
            for event in recorder.events
            if event.startswith("quantize:")
        }
        assert encoded == {
            (slot, component)
            for slot in set(slots[2:]) - set(retained)
            for component in ("fl2va_dit", "ref2va_dit")
        }, encoded
        assert not any(event == "commit:" + slot for slot in retained for event in recorder.events)
        for slot in retained:
            assert not any(event.startswith("quantize:" + slot) for event in recorder.events)

        recorder.events.clear()
        result = invoke(recorder)
        assert [row.lane for row in result.lanes] == list(slots)
        assert recorder.events == []
    print(
        f"H3 stage commits PASS lanes={len(slots)} table/FP8/MXFP8 interruption, "
        "completed receipt replay, full no-work replay"
    )


if __name__ == "__main__":
    main()
