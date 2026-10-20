#!/usr/bin/env python3
"""Exercise actual lane sequencing with interrupted computation and receipt replay.

This is a CPU orchestration proof. Recording collaborators replace expensive table
and quantization work and native storage; no GPU, payload custody, or numerical claim.
The callable still parses its real assets and builds its real output declarations.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from cozy_runtime.author import ModelArtifact, Weights, canonical_json
import h3_tables.source as source_api
from cozy_runtime.derive.quantization import QuantizationStats
from h3_tables import job, lanes
from h3_tables.quantization import reviewed_plan
from h3_tables.source import TARGET_COMPONENT, full_targets
from source_geometry import VIDEO_VAE_ROWS, conditioner_rows, plain
from h3_tables.metadata import PartShape, TensorShape, SourceLayout, ComponentSelection


class Interrupted(RuntimeError):
    pass


class RecordedOutput:
    def __init__(self, owner: Recorder, slot: str) -> None:
        self.owner, self.slot = owner, slot
        self.done = owner.committed.get(slot)
        self.configured = False
        self.order = ()

    def has(self, _component: str, _key: str) -> bool:
        return True  # Control-only recorder; actual native bytes are a separate proof.

    def __enter__(self) -> RecordedOutput:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class Recorder:
    def __init__(self) -> None:
        self.committed: dict[str, ModelArtifact] = {}
        self.events: list[str] = []
        self.fail_at = ""

    def inspect(self, source: Any) -> SourceLayout:
        """Geometry-only fixture; no native source or stored payload is claimed here."""
        plan = reviewed_plan()
        components = {
            component: {key: plain("f32", (1,)) for key in target.drop}
            for component, target in full_targets().items()
        }
        for component in TARGET_COMPONENT.values():
            components[component].update({
                tensor.key: plain(tensor.logical_dtype, tensor.shape) for tensor in plan.tensors
            })
        components["text_encoder"] = {key: plain(*spec) for key, spec in conditioner_rows().items()}
        components["vae"] = {key: plain(*spec) for key, spec in VIDEO_VAE_ROWS.items()}
        return SourceLayout(None, components, {})

    def model_output(self, _ctx: Any, slot: str, *, configs: Any, order: Any, **_: Any) -> RecordedOutput:
        output = RecordedOutput(self, slot)
        output.configured = bool(configs.get("model"))
        output.order = order
        return output

    def adopt_model(self, output: RecordedOutput) -> ModelArtifact:
        if output.done is not None:
            return output.done
        assert output.configured and output.slot not in self.committed
        artifact = ModelArtifact("stage-proof", output.slot, "sha256:" + "11" * 32,
                                 canonical_json.digest(output.slot))
        self.committed[output.slot] = artifact
        self.events.append("commit:" + output.slot)
        output.done = artifact
        return artifact

    def tables(
        self, task: str, _ctx: Any, _source: Any, active: dict[str, Any], _tel: Any, _range: Any
    ) -> tuple[int, int, int]:
        event = "tables:" + task
        self.events.append(event)
        assert not self.committed.keys() & active.keys()
        if self.fail_at == event:
            raise Interrupted(event)
        return 1, 1, 0

    def quantize(self, _source: Any, transaction: RecordedOutput, *_: Any, **kwargs: Any) -> QuantizationStats:
        event = "quantize:" + transaction.slot
        self.events.append(event + ":" + kwargs["component"])
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

    def cast(self, transaction: RecordedOutput, *_: Any, **kwargs: Any) -> Any:
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
    original_inspection = source_api.inspection
    original_model_output = job.model_output
    original_tables = job._write_tables
    original_quantize = module.quantize_component_into
    original_cast = module.write_cast
    original_cuda = torch.cuda.is_available
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    try:
        # Explicit computation seams only; all stage control stays in the lanes function.
        source_api.inspection = lambda ctx, source: ctx.inspect(source)
        module.model_output = recorder.model_output
        module._write_tables = recorder.tables
        module.quantize_component_into = recorder.quantize
        module.write_cast = recorder.cast
        module.torch.cuda.is_available = lambda: True
        source = Weights[Any]("sha256:" + "0" * 64)
        return job._produce_lanes(recorder, Telemetry(), source, tuple(lanes.LANES))
    finally:
        source_api.inspection = original_inspection
        job.model_output = original_model_output
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
        ("quantize:fp8", slots[:2]),
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
        assert list(result) == list(slots)
        # Every lane treats the video VAE, because the producer normalises it rather than
        # each lane authoring it; only the encoding lanes encode the two DiTs and the text
        # conditioner.
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
            for component in ("transformer", "transformer_ref", "text_encoder")
        }, encoded
        assert not any(event == "commit:" + slot for slot in retained for event in recorder.events)
        for slot in retained:
            assert not any(event.startswith("quantize:" + slot) for event in recorder.events)

        recorder.events.clear()
        result = invoke(recorder)
        assert list(result) == list(slots)
        assert recorder.events == []
    print(
        f"H3 stage commits PASS lanes={len(slots)} table/FP8 interruption, "
        "completed receipt replay, full no-work replay"
    )


if __name__ == "__main__":
    main()
