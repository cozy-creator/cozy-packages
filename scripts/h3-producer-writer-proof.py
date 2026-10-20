#!/usr/bin/env python3
"""Real native producer reads/writes, encoders, casts, replay and byte refusals.

This CPU proof binds public bounded Outputs to its own temporary TensorFS store. It makes
no claim about TensorD/Hub custody, full-model quality, CUDA kernels or GPU execution.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import tensorfs
from cozy_runtime.author import CapabilityError, Context, ModelArtifact, canonical_json
from cozy_runtime.author.fakes import fake_telemetry
from cozy_runtime.derive import Derivation, Output
from cozy_runtime.derive.quantization import ArtifactQuantizationRequest
from h3_tables import job, lanes
from h3_tables.metadata import (ComponentSelection, PartShape, TensorShape, PLAIN, SCHEMES,
                                inspect_checkpoint, model_output, adopt_output)


def context(store: Path, name: str, bound: int) -> Context:
    def output(slot: str, derivation: Derivation) -> Output:
        key = canonical_json.digest([name, slot, derivation.work(bound)])
        return Output(store, slot, key, derivation, bound)

    def adopt(value: Output) -> ModelArtifact:
        return ModelArtifact(name, value.slot, value.commit().id, value.transaction)

    return Context(name, math.inf, _tensorfs_output=output, _tensorfs_adopt=adopt)


def fixture(store: Path) -> tensorfs.Checkpoint:
    with tensorfs.Writer(cache_dir=store) as writer:
        held = writer.component("toy", cls="Example", config={"inputs": 256})
        held.put("w.weight", np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 256))
        held.put("keep", b"keep", dtype="U8", shape=[4])
        held.asset("NOTICE", b"original")
        return writer.commit()


def prove_copy(store: Path, source: tensorfs.Checkpoint) -> None:
    ctx = context(store, "copy-proof", 11)
    selected = ComponentSelection("source", "toy", drop=("w.weight",), add={
        "new": TensorShape("u8", (4,), PLAIN, {"value": PartShape("u8", (4,))})})
    order = (("renamed", "keep"), ("renamed", "new"))
    artifact = None
    for attempt in range(2):
        with model_output(ctx, "model", sources={"source": source},
                          components={"renamed": selected},
                          configs={"model": canonical_json.encode({"prepared": True})},
                          order=order) as out:
            if out.done is None:
                assert attempt == 0
                out.put("renamed", "new", "U8", [4], b"data")
                out.asset("renamed", "LICENSE", b"license")
                assert out.written == 11
            else:
                assert attempt == 1 and out.written == 0
            current = adopt_output(ctx, out)
            if artifact is not None:
                assert current == artifact
            artifact = current
    assert artifact is not None
    with tensorfs.open(artifact.checkpoint, cache_dir=store, local_files_only=True) as got:
        assert set(got.components["renamed"].tensors) == {"keep", "new"}
        assert got.components["renamed"].cls == "Example"
        assert got.components["renamed"].config == {"inputs": 256}
        assert got.components["model"].config == {"prepared": True}
        assert got.components["renamed"].asset("NOTICE") == b"original"
        assert got.components["renamed"].asset("LICENSE") == b"license"
        for key, expected in (("keep", b"keep"), ("new", b"data")):
            value = got.components["renamed"].tensors.read(key)
            assert isinstance(value, tensorfs.Array) and value.tobytes() == expected
    incomplete = context(store, "incomplete-proof", 11)
    with model_output(incomplete, "model", sources={"source": source},
                      components={"renamed": selected}, configs={}, order=order) as out:
        try:
            adopt_output(incomplete, out)
        except ValueError as error:
            assert "unfinished" in str(error) and out.done is None
        else:
            raise AssertionError("an output missing its new tensor was adopted")
    with context(store, "bound-proof", 3).output("model", Derivation(None)) as out:
        try:
            out.put("new", "w", "U8", [4], b"data")
        except CapabilityError as error:
            assert error.code == "weights_output_bound"
        else:
            raise AssertionError("an oversized payload was admitted")
        assert not out.has("new", "w")


def prove_cast(store: Path, source: tensorfs.Checkpoint) -> None:
    ctx = context(store, "cast-proof", 1024)
    selection = lanes.select("toy", lanes.Treatment(cast="bf16"), inspect_checkpoint(source))
    selected = lanes.apply(ComponentSelection("source", "toy"), selection)
    with model_output(ctx, "model", sources={"source": source}, components={"toy": selected},
                      configs={}, order=(("toy", "w.weight"), ("toy", "keep"))) as out:
        stats = lanes.write_cast(out, ctx, fake_telemetry(ctx=ctx), selection=selection, source=source,
                                 source_component="toy", target_component="toy")
        assert stats.converted_keys == 1 and stats.source_bytes_read == 2048
        assert stats.new_bytes_written == 1024 and out.written == 1024
        artifact = adopt_output(ctx, out)
    assert artifact is not None
    with tensorfs.open(artifact.checkpoint, cache_dir=store, local_files_only=True) as got:
        tensor = got.components["toy"].tensors["w.weight"]
        assert tensor.dtype == "BF16" and tensor.encoding is None
        value = got.components["toy"].tensors.read("w.weight")
        assert isinstance(value, tensorfs.Array) and len(value.tobytes()) == 1024


def prove_encodings(store: Path, source: tensorfs.Checkpoint) -> None:
    for encoding, scheme in SCHEMES.items():
        ctx = context(store, scheme, 2048)
        selection = lanes.select("toy", lanes.Treatment(encode=encoding), inspect_checkpoint(source))
        selected = lanes.apply(ComponentSelection("source", "toy"), selection)
        artifact = None
        for attempt in range(2):
            with model_output(ctx, "model", sources={"source": source}, components={"toy": selected},
                              configs={}, order=(("toy", "w.weight"), ("toy", "keep"))) as out:
                stats = job._treat(out, ctx, fake_telemetry(ctx=ctx), ArtifactQuantizationRequest(),
                                   selection=selection, source=source)
                assert stats.encoded_keys == 1 - attempt and stats.reused_keys == attempt
                assert stats.source_bytes_read == 2048 * (1 - attempt)
                expected = 292 if scheme == "nvfp4" else 520
                assert stats.new_bytes_written == expected * (1 - attempt)
                assert out.written == expected * (1 - attempt)
                current = adopt_output(ctx, out)
                if artifact is not None:
                    assert artifact == current
                artifact = current
        assert artifact is not None
        with tensorfs.open(artifact.checkpoint, cache_dir=store, local_files_only=True) as got:
            tensor = got.components["toy"].tensors["w.weight"]
            assert tensor.encoding is not None and tensor.encoding.scheme == scheme
            decoded = np.empty((2, 256), dtype=np.float32)
            got.components["toy"].tensors.read_into("w.weight", decoded, decode=True)
            expected_values = np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 256)
            relative = float(np.linalg.norm(decoded - expected_values) / np.linalg.norm(expected_values))
            assert np.isfinite(decoded).all() and relative < 0.1, (scheme, relative)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="h3-writer-proof-") as directory:
        store = Path(directory)
        source = fixture(store)
        prove_copy(store, source)
        prove_cast(store, source)
        prove_encodings(store, source)
    print("H3 native Writer PASS metadata/copy/drop/rename/assets/bounds/BF16/fp8/int8/nvfp4/replay")


if __name__ == "__main__":
    main()
