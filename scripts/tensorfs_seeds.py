#!/usr/bin/env python3
"""Exercise the installed native Writer/reader for every encoding the producers emit."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import tensorfs
import numpy as np
from tensorfs import raw


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tensorfs-schemes-") as directory:
        with tensorfs.Writer(cache_dir=Path(directory)) as writer:
            component = writer.component("proof", cls="", config={})
            component.put("plain", b"data", dtype="U8", shape=[4])
            for name, scheme, dtype, weight_bytes, params, scales in (
                ("fp8", "fp8.rowwise", "F8_E4M3", 256,
                 {"axis": 0},
                 {"scale": ("F32", [1], struct.pack("<f", 1))}),
                ("int8", "int8.convrot", "I8", 256,
                 {"rotation": "kron_regular_h4", "rotation_size": 256},
                 {"scale": ("F32", [1], struct.pack("<f", 1))}),
                ("nvfp4", "nvfp4", "F4", 128,
                 {"block_size": 16, "global_scale": "multiply", "nibble_order": "low_first",
                  "scale_layout": "row_major"},
                 {"scale": ("F8_E4M3", [1, 16], b"\x38" * 16),
                  "scale_2": ("F32", [], struct.pack("<f", 1))}),
            ):
                parts = {"weight": (dtype, [1, 256], bytes(weight_bytes)), **scales}
                component.put_encoded(name, dtype="BF16", shape=[1, 256], scheme=scheme,
                                      params=params, parts=parts)
            checkpoint = writer.commit()
        tensors = checkpoint.components["proof"].tensors
        plain = bytearray(4)
        tensors.read_into("plain", plain)
        if plain != b"data" or tensors["plain"].encoding is not None:
            raise RuntimeError("native plain Writer/read round trip differs")
        for key, scheme in (("fp8", "fp8.rowwise"), ("int8", "int8.convrot"), ("nvfp4", "nvfp4")):
            encoding = tensors[key].encoding
            if encoding is None or encoding.scheme != scheme:
                raise RuntimeError(f"native encoding differs: {key}")
            held = tensors.read(key)
            if not isinstance(held, tensorfs.Encoded):
                raise RuntimeError(f"{key} did not retain its native stored parts")
            if key in ("fp8", "int8") and held.parts["scale"].dtype != "F32":
                raise RuntimeError(f"native {scheme} did not retain its scale part dtype")
            into = np.empty((1, 256), dtype=np.float32)
            raw.decode_into(scheme, {name: (part.dtype, part.shape, part.tobytes())
                                    for name, part in held.parts.items()}, into,
                            dtype="F32", shape=[1, 256], params=encoding.params)
            if bool(np.any(into)):
                raise RuntimeError(f"native {scheme} zero-weight decode differs")
    print("TensorFS native schemes PASS plain=1 fp8.rowwise=1 int8.convrot=1 nvfp4=1")


if __name__ == "__main__":
    main()
