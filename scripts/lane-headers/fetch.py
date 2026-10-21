#!/usr/bin/env python3
"""Fetch the cozytensors headers `scripts/lane-plans-proof.py` resolves lane plans against: each
family's source lane on a Hub, its header only (no weights).

    <package>/.venv/bin/python scripts/lane-headers/fetch.py http://127.0.0.1:8819 paul [header...]
    <package>/.venv/bin/python scripts/lane-headers/fetch.py https://tensorhub.com fidika
"""

import sys
from pathlib import Path

import tensorfs

#: header -> (model, release, lane)
HEADERS = {
    "sdxl": ("sdxl", "1.0.0", "bf16"),
    "anima": ("anima", "1.0.0", "bf16"),
    "seedvr2": ("seedvr2-7b", "0.1.0", "bf16"),
    "qwen-image-2": ("qwen-image-2.1", "0.1.0", "bf16"),
    "qwen3.5-9b": ("qwen3.5-9b", "0.2.0", "bf16"),
    "qwen3.5-9b-mtp": ("qwen3.5-9b-mtp", "0.2.0", "bf16"),
    "flux-3-klein": ("flux-3-klein", "0.1.0", "bf16"),
    "flux-3-klein-upsampler": ("flux-3-klein-upsampler", "0.2.0", "bf16"),
    "birefnet": ("birefnet", "0.1.0", "fp16"),
    "minimax-h3": ("minimax-h3", "1.0.0-rc.3", "bf16"),
    "minimax-h3-full": ("minimax-h3", "1.0.0-rc.3", "bf16-full"),
}


def _end(b: bytes, i: int = 0) -> int:
    """The offset just past the CBOR item at `i` (the header is one array); IndexError: more."""
    head = b[i]
    major, info, i = head >> 5, head & 31, i + 1
    if info >= 28:
        raise ValueError(f"CBOR additional info {info}")
    if info >= 24:
        size = 1 << (info - 24)
        info, i = int.from_bytes(b[i : i + size], "big"), i + size
        if i > len(b):
            raise IndexError
    if major in (2, 3):
        if i + info > len(b):
            raise IndexError
        return i + info
    for _ in range({4: info, 5: 2 * info, 6: 1}.get(major, 0)):
        i = _end(b, i)
    return i


def header(ck: tensorfs.Checkpoint) -> bytes:
    f, got = ck.open_file(), b""
    while True:
        got += f.read(1 << 20)
        try:
            return got[: _end(got)]
        except IndexError:
            continue


def main() -> None:
    endpoint, account, only = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    for name, (model, release, lane) in HEADERS.items():
        if only and name not in only:
            continue
        ck = tensorfs.open(f"{account}/{model}@{release}", variant=lane, endpoint=endpoint)
        (Path(__file__).parent / f"{name}.cozytensors").write_bytes(header(ck))
        print(name, {c: len(component.tensors) for c, component in ck.components.items()}, flush=True)


if __name__ == "__main__":
    main()
