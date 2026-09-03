#!/usr/bin/env python3
"""Refuse package encoding identities outside TensorFS's installed seed set."""

from h3_tables.job import FP8_SPEC, MXFP8_SPEC, PLAIN_SPEC
from tensorfs import seed_digests


def main() -> None:
    expected = {
        ("plain/1", PLAIN_SPEC),
        ("fp8-rowwise/1", FP8_SPEC),
        ("mxfp8/1", MXFP8_SPEC),
    }
    missing = expected - set(seed_digests())
    if missing:
        raise RuntimeError(f"package encoding identities are not TensorFS seeds: {sorted(missing)}")
    print("TensorFS seed identities PASS plain=1 fp8-rowwise=1 mxfp8=1")


if __name__ == "__main__":
    main()
