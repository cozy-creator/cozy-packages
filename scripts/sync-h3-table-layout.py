#!/usr/bin/env python3
"""Check the producer's copy of H3 row parsing; --write updates it from the canonical source."""

import argparse
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source = files("cozy_runtime.models.minimax_h3").joinpath("table_layout.py")
    target = ROOT / "minimax-h3-tools/src/h3_tables/_table_layout.py"
    if args.write:
        target.write_bytes(source.read_bytes())
    if target.read_bytes() != source.read_bytes():
        raise SystemExit("H3 table-layout copy differs; run this script with --write")
    print("H3 table-layout source and producer copy are byte-identical")


if __name__ == "__main__":
    main()
