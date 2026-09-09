#!/usr/bin/env python3
"""PDD forward parity at Ulysses degrees 2/4 on CPU Gloo and tiny real H3 weights.

This checks package hooks, full component scopes and CP math. Runtime's follower
transport and real uneven NCCL layouts have their own integration gates.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from cozy_runtime.internal.parallel import cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def rank(rank: int, degree: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    fixture = runpy.run_path(str(ROOT / "scripts/h3-conform.py"))
    torch.manual_seed(512)
    config = {**fixture["TURBO_CONFIG"], "num_attention_heads": 4}
    dit = fixture["tiny_pruned_dit"](config)
    overlay = fixture["tiny_overlay"](config)
    with torch.no_grad():
        for parameter in dit.parameters():
            parameter.normal_(0, 0.02)
        for parameter in overlay.parameters():
            parameter.normal_(0, 0.02)
    dit.attach_overlay(fixture["TURBO_BANK"], overlay)
    dit.set_attention_backend("native")
    pipe = object.__new__(fixture["official"].OfficialH3TurboPipeline)
    pipe.components = {"fl2va_dit": dit, "fl2va_turbo": overlay}
    model = fixture["package"].H3TurboModel.for_test(pipe=pipe)
    schedule = overlay.schedule
    inputs = fixture["turbo_forward"](0, schedule)
    inputs["attention_kwargs"] = {fixture["ATTENTION_KWARG"]: fixture["TURBO_BANK"]}
    with torch.no_grad():
        expected = dit(**inputs)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=degree
    )
    try:
        cp.install_context_parallel(
            model,
            degree=degree,
            comms=cp.CpComms(torch.distributed.group.WORLD, rank, torch.device("cpu")),
        )
        with (
            torch.no_grad(),
            model._cozy_scope("sample_fl2va", ("fl2va_dit", "fl2va_turbo")),
            cp.gated_call(),
        ):
            actual = dit(**inputs)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        assert dit._arming is None
        assert model._cozy_active is None
        if rank == 0:
            print(f"PDD Ulysses degree {degree}: full scope, permanent hooks, output parity")
    finally:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--rendezvous")
    args = parser.parse_args()
    if args.rank is not None:
        rank(args.rank, args.degree, args.rendezvous)
        return
    for degree in (2, 4):
        with tempfile.TemporaryDirectory(prefix="pdd-cp-") as directory:
            children = [
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--rank",
                        str(i),
                        "--degree",
                        str(degree),
                        "--rendezvous",
                        "file://" + str(Path(directory) / "join"),
                    ]
                )
                for i in range(degree)
            ]
            codes = [child.wait() for child in children]
            assert codes == [0] * degree, codes


if __name__ == "__main__":
    main()
