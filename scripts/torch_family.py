#!/usr/bin/env python
"""Validate one installed package's native dependency requirements.

Package environments are independent: different packages may use different Torch
families. Their own wheel metadata is authoritative for Torch/Torchvision/CUDA
compatibility. The package import proof additionally loads the real compiled peers.
"""

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PREFIXES = ("torch", "triton", "nvidia-", "cuda-")


def cohort_errors(
    versions: Mapping[str, str],
    requirements: Mapping[str, Sequence[str]],
) -> list[str]:
    """Check active native dependency edges inside one environment only."""
    errors = []
    for name, declared in requirements.items():
        if not name.startswith(PREFIXES):
            continue
        for value in declared:
            dependency = Requirement(value)
            target = canonicalize_name(dependency.name)
            if not target.startswith(PREFIXES):
                continue
            if dependency.marker is not None and not dependency.marker.evaluate({"extra": ""}):
                continue
            actual = versions.get(target)
            if actual is None or not dependency.specifier.contains(actual, prereleases=True):
                errors.append(
                    f"{name} requires {dependency}; installed {target} is {actual or 'absent'}"
                )
    return errors


def check_current_environment() -> None:
    versions: dict[str, str] = {}
    requirements: dict[str, Sequence[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = canonicalize_name(distribution.metadata["Name"])
        if not name.startswith(PREFIXES):
            continue
        if name in versions and versions[name] != distribution.version:
            raise ValueError(f"{name} has multiple versions visible in one package environment")
        versions[name] = distribution.version
        requirements[name] = distribution.requires or ()
    errors = cohort_errors(versions, requirements)
    if errors:
        raise ValueError("incompatible package native dependencies:\n  " + "\n  ".join(errors))
    print(
        f"native dependencies: this environment satisfies {len(versions)} installed cohort members"
    )


def self_check() -> None:
    # Independent compatible environments are both valid; no cross-package version equality.
    first = {"torch": "2.13.0", "torchvision": "0.28.0", "triton": "3.7.1"}
    first_requires = {"torchvision": ["torch==2.13.0"], "torch": ["triton==3.7.1"]}
    second = {"torch": "2.14.0", "torchvision": "0.29.0", "triton": "3.8.0"}
    second_requires = {"torchvision": ["torch==2.14.0"], "torch": ["triton==3.8.0"]}
    assert not cohort_errors(first, first_requires)
    assert not cohort_errors(second, second_requires)
    errors = cohort_errors({**first, "torch": "2.14.0"}, first_requires)
    assert len(errors) == 1 and "torchvision requires torch==2.13.0" in errors[0]
    errors = cohort_errors({**first, "triton": "3.8.0"}, first_requires)
    assert len(errors) == 1 and "torch requires triton==3.7.1" in errors[0]
    print(
        "native cohorts: different package versions accepted; internal mismatch refused"
    )


def main() -> None:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    elif sys.argv[1:]:
        raise SystemExit("usage: torch_family.py [--self-check]")
    else:
        check_current_environment()


if __name__ == "__main__":
    main()
