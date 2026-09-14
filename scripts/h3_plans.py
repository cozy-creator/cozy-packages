#!/usr/bin/env python
"""Derive the committed MiniMax-H3 timestep plans from the official Diffusers scheduler.

    python scripts/h3_plans.py 30 40 50
    python scripts/h3_plans.py --turbo 8

Each argument is a denoise step count (transformer evaluations); its sigma grid is
`MiniMaxH3Scheduler.set_timesteps(steps + 1)` per modality, exactly as the official
pipeline spaces `num_inference_steps`. Writes both task plans for the serving package and
the producer, and copies the producer's ordered table keys into its model config.
`--turbo` writes the serving package's `<task>_turbo.json` plans only: one fixed schedule,
stamped with the trunk task, which the turbo overlay's tables are keyed to.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "minimax-h3"
TOOLS = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets"
PLANS_MODULE = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "plans.py"
CONFORM_SCRIPT = ROOT / "scripts" / "h3-conform.py"
sys.path.insert(0, str(H3))

from cozy_runtime.author import canonical_json  # noqa: E402
from diffusers import MiniMaxH3Scheduler  # noqa: E402

from cozy_runtime.models.minimax_h3.official import Schedule, TimestepPlan, Trunk  # noqa: E402

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
TASKS: tuple[Trunk, ...] = ("fl2va", "ref2va")


def official_schedule(steps: int) -> Schedule:
    sigmas = []
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps + 1)
        sigmas.append(tuple(float(value) for value in scheduler.sigmas.float().cpu()))
    schedule = Schedule(steps + 1, sigmas[0], sigmas[1])
    if schedule.transformer_evaluations != steps:
        raise SystemExit(f"{steps} steps collide on the float32 grid; pick another count")
    return schedule


def _rewrite(path: Path, edits: Sequence[tuple[str, str]]) -> None:
    """Apply each (pattern, replacement) exactly once, or refuse.

    A digest this script leaves behind is a conformance failure nobody sees until a job
    refuses at load, so every site it invalidates is rewritten HERE rather than printed for
    a human to paste. A pattern that stops matching means the constant moved and this list
    is stale, which must be loud.
    """
    text = original = path.read_text()
    for pattern, replacement in edits:
        text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise SystemExit(f"{path}: {pattern!r} matched {count} sites, expected 1")
    if text != original:
        path.write_text(text)
        print(f"restamped {path.relative_to(ROOT)}")


def _restamp_constants(digests: dict[str, str]) -> None:
    """Update generation provenance and scheduler-oracle fixtures together."""
    bare = {task: digest.removeprefix("sha256:") for task, digest in digests.items()}
    _rewrite(
        PLANS_MODULE,
        [(f'("{task}": )"sha256:[0-9a-f]{{64}}"', f'\\g<1>"{digests[task]}"') for task in TASKS],
    )
    _rewrite(
        CONFORM_SCRIPT,
        [(f'("{task}": )"[0-9a-f]{{64}}"', f'\\g<1>"{bare[task]}"') for task in TASKS],
    )


def main(arguments: list[str]) -> None:
    turbo = "--turbo" in arguments
    steps = sorted({int(value) for value in arguments if value != "--turbo"})
    if not steps or (turbo and len(steps) != 1):
        raise SystemExit(__doc__)
    schedules = tuple(official_schedule(count) for count in steps)
    digests: dict[str, str] = {}
    for task in TASKS:
        plan = TimestepPlan(task, VIDEO_SHIFT, AUDIO_SHIFT, schedules)
        raw = plan.canonical_bytes()
        if turbo:
            (H3 / "timestep-plans" / f"{task}_turbo.json").write_bytes(raw)
            print(f"{task}_turbo: sha256:{plan.digest} steps={plan.steps}")
            continue
        (H3 / "timestep-plans" / f"{task}.json").write_bytes(raw)
        (TOOLS / f"timestep-plan.{task}.json").write_bytes(raw)
        digests[task] = f"sha256:{plan.digest}"
        final_rows, block_rows = plan.table_layout()
        print(
            f"{task}: {digests[task]} steps={plan.steps} "
            f"final_rows={len(final_rows)} block_rows={len(block_rows)}"
        )
    if turbo:
        return

    config_path = TOOLS / "model-config.json"
    config = canonical_json.decode(config_path.read_bytes())
    for component, task in (("fl2va_dit", "fl2va"), ("ref2va_dit", "ref2va")):
        config[component]["cozy_h3"] = {
            "task": task,
            "modulation": "adaln-pruned",
            "table_keys": canonical_json.decode(
                (TOOLS / f"timestep-plan.{task}.json").read_bytes()
            )["table_keys"],
        }
    raw = canonical_json.encode(config)
    config_path.write_bytes(raw)
    config_digest = canonical_json.digest(config)
    print(f"model-config.json: {config_digest} length={len(raw)}")
    _restamp_constants(digests)


if __name__ == "__main__":
    main(sys.argv[1:])
