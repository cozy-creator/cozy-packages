#!/usr/bin/env python
"""Derive the committed MiniMax-H3 timestep plans from the official Diffusers scheduler.

    python scripts/h3_plans.py 30 40 50

Each argument is a denoise step count (transformer evaluations); its sigma grid is
`MiniMaxH3Scheduler.set_timesteps(steps + 1)` per modality, exactly as the official
pipeline spaces `num_inference_steps`. Writes both task plans for the serving package and
the producer, and restamps the producer's model config with the new plan identities.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H3 = ROOT / "minimax-h3"
TOOLS = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets"
sys.path.insert(0, str(H3))

from cozy_runtime.author import canonical_json  # noqa: E402

from official import Schedule, Task, TimestepPlan  # noqa: E402

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
TASKS: tuple[Task, ...] = ("fl2va", "ref2va")


def official_schedule(steps: int) -> Schedule:
    from diffusers import MiniMaxH3Scheduler

    sigmas = []
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps + 1)
        sigmas.append(tuple(float(value) for value in scheduler.sigmas.float().cpu()))
    schedule = Schedule(steps + 1, sigmas[0], sigmas[1])
    if schedule.transformer_evaluations != steps:
        raise SystemExit(f"{steps} steps collide on the float32 grid; pick another count")
    return schedule


def main(arguments: list[str]) -> None:
    steps = sorted({int(value) for value in arguments})
    if not steps:
        raise SystemExit(__doc__)
    schedules = tuple(official_schedule(count) for count in steps)
    digests: dict[str, str] = {}
    for task in TASKS:
        plan = TimestepPlan(task, VIDEO_SHIFT, AUDIO_SHIFT, schedules)
        raw = plan.canonical_bytes()
        (H3 / "timestep-plans" / f"{task}.json").write_bytes(raw)
        (TOOLS / f"timestep-plan.{task}.json").write_bytes(raw)
        digests[task] = f"sha256:{plan.digest}"
        final_rows, block_rows = plan.table_layout()
        print(
            f"{task}: {digests[task]} steps={plan.steps} "
            f"final_rows={len(final_rows)} block_rows={len(block_rows)}"
        )

    config_path = TOOLS / "model-config.json"
    config = canonical_json.decode(config_path.read_bytes())
    for component, task in (("fl2va_dit", "fl2va"), ("ref2va_dit", "ref2va")):
        config[component]["cozy_h3"]["timestep_plan_digest"] = digests[task]
    raw = canonical_json.encode(config)
    config_path.write_bytes(raw)
    print(f"model-config.json: {canonical_json.digest(config)} length={len(raw)}")


if __name__ == "__main__":
    main(sys.argv[1:])
