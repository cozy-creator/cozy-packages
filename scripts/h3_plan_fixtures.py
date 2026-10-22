"""Development fixtures using the producer's exact bytes through the model-asset reader.

Inference reads these bytes from its checkpoint. These helpers are outside every published
package and keep the numerical conformance oracle independent of that storage location.
"""

from importlib.resources import files
from typing import cast

from cozy_runtime.author import ModelAssets
from minimax_h3.official import Task, TimestepPlan, Trunk, model_plans


def plan_bytes(task: Task) -> bytes:
    name = task.replace("_turbo", ".turbo")
    return files("h3_tables").joinpath("assets", f"timestep-plan.{name}.json").read_bytes()


def plans(*, turbo: bool = False) -> dict[Trunk, TimestepPlan]:
    components = {"fl2va": "fl2va_turbo", "ref2va": "ref2va_turbo"} if turbo else {
        "fl2va": "transformer", "ref2va": "transformer_ref"
    }
    assets = ModelAssets(
        {f"{component}/timestep-plan.json": plan_bytes(cast(Task, f"{task}_turbo" if turbo else task))
         for task, component in components.items()},
        lambda _: None,
    )
    return model_plans(assets, turbo=turbo)


def canonical_timestep_plan(task: Task) -> TimestepPlan:
    return plans(turbo=task.endswith("_turbo"))[cast(Trunk, task.removesuffix("_turbo"))]


def supported_steps() -> tuple[int, ...]:
    steps = {plan.steps for plan in plans().values()}
    assert len(steps) == 1
    return steps.pop()
