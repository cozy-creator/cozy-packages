# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.16.10,<1", "cozy-mixed-model-qualification==0.0.1"]
# [tool.uv]
# default-groups = []
# [tool.uv.sources]
# cozy-mixed-model-qualification = {path = ".", editable = true}
# ///
"""Compare actual mixed GPU calls, then optionally hold a child for CLI cancellation."""

import json

from cozy_runtime.author import ScriptContext
from mixed_model_qualification import candidate, combine

NOTE = "initial"
HOLD_FOR_CANCEL = False


def result_fields(value: object) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in (
            "seed",
            "steps",
            "candidate_checkpoint",
            "base_checkpoint",
            "input_sum",
            "candidate_value",
            "base_value",
            "combined_value",
            "device",
            "cpu_rng",
            "cuda_rng",
            "next_noise",
        )
    }


async def main(ctx: ScriptContext) -> str:
    model = await candidate()
    first = combine(model=model, seed=24680, steps=2)
    baseline = await first
    if (
        baseline.candidate_checkpoint != model.manifest.digest
        or baseline.base_checkpoint == model.manifest.digest
    ):
        raise ValueError("the serving slots did not retain their separate checkpoint identities")
    second = combine(model=model, seed=24680, steps=2)
    repeated = await second
    if result_fields(baseline) != result_fields(repeated):
        raise ValueError("identical seeded calls produced different numerical or RNG results")
    if not first.request_id or not second.request_id or first.request_id == second.request_id:
        raise ValueError("inference calls did not receive independent request identities")
    report = {
        "note": NOTE,
        "producer_request_id": model.producer_request_id,
        "candidate_checkpoint": model.manifest.digest,
        "first_request_id": first.request_id,
        "second_request_id": second.request_id,
        "result": result_fields(baseline),
    }
    ctx.log(
        f"Mixed GPU comparison passed; producer={model.producer_request_id}, "
        f"calls={first.request_id},{second.request_id}"
    )
    if HOLD_FOR_CANCEL:
        await combine(model=model, seed=24680, steps=2, hold_for_cancel=True)
    return json.dumps(report, sort_keys=True)
