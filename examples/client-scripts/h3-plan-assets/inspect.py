# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30,<0.31", "tensorfs>=0.13,<0.14"]
# [tool.cozy]
# accelerator = false
# ///
"""Read a granted checkpoint's header and plan references without reading tensor data."""

import json
from typing import Any

from cozy_runtime.author import Context, Weights


def main(ctx: Context, *, source: Weights[Any]) -> dict[str, str]:
    checkpoint = ctx.tensorfs_source(source)
    plans = {
        name: component.assets["timestep-plan.json"]
        for name, component in checkpoint.components.items()
        if "timestep-plan.json" in component.assets
    }
    missing = set(checkpoint.missing())
    return {
        "header": checkpoint.id,
        "tree": checkpoint.tree or "",
        "variant": checkpoint.variant or "",
        "plans": json.dumps(plans, sort_keys=True),
        "tensor_chunk_count": str(len(checkpoint.chunks)),
        "missing_tensor_chunk_count": str(sum(chunk in missing for chunk in checkpoint.chunks)),
    }
