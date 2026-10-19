# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30.0,<1", "tensorfs>=0.12"]
# [tool.cozy]
# accelerator = false
# [tool.cozy.weights]
# original = 1
# ///
"""Return genuine native custody before a separate caller delegates H3 bank work."""
from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Weights
from cozy_runtime.tensord_channels import Derivation


def main(ctx: Context, *, source: Weights[Any]) -> ModelArtifact:
    opened = ctx.tensorfs_source(source)
    tensors = opened.checkpoint.tensors
    for component in ("transformer", "transformer_ref"):
        if "transformer_blocks.0.adaln_proj.linear.weight" not in tensors[component]:
            raise ValueError("positive bank qualification requires original generating weights")
        if "transformer_blocks.0.adaln_proj.table" in tensors[component]:
            raise ValueError("an already pruned model is not a positive bank source")
    order = [(name, key) for name, rows in tensors.items() for key in rows]
    with ctx.output("original", Derivation(base=opened, order=order)) as out:
        return ctx.adopt_model(out)
