# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30.0,<1", "tensorfs>=0.13,<0.14"]
# [tool.cozy]
# accelerator = false
# [tool.cozy.weights]
# original = 1
# ///
"""Return genuine native custody before a separate caller delegates H3 bank work."""
from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Weights
from cozy_runtime.derive import Derivation


def main(ctx: Context, *, source: Weights[Any]) -> ModelArtifact:
    capability = ctx.tensorfs_source(source)
    for component in ("transformer", "transformer_ref"):
        tensors = capability.components[component].tensors
        if "transformer_blocks.0.adaln_proj.linear.weight" not in tensors:
            raise ValueError("positive bank qualification requires original generating weights")
        if "transformer_blocks.0.adaln_proj.table" in tensors:
            raise ValueError("an already pruned model is not a positive bank source")
    order = tuple((name, key) for name, component in capability.components.items()
                  for key in component.tensors)
    with ctx.output("original", Derivation(capability, order=order)) as output:
        return ctx.adopt_model(output)
