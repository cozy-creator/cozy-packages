# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.25.0,<1", "tensorfs>=0.10"]
# [tool.cozy]
# accelerator = false
# [tool.cozy.weights]
# original = 1
# ///
"""Return genuine native custody before a separate caller delegates H3 bank work."""
from typing import Any

from cozy_runtime.author import Context, ModelArtifact, Weights
from tensorfs.derived import Config, Derivation, Target


def main(ctx: Context, *, source: Weights[Any]) -> ModelArtifact:
    capability = ctx.tensorfs_source(source)
    structure = capability.inspect()
    for component in ("fl2va_dit", "ref2va_dit"):
        tensors = structure.components[component]
        if "transformer_blocks.0.adaln_proj.linear.weight" not in tensors:
            raise ValueError("positive bank qualification requires original generating weights")
        if "transformer_blocks.0.adaln_proj.table" in tensors:
            raise ValueError("an already pruned model is not a positive bank source")
    with ctx.output("original").open(Derivation(
        sources={"source": capability},
        targets={name: Target("source", name) for name in structure.components},
        configs={name: Config("copy", "source", name) for name in structure.configs},
        order=tuple((name, key) for name, tensors in structure.components.items() for key in tensors),
    )) as writer:
        return ctx.adopt_model(writer.receipt if writer.receipt is not None else writer.commit())
