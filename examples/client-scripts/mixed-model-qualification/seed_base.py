# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.0,<1", "cozy-mixed-model-qualification==0.0.1"]
# [tool.uv]
# default-groups = []
# [tool.uv.sources]
# cozy-mixed-model-qualification = {path = "library", editable = true}
# [tool.cozy.weights]
# checkpoint = 4096
# ///
"""Create a tiny real base checkpoint; CLI publication remains explicit."""

from cozy_runtime.author import Context, ModelArtifact
from mixed_model_qualification import write_checkpoint


def main(ctx: Context) -> ModelArtifact:
    return write_checkpoint(ctx, 3)
