# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.16.5", "minimax-h3-tools==2.12.1"]
# [tool.uv.sources]
# minimax-h3-tools = { path = "../../minimax-h3-tools", editable = true }
# [tool.cozy.models]
# source = "paul/minimax-h3#sha256:d1b1fd76b6b67ad55c0d1692076ed32835229dfa4a34c0ad7b12645ba50ec2a5"
# [tool.cozy.weights]
# bf16_full = 6442450944
# ///
"""One-off H3 repair stage one: cast the shared VAE with one native writer."""

from cozy_runtime.author import (
    Context,
    Model,
    ModelArtifact,
    Telemetry,
    WeightsConfig,
    WeightsSink,
    WeightsTarget,
)
from h3_tables import lanes

SOURCE = "sha256:d1b1fd76b6b67ad55c0d1692076ed32835229dfa4a34c0ad7b12645ba50ec2a5"
MAX_NEW_BYTES = 6 << 30


def main(
    ctx: Context, *, source: Model[object], artifacts: WeightsSink, tel: Telemetry
) -> ModelArtifact:
    if source.checkpoint_ref != SOURCE:
        raise ValueError("stage one requires the reviewed BF16-full checkpoint")
    structure = artifacts.structure(source)
    if {row.component for row in structure.tensors} != set(lanes.COMPONENTS):
        raise ValueError("source must contain all five H3 components")
    selection = lanes.select(
        "video_vae", lanes.NORMALISED_COMPONENTS["video_vae"], structure.tensors
    )
    targets = {
        component: WeightsTarget(source="source", source_component=component)
        for component in lanes.COMPONENTS
    }
    targets["video_vae"] = lanes.apply(targets["video_vae"], selection)
    with artifacts.open(
        "bf16_full",
        sources={"source": source},
        targets=targets,
        configs={
            name: WeightsConfig(source="source", source_config=name) for name in structure.configs
        },
        order=tuple((row.component, row.key) for row in structure.tensors),
    ) as transaction:
        if transaction.replayed:
            assert transaction.receipt is not None
            return transaction.receipt.artifact
        stats = lanes.write_cast(
            transaction,
            ctx,
            tel,
            selection=selection,
            source="source",
            source_component="video_vae",
            target_component="video_vae",
        )
        tel.log(
            "shared VAE cast",
            cast_keys=stats.converted_keys,
            reused_keys=stats.reused_keys,
            source_bytes=stats.source_bytes_read,
            new_bytes=stats.new_bytes_written,
            worst_relative_frobenius=stats.worst_relative_frobenius,
        )
        return transaction.commit().artifact
