# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.0,<1", "minimax-h3-tools>=2.12.1"]
# [tool.uv.sources]
# minimax-h3-tools = { path = "../../minimax-h3-tools", editable = true }
# [tool.cozy.models]
# bf16_pruned = "paul/minimax-h3#sha256:d4499d1e7125b9b4c62a5a926c1fcc3e8069bfb5b9a95aa1eb785b05663303e2"
# fp8_pruned = "paul/minimax-h3#sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275"
# mxfp8_pruned = "paul/minimax-h3#sha256:f2e16ba7bc09be3d7ad76d66c336e2c1ebab55770b777c84b43f3719a17fc991"
# [tool.cozy.weights]
# bf16_pruned = 8388608
# fp8_pruned = 8388608
# mxfp8_pruned = 8388608
# ///
# ruff: noqa: E501
"""One-off stage two: inherit stage one's VAE into the three existing pruned lanes.

Supply model.updated_bf16=paul/minimax-h3#<stage-one-checkpoint> explicitly.
Only changed metadata is written; no tensor payload is read or added.
"""

from importlib.resources import files

from cozy_runtime.author import (
    Context,
    Model,
    ModelArtifact,
    Telemetry,
    canonical_json,
)
from h3_tables import lanes
from h3_tables.legacy_config import upgrade_legacy_table_config
from h3_tables.plans import TASKS, parse_declared_plan
from h3_tables.source import inspection
from tensorfs.derived import Config, Derivation, Part, Target

SOURCES = {
    "bf16_pruned": "sha256:d4499d1e7125b9b4c62a5a926c1fcc3e8069bfb5b9a95aa1eb785b05663303e2",
    "fp8_pruned": "sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275",
    "mxfp8_pruned": "sha256:f2e16ba7bc09be3d7ad76d66c336e2c1ebab55770b777c84b43f3719a17fc991",
}
MAX_NEW_BYTES = 8 << 20


def main(
    ctx: Context,
    *,
    updated_bf16: Model[object],
    bf16_pruned: Model[object],
    fp8_pruned: Model[object],
    mxfp8_pruned: Model[object],
    tel: Telemetry,
) -> dict[str, ModelArtifact]:
    sources = dict(bf16_pruned=bf16_pruned, fp8_pruned=fp8_pruned, mxfp8_pruned=mxfp8_pruned)
    if {name: source.checkpoint_ref for name, source in sources.items()} != SOURCES:
        raise ValueError("stage two requires the three reviewed pruned checkpoints")
    shared = inspection(ctx, updated_bf16)
    shared_vae = shared.components.get("video_vae", {})
    if not shared_vae:
        raise ValueError("the stage-one checkpoint has no VAE")
    shared_config = canonical_json.decode(shared.configs["model"])
    plans = {
        task: parse_declared_plan(
            files("h3_tables").joinpath("assets", f"timestep-plan.{task}.json").read_bytes()
        )
        for task in TASKS
    }
    result = {}
    for done, (name, source) in enumerate(sources.items(), 1):
        ctx.raise_if_cancelled()
        structure = inspection(ctx, source)
        if set(structure.components) != set(lanes.COMPONENTS):
            raise ValueError(f"{name}: expected five H3 components")
        original = structure.components.get("video_vae", {})
        selection = lanes.select("video_vae", lanes.NORMALISED_COMPONENTS["video_vae"], structure)
        cast = {key for key, _ in selection.cast}
        if original.keys() != shared_vae.keys():
            raise ValueError(f"{name}: VAE keys differ from stage one")
        for key, before in original.items():
            after = shared_vae[key]
            if key in cast:
                if (
                    after.logical_dtype != "f16"
                    or after.shape != before.shape
                    or after.encoding != before.encoding
                    or after.parts != {"value": Part("f16", before.shape)}
                ):
                    raise ValueError(f"{name}: stage-one cast geometry differs for {key}")
            elif after != before:
                raise ValueError(f"{name}: preserved VAE tensor geometry differs for {key}")
        raw = structure.configs["model"]
        config = upgrade_legacy_table_config(raw, plans)
        document = canonical_json.decode(config)
        if document["video_vae"] != shared_config["video_vae"]:
            raise ValueError(f"{name}: VAE config differs from stage one")
        if any(
            document[f"{task}_dit"]["cozy_h3"]["modulation"] != "adaln-pruned" for task in TASKS
        ):
            raise ValueError(f"{name}: expected pruned DiTs")
        targets = {
            component: Target(source="lane", source_component=component)
            for component in lanes.COMPONENTS
        }
        targets["video_vae"] = Target(source="shared", source_component="video_vae")
        with ctx.output(name).open(
            Derivation(
                sources={"lane": structure.source, "shared": shared.source},
                targets=targets,
                configs={
                    key: (
                        Config("add")
                        if key == "model" and config != raw
                        else Config("copy", source="lane", source_config=key)
                    )
                    for key in structure.configs
                },
                order=tuple(
                    (component, key)
                    for component, rows in structure.components.items()
                    for key in rows
                ),
            )
        ) as transaction:
            if (
                transaction.receipt is None
                and config != raw
                and "model" not in transaction.completed_configs()
            ):
                transaction.add_config("model", config)
            result[name] = ctx.adopt_model(transaction.commit())
        tel.progress(done / len(sources), stage="inherit-shared-vae")
    return result
