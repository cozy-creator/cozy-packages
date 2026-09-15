# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.0,<1", "minimax-h3-tools>=2.12.0"]
# [tool.uv.sources]
# minimax-h3-tools = { path = "../../minimax-h3-tools", editable = true }
# [tool.cozy.models]
# bf16_full = """\
# paul/minimax-h3#sha256:d1b1fd76b6b67ad55c0d1692076ed32835229dfa4a34c0ad7b12645ba50ec2a5"""
# bf16_pruned = """\
# paul/minimax-h3#sha256:d4499d1e7125b9b4c62a5a926c1fcc3e8069bfb5b9a95aa1eb785b05663303e2"""
# fp8_pruned = """\
# paul/minimax-h3#sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275"""
# mxfp8_pruned = """\
# paul/minimax-h3#sha256:f2e16ba7bc09be3d7ad76d66c336e2c1ebab55770b777c84b43f3719a17fc991"""
# [tool.cozy.weights]
# bf16_full = 6442450944
# bf16_pruned = 6442450944
# fp8_pruned = 6442450944
# mxfp8_pruned = 6442450944
# ///
"""Repair rc2 revision 6 using its four existing DiT bodies and one shared VAE cast.

Run with the end-user CLI: cozy run ./h3_shared_vae_repair.py --rental-only
The exact source roots are intentional. This script returns four native artifacts;
publishing and changing release lane names are separate, explicit client actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
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
from tensorfs.derived import Config, Derivation, DerivedTransaction, Target

SOURCES = {
    "bf16-full": "sha256:d1b1fd76b6b67ad55c0d1692076ed32835229dfa4a34c0ad7b12645ba50ec2a5",
    "bf16-pruned": "sha256:d4499d1e7125b9b4c62a5a926c1fcc3e8069bfb5b9a95aa1eb785b05663303e2",
    "fp8-pruned": "sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275",
    "mxfp8-pruned": "sha256:f2e16ba7bc09be3d7ad76d66c336e2c1ebab55770b777c84b43f3719a17fc991",
}
MAX_NEW_BYTES = 6 << 30


def repair(
    sources: Mapping[str, Model[object]], ctx: Context, tel: Telemetry
) -> dict[str, ModelArtifact]:
    if {name: source.checkpoint_ref for name, source in sources.items()} != SOURCES:
        raise ValueError("repair requires the four reviewed H3 rc2 revision 6 roots")
    structures = {name: inspection(ctx, source) for name, source in sources.items()}
    vae = structures["bf16-full"].components["video_vae"]
    selection = lanes.select(
        "video_vae", lanes.NORMALISED_COMPONENTS["video_vae"], structures["bf16-full"]
    )
    plans = {
        task: parse_declared_plan(
            files("h3_tables").joinpath("assets", f"timestep-plan.{task}.json").read_bytes()
        )
        for task in TASKS
    }
    configs: dict[str, bytes] = {}
    for name, structure in structures.items():
        if set(structure.components) != set(lanes.COMPONENTS):
            raise ValueError(f"{name}: expected the five H3 components")
        if structure.components.get("video_vae", {}) != vae:
            raise ValueError(f"{name}: shared VAE geometry differs")
        raw = structure.configs["model"]
        # This verifies the historical plan stamp before adding explicit table_keys.
        # Already explicit layouts are preserved; dynamic full AdaLN needs no table_keys.
        configs[name] = upgrade_legacy_table_config(raw, plans)
        document = canonical_json.decode(configs[name])
        for task in TASKS:
            expected = "full" if name == "bf16-full" else "adaln-pruned"
            if document[f"{task}_dit"]["cozy_h3"]["modulation"] != expected:
                raise ValueError(f"{name}: unexpected AdaLN modulation")
        if document["video_vae"] != canonical_json.decode(configs["bf16-full"])["video_vae"]:
            raise ValueError(f"{name}: shared VAE config differs")

    result: dict[str, ModelArtifact] = {}
    with ExitStack() as stack:
        active: dict[str, DerivedTransaction] = {}
        for name in sources:
            structure = structures[name]
            targets = {
                component: Target(source="lane", source_component=component)
                for component in lanes.COMPONENTS
            }
            targets["video_vae"] = lanes.apply(
                Target(source="shared", source_component="video_vae"), selection
            )
            transaction = stack.enter_context(
                ctx.output(name.replace("-", "_")).open(
                    Derivation(
                        sources={
                            "lane": structure.source,
                            "shared": structures["bf16-full"].source,
                        },
                        targets=targets,
                        configs={
                            config: (
                                Config("add")
                                if config == "model"
                                else Config("copy", source="lane", source_config=config)
                            )
                            for config in structure.configs
                        },
                        order=tuple(
                            (component, key)
                            for component, rows in structure.components.items()
                            for key in rows
                        ),
                    )
                )
            )
            if transaction.receipt is not None:
                assert transaction.receipt is not None
                result[name] = ctx.adopt_model(transaction.receipt)
            else:
                active[name] = transaction
        if active:
            stats = lanes.write_cast(
                next(iter(active.values())),
                ctx,
                tel,
                mirrors=tuple(active.values())[1:],
                selection=selection,
                source="shared",
                source_component="video_vae",
                target_component="video_vae",
            )
            tel.log(
                "shared VAE repair",
                cast_keys=stats.converted_keys,
                reused_keys=stats.reused_keys,
                source_bytes=stats.source_bytes_read,
                unique_cast_bytes=stats.new_bytes_written,
                worst_relative_frobenius=stats.worst_relative_frobenius,
            )
        for name, transaction in active.items():
            if "model" not in transaction.completed_configs():
                transaction.add_config("model", configs[name])
            result[name] = ctx.adopt_model(transaction.commit())
    return result


def main(
    *,
    bf16_full: Model[object],
    bf16_pruned: Model[object],
    fp8_pruned: Model[object],
    mxfp8_pruned: Model[object],
    ctx: Context,
    tel: Telemetry,
) -> dict[str, ModelArtifact]:
    return repair(
        {
            "bf16-full": bf16_full,
            "bf16-pruned": bf16_pruned,
            "fp8-pruned": fp8_pruned,
            "mxfp8-pruned": mxfp8_pruned,
        },
        ctx,
        tel,
    )
