"""H3's construction config: one section per component (D4 names), stored as
`<component>/config.json`; Runtime reassembles them into one `{component: config}` mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cozy_runtime.author import canonical_json

from .kernel import H3Topology
from .plans import Task, TimestepPlan
from .table_layout import TableLayout

COMPONENTS = ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae")
DIT_COMPONENT: dict[Task, str] = {"fl2va": "transformer", "ref2va": "transformer_ref"}
_TEXT_CONDITIONER_CONFIG = {
    "source_architecture": "Qwen3VLForConditionalGeneration",
    "retained_decoder_layers": 50,
    "conditioning_hidden_state": 50,
    "output": "pre_norm",
    "language_model_head": False,
}


def validate_text_conditioner(config: dict[str, Any]) -> None:
    if config.get("cozy_h3") != _TEXT_CONDITIONER_CONFIG:
        raise ValueError(
            "text_encoder config does not declare the exact 50-layer pre-norm conditioner"
        )


def _sections(raw: bytes, label: str) -> dict[str, dict[str, Any]]:
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or set(value) != set(COMPONENTS):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} sections are {actual}, expected {sorted(COMPONENTS)}")
    if not all(isinstance(section, dict) for section in value.values()):
        raise TypeError(f"{label} has a section that is not a mapping")
    return cast(dict[str, dict[str, Any]], value)


def parse_full_config(raw: bytes) -> dict[str, dict[str, Any]]:
    """The official constructor sections: both DiTs one architecture, the 50-layer conditioner."""
    sections = _sections(raw, "official H3 model config")
    fl, ref = (
        {k: v for k, v in sections[c].items() if k != "_diffusers_version"}
        for c in DIT_COMPONENT.values()
    )
    if fl != ref:
        raise ValueError("FL2VA and Ref2VA do not declare one identical DiT architecture config")
    H3Topology.from_config(fl)
    validate_text_conditioner(sections["text_encoder"])
    return sections


def parse_production_config(raw: bytes) -> dict[str, dict[str, Any]]:
    """The official sections of an AdaLN-pruned config, its task/table stamps removed."""
    sections = _sections(raw, "production H3 model config")
    for task, component in DIT_COMPONENT.items():
        config = dict(sections[component])
        stamp = config.pop("cozy_h3", None)
        if (
            not isinstance(stamp, dict)
            or set(stamp) != {"task", "modulation", "table_keys"}
            or stamp["task"] != task
            or stamp["modulation"] != "adaln-pruned"
            or not isinstance(stamp["table_keys"], dict)
        ):
            raise ValueError(f"model config {component!r} changed its task/table metadata")
        TableLayout.parse(stamp["table_keys"])
        sections[component] = config
    return parse_full_config(canonical_json.encode(sections))


def dual_full_config(sections: dict[str, dict[str, Any]]) -> bytes:
    document = dict(sections)
    for task, component in DIT_COMPONENT.items():
        document[component] = {**sections[component], "cozy_h3": {"task": task, "modulation": "full"}}
    return canonical_json.encode(document)


def task_config(sections: dict[str, dict[str, Any]], plan: TimestepPlan) -> dict[str, Any]:
    component = DIT_COMPONENT[plan.task]
    config = dict(sections[component])
    config["cozy_h3"] = {
        "task": plan.task,
        "modulation": "adaln-pruned",
        "table_keys": plan.table_keys,
    }
    return {component: config}


def dual_adaln_pruned_config(
    sections: dict[str, dict[str, Any]], fl2va: TimestepPlan, ref2va: TimestepPlan
) -> bytes:
    if fl2va.task != "fl2va" or ref2va.task != "ref2va":
        raise ValueError("dual AdaLN-pruned config requires each task's own table layout")
    return canonical_json.encode(
        {**sections, **task_config(sections, fl2va), **task_config(sections, ref2va)}
    )


def config_files(raw: bytes) -> dict[str, bytes]:
    """A `{component: config}` document as the checkpoint's `<component>/config.json` files."""
    value = canonical_json.decode(raw)
    if not isinstance(value, dict):
        raise ValueError("an H3 construction config is a mapping of component configs")
    return {f"{name}/config.json": canonical_json.encode(section) for name, section in value.items()}


def read_config(files: Mapping[str, bytes]) -> bytes:
    """The `{component: config}` document of a checkpoint's `<component>/config.json` files."""
    return canonical_json.encode(
        {
            path.removesuffix("/config.json"): canonical_json.decode(raw)
            for path, raw in files.items()
            if path.endswith("/config.json") and path.count("/") == 1
        }
    )
