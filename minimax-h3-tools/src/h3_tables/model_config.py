"""One-way official-Hugging-Face -> canonical dual-task H3 config transformation."""

from __future__ import annotations

from typing import Any, cast

from cozy_runtime.author import canonical_json
from h3_table_layout import TableLayout

from ._table_layout import TableLayout
from .kernel import H3Topology
from .plans import Task, TimestepPlan

_SOURCE_SECTIONS = {
    "transformer",
    "transformer_ref",
    "text_encoder",
    "video_vae",
    "audio_vae",
}
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


def parse_full_config(raw: bytes) -> dict[str, dict[str, Any]]:
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("official H3 model config is not JSON") from exc
    if not isinstance(value, dict) or set(value) != _SOURCE_SECTIONS:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"official H3 model config sections are {actual}, expected {sorted(_SOURCE_SECTIONS)}"
        )
    sections: dict[str, dict[str, Any]] = {}
    for name, section in value.items():
        if not isinstance(section, dict):
            raise TypeError(
                f"official H3 model config section {name!r} is not a mapping"
            )
        sections[name] = cast(dict[str, Any], section)
    fl = {k: v for k, v in sections["transformer"].items() if k != "_diffusers_version"}
    ref = {
        k: v
        for k, v in sections["transformer_ref"].items()
        if k != "_diffusers_version"
    }
    if fl != ref:
        raise ValueError(
            "FL2VA and Ref2VA do not declare one identical DiT architecture config"
        )
    H3Topology.from_config(fl)
    validate_text_conditioner(sections["text_encoder"])
    return sections


def parse_production_config(raw: bytes) -> dict[str, dict[str, Any]]:
    """Recover official constructor facts from the producer's packaged dual-task config."""
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("production H3 model config is not JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {"fl2va_dit", "ref2va_dit", "text_encoder", "video_vae", "audio_vae"}
    ):
        raise ValueError("model config is not a dual-task H3 config")
    sections: dict[str, dict[str, Any]] = {}
    stamps: tuple[tuple[str, str, Task], ...] = (
        ("fl2va_dit", "transformer", "fl2va"),
        ("ref2va_dit", "transformer_ref", "ref2va"),
    )
    for target, source, expected_task in stamps:
        row = value[target]
        if not isinstance(row, dict):
            raise TypeError(f"model config {target!r} is not a mapping")
        config = dict(row)
        stamp = config.pop("cozy_h3", None)
        if (
            not isinstance(stamp, dict)
            or set(stamp) != {"task", "modulation", "table_keys"}
            or stamp["task"] != expected_task
            or stamp["modulation"] != "adaln-pruned"
            or not isinstance(stamp["table_keys"], dict)
        ):
            raise ValueError(f"model config {target!r} changed its task/table metadata")
        TableLayout.parse(stamp["table_keys"])
        sections[source] = config
    for component in ("text_encoder", "video_vae", "audio_vae"):
        row = value[component]
        if not isinstance(row, dict):
            raise TypeError(f"model config {component!r} is not a mapping")
        sections[component] = cast(dict[str, Any], row)
    return parse_full_config(canonical_json.encode(sections))


def dual_full_config(sections: dict[str, dict[str, Any]]) -> bytes:
    document: dict[str, dict[str, Any]] = {
        "audio_vae": sections["audio_vae"],
        "fl2va_dit": {
            **sections["transformer"],
            "cozy_h3": {"task": "fl2va", "modulation": "full"},
        },
        "ref2va_dit": {
            **sections["transformer_ref"],
            "cozy_h3": {"task": "ref2va", "modulation": "full"},
        },
        "text_encoder": sections["text_encoder"],
        "video_vae": sections["video_vae"],
    }
    return canonical_json.encode(document)


def task_config(
    sections: dict[str, dict[str, Any]], plan: TimestepPlan
) -> dict[str, Any]:
    source_name = "transformer" if plan.task == "fl2va" else "transformer_ref"
    component = "fl2va_dit" if plan.task == "fl2va" else "ref2va_dit"
    config = dict(sections[source_name])
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
    document = {
        "audio_vae": sections["audio_vae"],
        "fl2va_dit": task_config(sections, fl2va)["fl2va_dit"],
        "ref2va_dit": task_config(sections, ref2va)["ref2va_dit"],
        "text_encoder": sections["text_encoder"],
        "video_vae": sections["video_vae"],
    }
    return canonical_json.encode(document)
