"""One-way official-Hugging-Face -> canonical dual-task H3 config transformation."""

from __future__ import annotations

from typing import Any, cast

from cozy_runtime.author import canonical_json

from .kernel import H3Topology
from .plans import LAUNCH_PLAN_DIGESTS, Task, TimestepPlan

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
_CURRENT_MODEL_CONFIG = (
    "sha256:f587d48a97661ce51dbdddb54b334981d9400c0b2467717a3445bb4108c50571"
)
_CURRENT_MODEL_CONFIG_LENGTH = 5817


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
    """Recover the official source sections from the one exact production config.

    The package asset is the final dual-task config, so it also pins both task/plan stamps.
    Removing those package-owned stamps recovers the official Diffusers constructor facts;
    no second model-config document is needed.
    """
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError("production H3 model config is not JSON") from exc
    if (
        not isinstance(value, dict)
        or canonical_json.digest(value) != _CURRENT_MODEL_CONFIG
        or len(raw) != _CURRENT_MODEL_CONFIG_LENGTH
        or set(value)
        != {"fl2va_dit", "ref2va_dit", "text_encoder", "video_vae", "audio_vae"}
    ):
        raise ValueError("model config is not the exact current dual-task H3 config")
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
        if stamp != {
            "task": expected_task,
            "modulation": "adaln-pruned",
            "timestep_plan_digest": LAUNCH_PLAN_DIGESTS[expected_task],
        }:
            raise ValueError(f"model config {target!r} changed its task/plan stamp")
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
        "timestep_plan_digest": plan.digest,
    }
    return {component: config}


def dual_adaln_pruned_config(
    sections: dict[str, dict[str, Any]], fl2va: TimestepPlan, ref2va: TimestepPlan
) -> bytes:
    document = {
        "audio_vae": sections["audio_vae"],
        "fl2va_dit": task_config(sections, fl2va)["fl2va_dit"],
        "ref2va_dit": task_config(sections, ref2va)["ref2va_dit"],
        "text_encoder": sections["text_encoder"],
        "video_vae": sections["video_vae"],
    }
    raw = canonical_json.encode(document)
    if (
        len(raw) != _CURRENT_MODEL_CONFIG_LENGTH
        or canonical_json.digest(document) != _CURRENT_MODEL_CONFIG
    ):
        raise ValueError(
            "dual AdaLN-pruned config is not the exact current H3 model config"
        )
    return raw
