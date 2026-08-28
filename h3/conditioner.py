"""Exact 50-layer Qwen3-VL conditioner used by MiniMax-H3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cozy_runtime.author import ConformanceError

_SCHEMA = "cozy.minimax_h3.text_conditioner/1"
_SOURCE_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
_SOURCE_LAYERS = 64
_RETAINED_LAYERS = 50
_PERSISTENT_TENSORS = 902


def text_conditioner_config() -> dict[str, object]:
    """The closed artifact extension, shared by fixtures and production publishers."""
    return {
        "schema": _SCHEMA,
        "source_architecture": _SOURCE_ARCHITECTURE,
        "retained_decoder_layers": _RETAINED_LAYERS,
        "conditioning_hidden_state": _RETAINED_LAYERS,
        "output": "pre_norm",
        "language_model_head": False,
    }


def build_text_conditioner(config: Mapping[str, object]) -> Any:
    """Build the upstream surface, then remove computation after hidden state 50."""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    source = dict(config)
    extension = source.pop("cozy_h3", None)
    if extension != text_conditioner_config():
        raise ConformanceError(
            "artifact config 'text_encoder.cozy_h3' does not declare the exact H3 "
            "50-layer pre-norm conditioner",
            code="artifact_config",
            fields=["text_encoder", "cozy_h3"],
        )
    if source.get("architectures") != [_SOURCE_ARCHITECTURE]:
        raise ConformanceError(
            "text conditioner source architecture is not Qwen3VLForConditionalGeneration",
            code="artifact_config",
            fields=["text_encoder", "architectures"],
        )
    text = source.get("text_config")
    if not isinstance(text, Mapping) or text.get("num_hidden_layers") != _SOURCE_LAYERS:
        raise ConformanceError(
            "text conditioner source must declare the exact 64-layer Qwen architecture",
            code="artifact_config",
            fields=["text_encoder", "text_config", "num_hidden_layers"],
        )

    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**source))
    language = cast(Any, model.model.language_model)
    if len(language.layers) != _SOURCE_LAYERS:
        raise ConformanceError(
            "constructed Qwen language stack disagrees with its source config",
            code="artifact_config",
            fields=["text_encoder", "text_config", "num_hidden_layers"],
        )
    language.layers = torch.nn.ModuleList(list(language.layers[:_RETAINED_LAYERS]))
    language.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    model.to(dtype=torch.bfloat16).eval()
    _validate_census(model, torch)
    return model


def _validate_census(model: Any, torch: Any) -> None:
    state = model.state_dict()
    visual = [key for key in state if key.startswith("model.visual.")]
    language = [key for key in state if key.startswith("model.language_model.")]
    removed = [
        key
        for key in state
        if key == "lm_head.weight"
        or key == "model.language_model.norm.weight"
        or _removed_layer(key)
    ]
    if (
        len(state) != _PERSISTENT_TENSORS
        or len(visual) != 351
        or len(language) != 551
        or removed
        or any(value.dtype != torch.bfloat16 for value in state.values())
    ):
        raise ConformanceError(
            "H3 text conditioner is not exactly 351 vision plus 551 retained BF16 "
            "language destinations",
            code="artifact_config",
            fields=removed[:6] or ["text_encoder"],
        )


def _removed_layer(key: str) -> bool:
    prefix = "model.language_model.layers."
    if not key.startswith(prefix):
        return False
    try:
        return int(key[len(prefix) :].split(".", 1)[0]) >= _RETAINED_LAYERS
    except ValueError:
        return True
