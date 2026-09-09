"""Bounded source-header validation for the two identical H3 DiT topologies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.resources import files
from typing import Any, Protocol

import msgspec
from cozy_runtime.author import (
    Loader,
    Model,
    WeightsSink,
    WeightsSource,
    WeightsTarget,
    canonical_json,
)

from .kernel import H3Topology, adapter_shapes, removed_keys, table_shapes
from .plans import TimestepPlan

TARGET_COMPONENT = {"fl2va": "fl2va_dit", "ref2va": "ref2va_dit"}

#: PDD-8's released adapter geometry (`lora_rank` / `lora_alpha` in the safetensors header).
#: The scale alpha/rank is a training fact the structure cannot carry, so the slice is
#: admitted at exactly this rank and no other.
ADAPTER_RANK = 64
ADAPTER_ALPHA = 64.0


class H3FullTransformer(Model[object]):
    """Native H3 source used only for derivation, never inference construction."""

    def load(self, loader: Loader) -> None:
        del loader


class H3TurboAdapter(Model[object]):
    """PDD acceleration LoRA, granted only for derivation: its `adaln_proj.linear` slice is
    tabled here; the six inference-time target families and the head bank are not read."""

    def load(self, loader: Loader) -> None:
        del loader


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_FULL_SPECS_DIGEST = "sha256:3af7354b5080f4c117922157971261fd079f12790f9584615e87b4fdd95ae3e2"
_FULL_CONFIG_DIGEST = "sha256:4150e2b9009aad13cf5a18b9878337346b2bfa7662b3f8037a0286cdaa806382"
_NATIVE_ONLY_SOURCE_SPECS = {"rope.inv_freq": ("f32", (16,))}
_TEXT_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)


class PartView(Protocol):
    @property
    def role(self) -> str: ...

    @property
    def dtype(self) -> str: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


class TensorView(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def logical_dtype(self) -> str: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def encoding(self) -> str: ...

    @property
    def parts(self) -> tuple[PartView, ...]: ...


class _FullSpec(msgspec.Struct, forbid_unknown_fields=True):
    dtype: str
    shape: tuple[int, ...]


class _FullSpecs(msgspec.Struct, forbid_unknown_fields=True):
    config_digest: str
    specs: tuple[tuple[str, _FullSpec], ...]


def official_full_specs(
    config: dict[str, Any],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Load the ordered 638-row serving destination contract without importing Diffusers.

    The existing resource is an ordered pair array, so canonical JSON cannot sort
    away the real factory's state_dict traversal. h3-conform checks every row,
    shape and dtype against the same OfficialH3Pipeline used by inference.
    """
    if canonical_json.digest(config) != _FULL_CONFIG_DIGEST:
        raise ValueError("H3 model config changed without a matching full-spec contract")
    raw = files(__package__).joinpath("assets", "dit-full-specs.json").read_bytes()
    if (
        canonical_json.normalize(raw) != raw
        or canonical_json.digest_bytes(raw) != _FULL_SPECS_DIGEST
    ):
        raise ValueError("H3 full-spec resource is not its exact canonical package bytes")
    document = msgspec.convert(canonical_json.decode(raw), type=_FullSpecs)
    if document.config_digest != _FULL_CONFIG_DIGEST or len(document.specs) != 638:
        raise ValueError("H3 full-spec resource changed its config binding or 638-row census")
    result = {name: (value.dtype, value.shape) for name, value in document.specs}
    if len(result) != len(document.specs):
        raise ValueError("H3 full-spec resource repeats a tensor destination")
    if any(
        dtype not in {"bf16", "f32"} or not shape or any(n <= 0 for n in shape)
        for dtype, shape in result.values()
    ):
        raise ValueError("H3 full-spec resource contains an invalid dtype or shape")
    return result


def validate_full_component(tensors: Sequence[TensorView], config: dict[str, Any]) -> str:
    """Require the exact production source topology and return its plain encoding id.

    The canonicalized official-native sources retain one 64-byte ``rope.inv_freq``
    buffer that Diffusers derives internally and therefore omits from its persistent
    ``state_dict``. It is the sole admitted source-only tensor and is removed from the
    derived target. Any other extra remains a closed-set refusal.
    """
    expected = official_full_specs(config)
    expected.update(_NATIVE_ONLY_SOURCE_SPECS)
    return _validate_component(tensors, expected, "FULL source")


def source_only_keys() -> tuple[str, ...]:
    """Exact native buffers with no Diffusers persistent destination."""
    return tuple(sorted(_NATIVE_ONLY_SOURCE_SPECS))


def text_source_only_keys() -> tuple[str, ...]:
    """Official Qwen rows excluded by the reviewed 50-layer pre-norm conditioner."""
    rows = [
        f"model.language_model.layers.{layer}.{suffix}"
        for layer in range(50, 64)
        for suffix in _TEXT_LAYER_SUFFIXES
    ]
    rows.extend(("lm_head.weight", "model.language_model.norm.weight"))
    if len(rows) != 156:
        raise AssertionError("the text-conditioner source-only census changed")
    return tuple(sorted(rows))


def _validate_component(
    tensors: Sequence[TensorView],
    expected: dict[str, tuple[str, tuple[int, ...]]],
    label: str,
) -> str:
    actual = {value.key: value for value in tensors}
    if len(actual) != len(tensors) or set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        unexpected = sorted(set(actual) - set(expected))[:5]
        raise ValueError(
            f"{label} H3 component has the wrong closed tensor set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    encodings: set[str] = set()
    for key, (dtype, shape) in expected.items():
        value = actual[key]
        parts = tuple((part.role, part.dtype, tuple(part.shape)) for part in value.parts)
        if value.logical_dtype != dtype or tuple(value.shape) != shape:
            raise ValueError(
                f"{label} H3 tensor {key} is {value.logical_dtype} {tuple(value.shape)}, "
                f"expected {dtype} {shape}"
            )
        if parts != (("value", dtype, shape),):
            raise ValueError(f"{label} H3 tensor {key} is not one plain value role: {parts}")
        if _DIGEST.fullmatch(value.encoding) is None:
            raise ValueError(f"{label} H3 tensor {key} has no exact encoding identity")
        encodings.add(value.encoding)
    if len(encodings) != 1:
        raise ValueError(f"{label} H3 component is not one plain encoding: {sorted(encodings)}")
    return encodings.pop()


def validate_adaln_pruned_component(
    tensors: Sequence[TensorView], config: dict[str, Any], plan: TimestepPlan
) -> str:
    """Require FULL minus 106 modulation tensors plus the exact 51 table destinations."""
    topology = H3Topology.from_config(config)
    expected = official_full_specs(config)
    for key in removed_keys(topology):
        expected.pop(key)
    expected.update({key: ("bf16", shape) for key, shape in table_shapes(topology, plan).items()})
    return _validate_component(tensors, expected, "AdaLN-pruned")


def adapter_slice(structure: WeightsSource, topology: H3Topology) -> tuple[str, tuple[str, ...]]:
    """The adapter's one component and every key of it the tables never read.

    Refuses unless one component carries the complete bf16 `adaln_proj.linear` LoRA slice
    at the admitted rank as plain single-role values.
    """
    components = {tensor.component for tensor in structure.tensors}
    if len(components) != 1:
        raise ValueError(f"an H3 adapter is one component, not {sorted(components)}")
    component = components.pop()
    present = {tensor.key: tensor for tensor in structure.tensors}
    wanted = adapter_shapes(topology, ADAPTER_RANK)
    for key, (_, shape) in wanted.items():
        tensor = present.get(key)
        if (
            tensor is None
            or tensor.logical_dtype != "bf16"
            or tuple(tensor.shape) != shape
            or tuple((part.name, part.dtype, tuple(part.shape)) for part in tensor.parts)
            != (("value", "bf16", shape),)
        ):
            raise ValueError(
                f"adapter lacks the bf16 rank-{ADAPTER_RANK} modulation slice {key} {shape}"
            )
    return component, tuple(sorted(set(present) - set(wanted)))


def full_targets() -> dict[str, WeightsTarget]:
    return {
        "fl2va_dit": WeightsTarget(
            source="dits",
            source_component="fl2va_dit",
            drop=source_only_keys(),
        ),
        "ref2va_dit": WeightsTarget(
            source="dits",
            source_component="ref2va_dit",
            drop=source_only_keys(),
        ),
        "text_encoder": WeightsTarget(
            source="shared",
            source_component="text_encoder",
            drop=text_source_only_keys(),
        ),
        "video_vae": WeightsTarget(source="shared", source_component="video_vae"),
        "audio_vae": WeightsTarget(source="shared", source_component="audio_vae"),
    }


def structures(
    artifacts: WeightsSink, sources: Mapping[str, H3FullTransformer]
) -> dict[str, WeightsSource]:
    """One bounded structure read per DISTINCT granted checkpoint, keyed by source alias.

    Both slots may bind the same complete checkpoint, and every consumer here — target
    selection and per-component treatment selection alike — wants the same facts.
    """
    read: dict[str, WeightsSource] = {}
    for source in sources.values():
        if source.checkpoint_ref not in read:
            read[source.checkpoint_ref] = artifacts.structure(source)
    return {alias: read[source.checkpoint_ref] for alias, source in sources.items()}


def select_full_targets(
    artifacts: WeightsSink,
    sources: Mapping[str, H3FullTransformer],
    granted: Mapping[str, WeightsSource] | None = None,
) -> dict[str, WeightsTarget]:
    """Drop source-only rows that remain in these exact granted checkpoints."""
    observed = structures(artifacts, sources) if granted is None else granted
    keys: dict[str, set[tuple[str, str]]] = {}
    for alias, source in sources.items():
        if source.checkpoint_ref not in keys:
            keys[source.checkpoint_ref] = {
                (tensor.component, tensor.key) for tensor in observed[alias].tensors
            }
    targets = full_targets()
    return {
        component: replace(
            target,
            drop=tuple(
                key
                for key in target.drop
                if (target.source_component, key) in keys[sources[target.source].checkpoint_ref]
            ),
        )
        for component, target in targets.items()
    }
