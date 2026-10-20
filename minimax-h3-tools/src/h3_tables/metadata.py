"""H3 planning facts read from public TensorFS checkpoint metadata.

These records describe this producer's selections. Storage, resumability and byte admission
remain the public Runtime Output and TensorFS Writer contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

from cozy_runtime.author import Context, ModelArtifact, canonical_json
from cozy_runtime.derive import Checkpoint, Derivation, Output
from tensorfs import raw

DTYPES = {
    "BF16": "bf16", "F16": "f16", "F32": "f32", "F64": "f64", "I8": "i8",
    "U8": "u8", "I16": "i16", "I32": "i32", "I64": "i64", "U16": "u16",
    "F8_E4M3": "f8_e4m3fn", "F4": "f4",
}
NATIVE_DTYPES = {value: key for key, value in DTYPES.items()}
PLAIN = "plain"
SCHEMES: dict[Literal["fp8-rowwise/1", "int8-convrot256/1", "nvfp4-w4a4/1"], str] = {
           "fp8-rowwise/1": "fp8.rowwise", "int8-convrot256/1": "int8.convrot",
           "nvfp4-w4a4/1": "nvfp4"}


@dataclass(frozen=True, slots=True)
class PartShape:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TensorShape:
    logical_dtype: str
    shape: tuple[int, ...]
    encoding: str
    parts: Mapping[str, PartShape]
    copy_from: tuple[str, str, str] | None = None


@dataclass(frozen=True, slots=True)
class SourceLayout:
    checkpoint: Checkpoint
    components: Mapping[str, Mapping[str, TensorShape]]
    configs: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class ComponentSelection:
    source: str = ""
    source_component: str = ""
    drop: tuple[str, ...] = ()
    add: Mapping[str, TensorShape] = field(default_factory=dict)


def inspect_checkpoint(checkpoint: Checkpoint) -> SourceLayout:
    components: dict[str, dict[str, TensorShape]] = {}
    configs: dict[str, bytes] = {}
    for name, component in checkpoint.components.items():
        if not component.tensors:
            configs[name] = canonical_json.encode(component.config)
            continue
        components[name] = {
            key: TensorShape(
                DTYPES.get(tensor.dtype, tensor.dtype), tuple(tensor.shape),
                tensor.encoding.scheme if tensor.encoding is not None else PLAIN,
                {role or "value": PartShape(DTYPES.get(part.dtype, part.dtype), tuple(part.shape))
                 for role, part in raw.parts(checkpoint, name, key).items()},
            )
            for key, tensor in component.tensors.items()
        }
    return SourceLayout(checkpoint, components, configs)


def configure(output: Output, name: str, data: bytes) -> None:
    value = canonical_json.decode(data)
    if not isinstance(value, dict):
        raise ValueError(f"{name} constructor config must be an object")
    output.component(name, cls="", config=value)


@contextmanager
def model_output(
    ctx: Context, slot: str, *, sources: Mapping[str, Checkpoint],
    components: Mapping[str, ComponentSelection], configs: Mapping[str, bytes],
    order: Sequence[tuple[str, str]],
) -> Iterator[Output]:
    """Copy H3's selected source tensors by reference into one bounded modern output.

    The computation writes only the selections' additions with Output.put/put_encoded.
    adopt_output checks that the ordered tensors are complete before adoption.
    """
    declared = set(order)
    if len(declared) != len(order):
        raise ValueError("H3 construction order repeats a tensor")
    selected = set()
    for name, selection in components.items():
        original = (sources[selection.source].components[selection.source_component]
                    if selection.source else None)
        carried = set(original.tensors) - set(selection.drop) if original is not None else set()
        if carried & selection.add.keys():
            raise ValueError(f"{name} adds a tensor it still carries from its source")
        selected.update((name, key) for key in carried | selection.add.keys())
    if selected != declared:
        raise ValueError(f"H3 source/selection differs from construction: "
                         f"missing={sorted(declared - selected)[:5]}, "
                         f"unexpected={sorted(selected - declared)[:5]}")
    with ctx.output(slot, Derivation(None, sources=tuple(sources.values()), order=order)) as output:
        if output.done is None:
            for name, selection in components.items():
                source = sources[selection.source] if selection.source else None
                original = source.components[selection.source_component] if source is not None else None
                writer = output.component(
                    name, cls=original.cls if original else "",
                    config=original.config if original else {},
                )
                if source is not None and original is not None:
                    missing = set(selection.drop) - set(original.tensors)
                    if missing:
                        raise ValueError(f"{name} drops absent source rows: {sorted(missing)}")
                    # Existing source assets retain their exact content, as Output.inherit does.
                    for asset in original.assets:
                        writer.asset(asset, original.asset(asset))
                    for key in original.tensors:
                        if key not in selection.drop:
                            output.copy(name, key, source, selection.source_component)
                for key, tensor in selection.add.items():
                    if tensor.copy_from is not None:
                        alias, component, original_key = tensor.copy_from
                        output.copy(name, key, sources[alias], component, original_key)
            for name, data in configs.items():
                configure(output, name, data)
        yield output


def adopt_output(ctx: Context, output: Output) -> ModelArtifact:
    """An H3 output must contain its entire declared construction before it is retained."""
    missing = [(component, key) for component, key in output.order
               if not output.has(component, key)]
    if missing:
        raise ValueError(f"H3 output {output.slot} is unfinished: {missing[:5]}")
    return ctx.adopt_model(output)
