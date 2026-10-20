"""The reviewed MiniMax-H3 policy supplied to generic quantization machinery."""

from importlib.resources import files

import msgspec
from cozy_runtime.derive.quantization import ArtifactQuantizationPlan, QuantizationTensor


def h3_quantization_plan() -> bytes:
    """Return the exact pruned-DiT plan owned by this model package."""
    return files(__package__).joinpath("assets/h3-dit-quantization-plan.json").read_bytes()


class _Weight(msgspec.Struct, forbid_unknown_fields=True):
    component: str
    key: str
    logical_dtype: str
    shape: tuple[int, int]
    source_role: str


class _Reviewed(msgspec.Struct, forbid_unknown_fields=True):
    components: list[str]
    configs: list[str]
    order: list[tuple[str, str]]
    tensors: list[_Weight]


def reviewed_plan() -> ArtifactQuantizationPlan:
    """Read the retained 313-weight model policy into the current public encoder plan."""
    value = msgspec.json.decode(h3_quantization_plan(), type=_Reviewed)
    if value.components != ["dit"] or value.configs != ["model"] or len(value.tensors) != 313:
        raise ValueError("H3 reviewed quantization policy changed its component or weight census")
    tensors = []
    for row in value.tensors:
        if (row.component != "dit" or row.source_role != "value"
                or row.logical_dtype != "bf16" or min(row.shape) <= 0 or row.shape[1] % 32
                or not row.key.endswith(".weight")):
            raise ValueError(f"invalid reviewed H3 weight: {row.key}")
        tensors.append(QuantizationTensor(row.component, row.key, row.logical_dtype, row.shape))
    if len({row.key for row in tensors}) != len(tensors):
        raise ValueError("H3 reviewed quantization policy repeats a weight")
    return ArtifactQuantizationPlan(value.components, value.order, tensors)
