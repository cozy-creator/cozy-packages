"""The H3 lane catalogue: what each lane does to each of the five components.

A lane is a NAME plus, per component, what this producer does to it. A component the
lane does not name is INHERITED BY REFERENCE — TensorFS copies its tensor metadata and
ObjectRefs unchanged through the zero-read/zero-hash inherit gate — so every lane that
leaves the conditioner and the two VAEs alone points at exactly the same stored objects.
That is why the treatment map is sparse and must stay sparse: naming a component costs
its bytes once per lane that names it, and the four published lanes share 58.2 GiB of
untouched conditioner/VAE today precisely because none of them names those three.

The catalogue is CODE, not a request field. Output slot names are decorator-time facts,
so they cannot come off the wire; and every lane here mints a MiniMax H3 Model Derivative
under §I.11(i) of the community licence, which is a reviewed act rather than a caller
choice. The request selects a subset of these rows; it can never author one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from cozy_runtime.author import (
    Context,
    Telemetry,
    UnsupportedInput,
    WeightsPart,
    WeightsSourceTensor,
    WeightsTarget,
    WeightsTensor,
    WeightsTransaction,
)
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.quantization import (
    BF16_ROUND_TRIP_BOUND,
    MAX_OUTPUT_BYTES,
    PLAIN_SPEC,
    ArtifactQuantizationPlan,
    prepare_source_quantization,
    quantization_additions,
)

from .source import TARGET_COMPONENT

CastDtype = Literal["bf16", "f16"]
Encoding = Literal["fp8-rowwise/1", "mxfp8/1"]
Modulation = Literal["full", "adaln-pruned"]

#: The five components of an H3 checkpoint. Every lane declares all five: the ones it does
#: not treat are declared as plain inheriting targets, which is what carries their objects.
COMPONENTS = ("fl2va_dit", "ref2va_dit", "text_encoder", "video_vae", "audio_vae")

#: The component name the reviewed DiT quantization plan addresses its 313 keys under.
DIT_PLAN_COMPONENT = "dit"

#: The two DiTs quantize through the package-owned reviewed plan
#: (``cozy_runtime.derive.h3-dit-quantization-plan.json``), never the generic structural
#: selector: their AdaLN table rows are rank-2 bf16 tensors whose keys end in `.weight`
#: too, and a shape rule would happily encode them.
DIT_COMPONENTS = frozenset(TARGET_COMPONENT.values())

#: The one encoding with a rung on the cards we serve. `RowwiseNativeLeaf.device_predicate`
#: is `cuda.sm89+`; `MicroScaledNativeLeaf.device_predicate` is the EQUALITY `cuda.sm120`,
#: which `DeviceFacts.satisfies` tests exactly — so `mxfp8/1` executes on no H100, H200 or
#: B200 and degrades to `cozy.mxfp8.dequant/1` plus a bf16 GEMM. A lane that cannot be
#: served on the card we serve is not a lane, so no new one may use it.
SERVABLE_ENCODINGS = frozenset({"fp8-rowwise/1"})

#: The exact, named exception: `mxfp8-adaln-pruned` predates the rule above and is still
#: published. It is grandfathered BY NAME, so adding any other mxfp8 lane refuses.
UNSERVABLE_LANE = "mxfp8-adaln-pruned"

#: Components no lane may treat, and why. This is a refusal BY NAME, not by shape, because
#: the shape rule does not catch it: the audio VAE's 1,087 rows are 637 rank-3
#: `weight_g`/`weight_v` weight-norm pairs (unrepresentable — `fp8-rowwise/1` pins
#: `logical_rank: Some(2)` and `weight_g` never matches the `.weight` suffix) plus exactly
#: SIX rank-2 float `.weight` rows, all of them in the `pre_block` ENCODER attention/MLP
#: and all block-aligned. So the structural selector does NOT refuse this component — it
#: silently selects those six (12.6 M of 151 M parameters, dominated by
#: `pre_block.attn.qkv.weight [6144, 2048]`), quantizes the audio CONDITIONING path, leaves
#: the BigVGAN decoder untouched and saves nothing. Every ecosystem publisher pins audio
#: autoencoders to fp32 and h3a-006 already records the same conclusion.
REFUSED_COMPONENTS: Mapping[str, str] = {
    "audio_vae": (
        "the audio VAE is not representable: 637 of its 1,087 rows are rank-3 weight_norm "
        "weight_g/weight_v pairs, and the only six rank-2 float weights it has are the "
        "pre_block ENCODER attention/MLP linears, so treating it would encode the audio "
        "conditioning path and leave the BigVGAN decoder alone"
    ),
}

#: Author-declared ceilings on the new bytes ONE treatment of a component may write. Not a
#: second copy of the census: the transaction's own byte accounting is exact and TensorFS
#: locks the output bound before it pulls a byte. This sizes the declared output slot,
#: which is a decorator-time fact and cannot be measured from a source nobody has granted
#: yet. Each is comfortably above the component's whole plain size.
COMPONENT_MAX_NEW_BYTES: Mapping[str, int] = {
    "fl2va_dit": MAX_OUTPUT_BYTES,
    "ref2va_dit": MAX_OUTPUT_BYTES,
    "text_encoder": 56 << 30,
    "video_vae": 12 << 30,
}


@dataclass(frozen=True, slots=True)
class Treatment:
    """What one lane does to one component.

    ``cast`` rewrites every plain float32 tensor of the component at the named dtype;
    an already-16-bit tensor inherits untouched, exactly as the ingest normalization does
    (#695 — fp16 carries the same bytes as bf16 and re-rounding it loses three mantissa
    bits for nothing). ``encode`` replaces the component's block-aligned rank-2 float
    weights with the named encoding. ``keep`` names exact keys the encoding must NOT
    select; they still follow ``cast`` like any other tensor.

    Both fields are optional and at least one must be present — a treatment that changes
    nothing is spelled by leaving the component out of the lane, which is what preserves
    the shared ObjectRefs.
    """

    cast: CastDtype | None = None
    encode: Encoding | None = None
    keep: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cast is None and self.encode is None:
            raise ValueError("a treatment casts, encodes, or both; absence means inherit")
        if self.keep and self.encode is None:
            raise ValueError("a keep list only excludes keys from an encoding")
        if len(set(self.keep)) != len(self.keep):
            raise ValueError("a treatment keep list repeats a key")

    def describe(self) -> str:
        spelled = []
        if self.cast is not None:
            spelled.append(f"cast={self.cast}")
        if self.encode is not None:
            spelled.append(f"encode={self.encode}")
            if self.keep:
                spelled.append(f"keep={len(self.keep)}")
        return " ".join(spelled)


@dataclass(frozen=True, slots=True)
class Lane:
    """One published checkpoint: a modulation plus the components this lane rewrites."""

    modulation: Modulation
    components: Mapping[str, Treatment] = field(default_factory=dict)


_DIT_FP8 = Treatment(encode="fp8-rowwise/1")
_DIT_MXFP8 = Treatment(encode="mxfp8/1")


def _both_dits(treatment: Treatment) -> dict[str, Treatment]:
    return {component: treatment for component in sorted(DIT_COMPONENTS)}


#: The conditioner rows h3a-028 holds at bf16, measured against the real shard headers.
#: The structural selector takes 441 rows of the reviewed 902-row conditioner; exactly 350
#: of them are the wanted `layers.{0..49}` projections. The other 91 are the token
#: embedding table, the whole 27-block visual tower, its two mergers and the three
#: deepstack mergers — all rank-2, all block-aligned, and all things a shape rule takes and
#: a conditioner must not lose. `blocks.N.mlp.linear_fc2 [1152, 4304]` is deliberately
#: ABSENT: 4304 % 32 == 16, so the selector never picks it and naming it would trip the
#: unmatched-keep refusal.
TEXT_ENCODER_KEEP: tuple[str, ...] = (
    "model.language_model.embed_tokens.weight",
    "model.visual.pos_embed.weight",
    "model.visual.merger.linear_fc1.weight",
    "model.visual.merger.linear_fc2.weight",
    *(
        f"model.visual.deepstack_merger_list.{index}.linear_fc{leaf}.weight"
        for index in range(3)
        for leaf in (1, 2)
    ),
    *(
        f"model.visual.blocks.{index}.{leaf}.weight"
        for index in range(27)
        for leaf in ("attn.qkv", "attn.proj", "mlp.linear_fc1")
    ),
)

#: The reviewed catalogue. Adding a lane is ONE row here: `job.py` derives its
#: `WeightsOutput`, its byte ceiling, its targets and its writers from this mapping, so
#: there is no second list of names to keep in step.
#:
#: The two rows the acceleration lanes are converging on are deliberately absent until
#: their owners land them, because each is one line and each names a format this producer
#: must not pick for them. When they land they are, respectively,
#:
#:     "fp8-vae-adaln-pruned": Lane("adaln-pruned", {**_both_dits(_DIT_FP8),
#:         "video_vae": Treatment(cast="f16", encode="fp8-rowwise/1")}),
#:     "fp8-te-adaln-pruned": Lane("adaln-pruned", {**_both_dits(_DIT_FP8),
#:         "text_encoder": Treatment(encode="fp8-rowwise/1", keep=TEXT_ENCODER_KEEP)}),
#:
#: The video VAE's fp32→fp16-versus-bf16 question is still open with the VAE conversion
#: lane (h3a-027 §7 recommends the cast FIRST and the 217 decoder linears second, and the
#: serving package must construct the VAE at the stored dtype before either can be served
#: — `_keep_in_fp32_modules` pins it to fp32 today, so an f16 lane refuses at fit). The
#: conditioner's is settled: h3a-028 ruled fp8-rowwise/1 with `TEXT_ENCODER_KEEP`, never
#: int8 (proto-001 candidate H failed on flicker) and never mxfp8.
LANES: Mapping[str, Lane] = {
    "bf16-full": Lane("full"),
    "bf16-adaln-pruned": Lane("adaln-pruned"),
    "fp8-adaln-pruned": Lane("adaln-pruned", _both_dits(_DIT_FP8)),
    "mxfp8-adaln-pruned": Lane("adaln-pruned", _both_dits(_DIT_MXFP8)),
}


def validate_catalogue(lanes: Mapping[str, Lane] = LANES) -> None:
    """Refuse an unservable, unrepresentable or unbudgeted lane before anything imports."""
    if not lanes:
        raise ValueError("the H3 lane catalogue is empty")
    for name, lane in lanes.items():
        if lane.modulation == "full" and lane.components:
            raise ValueError(f"lane {name!r} is a FULL lane and must inherit every component")
        for component, treatment in lane.components.items():
            if component in REFUSED_COMPONENTS:
                raise ValueError(
                    f"lane {name!r} treats {component}: {REFUSED_COMPONENTS[component]}"
                )
            if component not in COMPONENT_MAX_NEW_BYTES:
                raise ValueError(f"lane {name!r} treats unknown component {component!r}")
            if (
                treatment.encode is not None
                and treatment.encode not in SERVABLE_ENCODINGS
                and name != UNSERVABLE_LANE
            ):
                raise ValueError(
                    f"lane {name!r} encodes {component} as {treatment.encode}, which has no "
                    f"rung on the cards we serve; servable encodings are "
                    f"{sorted(SERVABLE_ENCODINGS)}"
                )
            if component in DIT_COMPONENTS and treatment.cast is not None:
                raise ValueError(
                    f"lane {name!r} casts {component}: the DiTs are bf16 at source and their "
                    "AdaLN rows are computed, never converted"
                )


validate_catalogue()


def lane_max_new_bytes(lane: Lane, *, full_bytes: int, pruned_bytes: int) -> int:
    """The declared new-byte ceiling of one lane's output slot."""
    base = full_bytes if lane.modulation == "full" else pruned_bytes
    return base + sum(COMPONENT_MAX_NEW_BYTES[component] for component in lane.components)


@dataclass(frozen=True, slots=True)
class _StructureView:
    """The Runtime `SourceStructure` protocol over one component's granted tensors."""

    tensors: tuple[WeightsSourceTensor, ...]
    configs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Selection:
    """One component's resolved work: which keys cast, which encode, under what plan."""

    component: str
    treatment: Treatment
    cast: tuple[tuple[str, tuple[int, ...]], ...]
    plan: ArtifactQuantizationPlan | None
    plan_component: str

    @property
    def encoded(self) -> tuple[str, ...]:
        return () if self.plan is None else tuple(tensor.key for tensor in self.plan.tensors)

    @property
    def replaced(self) -> tuple[str, ...]:
        return tuple(sorted({key for key, _ in self.cast} | set(self.encoded)))


def carried(
    target: WeightsTarget, tensors: Sequence[WeightsSourceTensor]
) -> tuple[WeightsSourceTensor, ...]:
    """The source rows one target actually carries: its component MINUS its dropped rows.

    A treatment must never see a row the lane removes. The conditioner is where this bites:
    the reviewed 50-layer pre-norm target drops the official model's layers 50-63, its final
    norm and its language-model head — 156 rows, of which `lm_head.weight [151936, 5120]` is
    rank-2, block-aligned and therefore exactly what a structural selector would take. Left
    unscoped, an encoding would resurrect the row the lane exists to delete.
    """
    dropped = set(target.drop)
    return tuple(
        tensor
        for tensor in tensors
        if tensor.component == target.source_component and tensor.key not in dropped
    )


def _plain_value(tensor: WeightsSourceTensor) -> bool:
    return (
        len(tensor.parts) == 1
        and tensor.parts[0].name == "value"
        and tensor.parts[0].dtype == tensor.logical_dtype
        and tuple(tensor.parts[0].shape) == tuple(tensor.shape)
    )


def select(
    component: str,
    treatment: Treatment,
    tensors: Sequence[WeightsSourceTensor],
    *,
    dit_plan: ArtifactQuantizationPlan | None = None,
) -> Selection:
    """Resolve one treatment against the granted source structure.

    The two DiTs encode through the reviewed plan, never through shape. Every other
    component selects structurally with the Runtime's own ``prepare_source_quantization``
    minus the treatment's ``keep`` list — and a ``keep`` entry that excludes nothing
    REFUSES, because the failure this list exists to stop is a typo that silently
    quantizes an embedding table.
    """
    if component in REFUSED_COMPONENTS:
        raise UnsupportedInput(
            f"{component} is refused: {REFUSED_COMPONENTS[component]}",
            code="h3_component_refused",
        )
    present = tuple(tensor for tensor in tensors if tensor.component == component)
    if not present:
        raise UnsupportedInput(
            f"source has no component {component!r}", code="h3_component_absent"
        )

    plan: ArtifactQuantizationPlan | None = None
    plan_component = component
    if treatment.encode is not None:
        if component in DIT_COMPONENTS:
            if dit_plan is None:
                raise UnsupportedInput(
                    f"{component} encodes through the reviewed DiT plan, which was not supplied",
                    code="h3_plan_absent",
                )
            plan, plan_component = dit_plan, DIT_PLAN_COMPONENT
        else:
            structural = prepare_source_quantization(
                _StructureView(present), components=(component,)
            )
            kept = set(treatment.keep)
            missing = sorted(kept - {tensor.key for tensor in structural.tensors})
            if missing:
                raise UnsupportedInput(
                    f"{component} keep list names {missing}, which the structural selection "
                    "never picked; a keep entry that excludes nothing is a typo",
                    code="h3_keep_unmatched",
                )
            selected = [tensor for tensor in structural.tensors if tensor.key not in kept]
            if not selected:
                raise UnsupportedInput(
                    f"{component} keep list excludes every selected weight",
                    code="h3_keep_exhaustive",
                )
            plan = ArtifactQuantizationPlan(
                components=[component],
                configs=[],
                order=[(component, tensor.key) for tensor in present],
                tensors=selected,
            )

    encoded = {tensor.key for tensor in plan.tensors} if plan is not None else set()
    cast: list[tuple[str, tuple[int, ...]]] = []
    if treatment.cast is not None:
        for tensor in present:
            # Only float32 converts. A 16-bit source already carries the target's byte
            # width and re-rounding it loses mantissa bits for nothing (#695), so it
            # inherits by reference — which is what stops a cast lane from rewriting a
            # component it barely changes.
            if tensor.key in encoded or tensor.logical_dtype != "f32":
                continue
            if not _plain_value(tensor):
                raise UnsupportedInput(
                    f"{component}.{tensor.key} is not one plain value role and cannot be cast",
                    code="h3_cast_encoded_source",
                )
            cast.append((tensor.key, tuple(tensor.shape)))
    if not cast and plan is None:
        raise UnsupportedInput(
            f"{component} treatment {treatment.describe()!r} changes nothing on this source",
            code="h3_treatment_inert",
        )
    return Selection(component, treatment, tuple(cast), plan, plan_component)


def additions(selection: Selection) -> dict[str, WeightsTensor]:
    """The declared replacement tensors for one resolved treatment."""
    dtype = selection.treatment.cast
    result: dict[str, WeightsTensor] = {}
    for key, shape in selection.cast:
        assert dtype is not None
        result[key] = WeightsTensor(
            logical_dtype=dtype,
            shape=shape,
            encoding=PLAIN_SPEC,
            parts={"value": WeightsPart(dtype, shape)},
        )
    if selection.plan is not None and selection.treatment.encode is not None:
        result.update(
            quantization_additions(
                selection.treatment.encode,
                selection.plan,
                selection.plan_component,
                target_logical_dtype=dtype,
            )
        )
    return result


def apply(target: WeightsTarget, selection: Selection) -> WeightsTarget:
    """Fold one resolved treatment into a component's target declaration.

    Only the treated keys are dropped. A target's other additions — the AdaLN timestep
    tables — are pure creations with no source row to drop, and TensorFS refuses a drop of
    a key its selected source component does not carry.
    """
    return WeightsTarget(
        source=target.source,
        source_component=target.source_component,
        drop=tuple(sorted(set(target.drop) | set(selection.replaced))),
        add={**target.add, **additions(selection)},
    )


@dataclass(frozen=True, slots=True)
class CastStats:
    """What one cast recorded — the same tier-1 round-trip shape the encoders record."""

    converted_keys: int
    reused_keys: int
    source_bytes_read: int
    new_bytes_written: int
    worst_relative_frobenius: float | None


#: Half-ulp of each carrier's significand, as a multiple of the tensor's amax. bf16 keeps
#: 8 significand bits, fp16 keeps 11. Overflow is not covered by a relative bound and is
#: caught separately: fp32 reaches 3.4e38 and fp16 stops at 65504, so an fp16 cast of a
#: tensor with a large outlier produces inf, and that is a refusal, never a saturation.
CAST_ROUND_TRIP_BOUND: Mapping[str, float] = {"bf16": BF16_ROUND_TRIP_BOUND, "f16": 2.0**-11}
CAST_READ_CHUNK = 32 << 20
MAX_CAST_TENSOR_BYTES = 1 << 30


def write_cast(
    transaction: WeightsTransaction,
    ctx: Context,
    tel: Telemetry,
    *,
    selection: Selection,
    source: str,
    source_component: str,
    target_component: str,
) -> CastStats:
    """Stream one component's float32 cut through RNE into the declared carrier.

    Every converted tensor's round trip is measured against the carrier's representational
    bound and the worst relative Frobenius rides the returned stats — the same recorded
    derivation the BF16 ingest normalization performs, never a silent `.astype`. A cast
    that produced a non-finite value refuses: fp16's exponent range is three orders of
    magnitude narrower than fp32's and an overflowed weight is not a rounding error.
    """
    dtype = selection.treatment.cast
    if dtype is None or not selection.cast:
        return CastStats(0, 0, 0, 0, None)
    carrier = dtype.upper()
    bound = CAST_ROUND_TRIP_BOUND[dtype]
    completed = transaction.completed_parts
    read = written = reused = 0
    worst = 0.0
    total = len(selection.cast)
    for done, (key, shape) in enumerate(selection.cast, 1):
        ctx.raise_if_cancelled()
        stage = f"cast-{dtype}-{target_component}"
        if (target_component, key, "value") in completed:
            reused += 1
            tel.progress(done / total, stage=stage)
            continue
        values, length = _read_f32(
            transaction,
            source=source,
            component=source_component,
            key=key,
            shape=shape,
        )
        with np.errstate(over="ignore"):
            raw = st.from_f32(values, carrier)
        restored = st.to_f32(raw, carrier, list(shape))
        if not bool(np.isfinite(restored).all()):
            raise UnsupportedInput(
                f"{source_component}.{key}: the {dtype} cast overflowed to a non-finite "
                f"value (source amax {float(np.max(np.abs(values))):.6g})",
                code="quantization_tripwire",
            )
        error = np.abs(values - restored)
        allowed = float(np.max(np.abs(values))) * bound
        if float(np.max(error)) > allowed:
            raise UnsupportedInput(
                f"{source_component}.{key}: {dtype} round-trip error "
                f"{float(np.max(error)):.4g} exceeds the representational bound {allowed:.4g}",
                code="quantization_tripwire",
            )
        norm = float(np.linalg.norm(values))
        relative = float(np.linalg.norm(error)) / norm if norm else 0.0
        worst = max(worst, relative)
        transaction.add_part(target_component, key, "value", raw)
        transaction.checkpoint()
        read += length
        written += len(raw)
        tel.log(
            "cast tensor",
            level="info",
            dtype=dtype,
            component=target_component,
            key=key,
            relative_frobenius=relative,
            done=done,
            total=total,
        )
        tel.progress(done / total, stage=stage)
        del values, raw, restored, error
    converted = total - reused
    return CastStats(converted, reused, read, written, worst if converted else None)


def _read_f32(
    transaction: WeightsTransaction,
    *,
    source: str,
    component: str,
    key: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    length = math.prod(shape) * 4
    if length > MAX_CAST_TENSOR_BYTES:
        raise UnsupportedInput(
            f"source {component}.{key} is {length} bytes; the per-tensor cast bound is "
            f"{MAX_CAST_TENSOR_BYTES}",
            code="h3_cast_tensor_bound",
        )
    raw = bytearray(length)
    view = memoryview(raw)
    for offset in range(0, length, CAST_READ_CHUNK):
        transaction.source_read_into(
            source,
            component,
            key,
            "value",
            offset,
            view[offset : min(offset + CAST_READ_CHUNK, length)],
        )
    values = st.to_f32(raw, "F32", list(shape))
    if not bool(np.isfinite(values).all()):
        raise UnsupportedInput(
            f"source {component}.{key} contains non-finite weights",
            code="h3_cast_source_nonfinite",
        )
    return values, length
