"""The installable ``tensorhub/quantize`` package surface."""

import msgspec
from cozy_runtime.author import (
    App,
    Context,
    Telemetry,
    WeightsOutput,
    WeightsReceipt,
    WeightsSink,
    WeightsTarget,
    WeightsTransaction,
)
from cozy_runtime.derive.quantization import (
    MAX_OUTPUT_BYTES,
    ArtifactQuantizationPlan,
    ArtifactQuantizationRequest,
    ArtifactQuantizationResult,
    Bf16Plan,
    QuantizationSource,
    bf16_targets,
    derive_fp8,
    derive_mxfp8,
    inherited_configs,
    prepare_bf16,
    prepare_source_quantization,
    quantization_additions,
    quantize_component_into,
    write_bf16,
)

from sdxl import SdxlModel

app = App()

SDXL_OUTPUT_BYTES = 16 << 30


class SdxlThreeLaneResult(msgspec.Struct):
    bf16_receipt_digest: str
    fp8_receipt_digest: str
    mxfp8_receipt_digest: str
    replayed_outputs: int
    source_bytes_read_this_run: int
    encoded_keys_this_run: int


def _receipt(transaction: WeightsTransaction) -> WeightsReceipt:
    value = transaction.receipt if transaction.replayed else transaction.commit()
    assert value is not None
    return value


def _sdxl_targets(
    bf16: Bf16Plan,
    *,
    encoding: str | None,
    quantization: ArtifactQuantizationPlan,
) -> dict[str, WeightsTarget]:
    if encoding is None:
        return bf16_targets(bf16, source="source")
    selected = {(tensor.component, tensor.key) for tensor in quantization.tensors}
    targets = bf16_targets(bf16, source="source", excluded=selected)
    additions = quantization_additions(
        encoding,
        quantization,
        "unet",
        target_logical_dtype="bf16",
    )
    current = targets["unet"]
    combined = {**current.add, **additions}
    targets["unet"] = WeightsTarget(
        source="source",
        source_component="unet",
        drop=tuple(sorted(combined)),
        add=combined,
    )
    return targets


@app.job(
    name="sdxl-three-lane",
    weights=(
        WeightsOutput("bf16", max_new_bytes=SDXL_OUTPUT_BYTES),
        WeightsOutput("fp8", max_new_bytes=SDXL_OUTPUT_BYTES),
        WeightsOutput("mxfp8", max_new_bytes=SDXL_OUTPUT_BYTES),
    ),
    # Four segments, no th-114 config selector: SdxlModel constructs from the
    # runtime-assembled mapping of every named config, and evidence identity
    # must equal serving identity (cr-073; the selector retires with cr-077).
    source_profiles={"source": "paul/sdxl/1.0.0/bf16"},
)
def sdxl_three_lane(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: SdxlModel,
    weights: WeightsSink,
    tel: Telemetry,
) -> SdxlThreeLaneResult:
    """Normalize one reviewed SDXL source to BF16 and derive both row-wise lanes."""
    structure = weights.structure(source)
    bf16 = prepare_bf16(structure)
    quantization = prepare_source_quantization(structure, components=("unet",))
    selected = {(tensor.component, tensor.key) for tensor in quantization.tensors}
    receipts: dict[str, WeightsReceipt] = {}
    source_bytes = 0
    encoded = 0

    for output, encoding in (
        ("bf16", None),
        ("fp8", "fp8-rowwise/1"),
        ("mxfp8", "mxfp8/1"),
    ):
        with weights.open(
            output,
            sources={"source": source},
            targets=_sdxl_targets(
                bf16,
                encoding=encoding,
                quantization=quantization,
            ),
            configs=inherited_configs(bf16, source="source"),
            order=bf16.order,
        ) as transaction:
            if transaction.replayed:
                receipts[output] = _receipt(transaction)
                continue
            source_bytes += write_bf16(
                transaction,
                ctx,
                tel,
                plan=bf16,
                source="source",
                excluded=selected if encoding is not None else None,
            )
            if encoding is not None:
                stats = quantize_component_into(
                    transaction,
                    ctx,
                    payload,
                    tel,
                    encoding=encoding,
                    plan=quantization,
                    component="unet",
                    source="source",
                    source_component="unet",
                    target_component="unet",
                )
                source_bytes += stats.source_bytes_read
                encoded += stats.encoded_keys
            receipts[output] = transaction.commit()

    return SdxlThreeLaneResult(
        receipts["bf16"].tensorfs_receipt_digest,
        receipts["fp8"].tensorfs_receipt_digest,
        receipts["mxfp8"].tensorfs_receipt_digest,
        sum(receipt.replayed for receipt in receipts.values()),
        source_bytes,
        encoded,
    )


@app.job(
    weights=(WeightsOutput("model", max_new_bytes=MAX_OUTPUT_BYTES),),
)
def fp8(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> ArtifactQuantizationResult:
    """Derive row-wise FP8 directly from the exact source Manifest."""
    return derive_fp8(ctx, payload, source, artifacts, tel)


@app.job(
    weights=(WeightsOutput("model", max_new_bytes=MAX_OUTPUT_BYTES),),
)
def mxfp8(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> ArtifactQuantizationResult:
    """Derive MXFP8 directly from the exact source Manifest."""
    return derive_mxfp8(ctx, payload, source, artifacts, tel)
