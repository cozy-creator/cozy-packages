"""Private calibration leaves, never part of the deployed SDXL application."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import msgspec
import numpy as np
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    MediaDecoder,
    ModelArtifact,
    Outputs,
    Telemetry,
    UnsupportedInput,
    WeightsConfig,
    WeightsOutput,
    WeightsPart,
    WeightsSink,
    WeightsTarget,
    WeightsTensor,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource
from PIL import Image, ImageFilter

PLAIN = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
MAX_PART_BYTES = 64 << 20
MAX_NEW_BYTES = 16 << 30
CHUNK_BYTES = 4 << 20
BLUR_RADIUS = 8.0
FLAT_RGB = (127, 127, 127)


class MediaControl(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    image: ImageAsset


@invocable(memoize=True)
async def scale_unet_x2(
    ctx: Context, *, source: QuantizationSource, weights: WeightsSink, tel: Telemetry
) -> ModelArtifact:
    """Deliberately double normalized plain F16 UNet values for a negative control.

    This is a real derivation with native source custody and resumable output parts.
    Its writer reads are used to produce changed weights, never to bypass measurement
    access. Other components, configs and construction order remain inherited exactly.
    """
    structure = weights.structure(source)
    components = tuple(dict.fromkeys(tensor.component for tensor in structure.tensors))
    if set(components) != {"unet", "text_encoder", "text_encoder_2", "vae"}:
        raise UnsupportedInput("the control requires the normalized SDXL component roster")
    selected = tuple(tensor for tensor in structure.tensors if tensor.component == "unet")
    additions = {}
    total = 0
    for tensor in selected:
        size = math.prod(tensor.shape) * 2
        if (
            tensor.logical_dtype != "f16"
            or tensor.encoding != PLAIN
            or len(tensor.parts) != 1
            or tensor.parts[0].name != "value"
            or tensor.parts[0].dtype != "f16"
            or tensor.parts[0].shape != tensor.shape
            or not 0 < size <= MAX_PART_BYTES
        ):
            raise UnsupportedInput("x2 control requires bounded normalized plain F16 UNet parts")
        total += size
        additions[tensor.key] = WeightsTensor(
            "f16", tensor.shape, PLAIN, {"value": WeightsPart("f16", tensor.shape)}
        )
    if not additions or total > MAX_NEW_BYTES or not structure.configs:
        raise UnsupportedInput("x2 control has no UNet, no config or exceeds its output budget")
    targets = {
        name: WeightsTarget(source="source", source_component=name) for name in components
    }
    targets["unet"] = WeightsTarget(
        source="source", source_component="unet", drop=tuple(additions), add=additions
    )
    with weights.open(
        "model",
        sources={"source": source},
        targets=targets,
        configs={
            name: WeightsConfig(source="source", source_config=name) for name in structure.configs
        },
        order=tuple((tensor.component, tensor.key) for tensor in structure.tensors),
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return receipt.artifact
        completed = transaction.completed_parts
        for index, tensor in enumerate(selected):
            ctx.raise_if_cancelled()
            if ("unet", tensor.key, "value") in completed:
                continue
            raw = bytearray(math.prod(tensor.shape) * 2)
            for offset in range(0, len(raw), CHUNK_BYTES):
                ctx.raise_if_cancelled()
                transaction.source_read_into(
                    "source", "unet", tensor.key, "value", offset,
                    memoryview(raw)[offset : offset + CHUNK_BYTES],
                )
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            values *= 2.0
            if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float16).max):
                raise UnsupportedInput("x2 control refuses nonfinite or overflowing F16 values")
            # Finish the source read before native add_part; no callback re-enters a writer.
            changed = values.astype("<f2").tobytes()
            transaction.add_part("unet", tensor.key, "value", changed)
            transaction.checkpoint()
            tel.progress((index + 1) / len(selected), stage="calibration-unet-x2")
        return transaction.commit().artifact


@invocable(memoize=True)
async def corrupt_media(
    ctx: Context,
    *,
    image: Annotated[ImageAsset, AssetBound(max_bytes=16 << 20, max_decoded_bytes=16 << 20)],
    kind: Literal["blur", "flat"],
    decoder: MediaDecoder,
    out: Outputs,
) -> MediaControl:
    """Produce declared pixel controls from a real retained reference image."""
    ctx.raise_if_cancelled()
    frame = decoder.decode_image(image)
    source = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
    if kind == "blur":
        changed = source.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    elif kind == "flat":
        changed = Image.new("RGB", source.size, FLAT_RGB)
    else:
        raise UnsupportedInput("unknown media corruption control")
    ctx.raise_if_cancelled()
    return MediaControl(out.save_image(ImageFrame(frame.width, frame.height, changed.tobytes())))


app = App()
app.job(scale_unet_x2, weights=(WeightsOutput("model", max_new_bytes=MAX_NEW_BYTES),))
app.job(corrupt_media)
