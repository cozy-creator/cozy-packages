"""Private calibration leaves, never part of the deployed SDXL application."""

from __future__ import annotations

import io
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
    WeightsOutput,
    invocable,
)
from cozy_runtime.derive.quantization import QuantizationSource
from PIL import Image, ImageFilter
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

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
    ctx: Context, *, source: QuantizationSource, tel: Telemetry
) -> ModelArtifact:
    """Deliberately double normalized plain F16 UNet values for a negative control.

    This is a real derivation with native source custody and resumable output parts.
    Its writer reads are used to produce changed weights, never to bypass measurement
    access. Other components, configs and construction order remain inherited exactly.
    """
    with ctx.tensorfs_source(source) as capability:
        structure = capability.inspect()
    components = tuple(structure.components)
    if set(components) != {"unet", "text_encoder", "text_encoder_2", "vae"}:
        raise UnsupportedInput("the control requires the normalized SDXL component roster")
    selected = tuple(structure.components["unet"].items())
    additions = {}
    total = 0
    for key, tensor in selected:
        size = math.prod(tensor.shape) * 2
        if (
            tensor.logical_dtype != "f16"
            or tensor.encoding != PLAIN
            or set(tensor.parts) != {"value"}
            or tensor.parts["value"].dtype != "f16"
            or tensor.parts["value"].shape != tensor.shape
            or not 0 < size <= MAX_PART_BYTES
        ):
            raise UnsupportedInput("x2 control requires bounded normalized plain F16 UNet parts")
        total += size
        additions[key] = Tensor("f16", tensor.shape, PLAIN, {"value": Part("f16", tensor.shape)})
    if not additions or total > MAX_NEW_BYTES or not structure.configs:
        raise UnsupportedInput("x2 control has no UNet, no config or exceeds its output budget")
    targets = {name: Target(source="source", source_component=name) for name in components}
    targets["unet"] = Target(
        source="source", source_component="unet", drop=tuple(additions), add=additions
    )
    with ctx.output("model").open(
        Derivation(
            sources={"source": structure.source},
            targets=targets,
            configs={
                name: Config("copy", source="source", source_config=name)
                for name in structure.configs
            },
            order=tuple(
                (component, key) for component, rows in structure.components.items() for key in rows
            ),
        )
    ) as transaction:
        if transaction.receipt is not None:
            receipt = transaction.receipt
            assert receipt is not None
            return ctx.adopt_model(receipt)
        completed = frozenset(transaction.completed_parts())
        for index, (key, tensor) in enumerate(selected):
            ctx.raise_if_cancelled()
            if ("unet", key, "value") in completed:
                continue
            raw = bytearray(math.prod(tensor.shape) * 2)
            for offset in range(0, len(raw), CHUNK_BYTES):
                ctx.raise_if_cancelled()
                transaction.source_read_into(
                    "source",
                    "unet",
                    key,
                    "value",
                    offset,
                    memoryview(raw)[offset : offset + CHUNK_BYTES],
                )
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            values *= 2.0
            if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float16).max):
                raise UnsupportedInput("x2 control refuses nonfinite or overflowing F16 values")
            # Finish the source read before native add_part; no callback re-enters a writer.
            changed = values.astype("<f2").tobytes()
            transaction.add_part("unet", key, "value", io.BytesIO(changed))
            transaction.checkpoint()
            tel.progress((index + 1) / len(selected), stage="calibration-unet-x2")
        return ctx.adopt_model(transaction.commit())


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
