"""Project and retain encoded image bytes with their observed media type."""

from __future__ import annotations

from pathlib import Path

from cozy_eval.errors import DataError
from cozy_runtime.author import FileAsset, ImageAsset, Outputs


def image_suffix(media_type: str) -> str:
    if media_type == "image/webp":
        return ".webp"
    if media_type == "image/png":
        return ".png"
    raise DataError(f"unsupported assessment image media type: {media_type}")


def project_image(image: ImageAsset, out: Outputs) -> Path:
    path = out.temporary_file(image_suffix(image.media_type))
    path.write_bytes(image.read_bytes())
    return path


async def retain_image(image: ImageAsset, out: Outputs) -> FileAsset:
    image_suffix(image.media_type)
    return await out.commit(out.save_bytes(image.read_bytes(), media_type=image.media_type))
