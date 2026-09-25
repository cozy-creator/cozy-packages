# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cozy-runtime>=0.18.21,<1", "tensorfs>=0.3.51,<0.4",
#   "reference-preparation>=0.1.0",
# ]
# [tool.uv.sources]
# reference-preparation = {path = "./reference-preparation"}
# ///
"""Prepare and publish the research checkpoint through native library operations.

    cozy run ./prepare_reference_image.py --rental=<machine> \
      --allow-publish paul/reference-image --await

The adjacent helper is captured as a local dependency, never published. Its
registered transformation accepts the conversion receipt as an injected Model.
Each expensive operation reuses retained work independently of this script.
"""

from cozy_runtime.author import ScriptContext
from cozy_runtime.author.publication import publish_release, upload_checkpoint
from cozy_runtime.author.sources import convert_cozytensors, download_huggingface, source_files
from reference_preparation import FILES, REPOSITORY, REVISION, prepare

DESTINATION = "paul/reference-image"


async def main(ctx: ScriptContext) -> dict[str, str]:
    ctx.log("Downloading pinned Qwen model components")
    source = await download_huggingface(
        REPOSITORY,
        revision=REVISION,
        carriers=(
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "text_encoder/model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
    original = await convert_cozytensors(source, profile="hf/qwen/qwen-image-2.1/original/1")
    metadata_source = await download_huggingface(REPOSITORY, revision=REVISION, files=tuple(FILES))
    metadata = await source_files(metadata_source)
    prepared = await prepare(source=original, metadata=metadata)
    checkpoint = await upload_checkpoint(prepared, destination=DESTINATION)
    release = await publish_release(
        destination=DESTINATION,
        release="0.1.0",
        lanes={"original": checkpoint},
    )
    ctx.log(f"Published {DESTINATION}@{release.release}/original")
    return {"checkpoint": checkpoint.checkpoint, "release": release.release}
