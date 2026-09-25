# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.18.23,<1"]
# ///
"""Optional composition example for complete native model ingestion and upload.

For ordinary preparation, use `cozy model upload <source> <destination>`.
To compose the same operations in a private script:

    cozy run examples/client-scripts/prepare_reference_image.py \
      --rental=YOUR_RENTAL --allow-publish paul/reference-image --await

Runtime owns the reviewed tensor census, configuration, tokenizer and notices.
Upload returns a checkpoint; release publication stays disabled unless explicitly
selected below. Discovery with --describe does not execute these operations.
"""

from cozy_runtime.author.publication import publish_release, upload_checkpoint
from cozy_runtime.author.sources import ingest_huggingface

REPOSITORY = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DESTINATION = "paul/reference-image"
PUBLISH_RELEASE = False
RELEASE = "0.1.0"


async def main() -> dict[str, str]:
    model = await ingest_huggingface(REPOSITORY, revision=REVISION)
    checkpoint = await upload_checkpoint(model, destination=DESTINATION)
    result = {"destination": checkpoint.destination, "checkpoint": checkpoint.checkpoint}
    if PUBLISH_RELEASE:
        release = await publish_release(
            destination=DESTINATION,
            release=RELEASE,
            lanes={"original": checkpoint},
        )
        result["release"] = release.release
    return result
