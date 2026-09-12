"""One native tree result for a variable set of retained assessment files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from cozy_eval import contract
from cozy_eval.errors import DataError
from cozy_runtime.author import FileAsset, Outputs, Tree

from .image_files import image_suffix


def file_suffix(media_type: str) -> str:
    return ".json" if media_type == "application/json" else image_suffix(media_type)


def report_bundle(files: Mapping[str, FileAsset], out: Outputs) -> Tree:
    """Save exact encoded files under one declared output; include an identity index."""
    entries = []
    with TemporaryDirectory(prefix="sdxl-report-") as temporary:
        directory = Path(temporary)
        for name, file in files.items():
            if Path(name).name != name or name in ("", ".", "..", "manifest.json"):
                raise DataError("report bundle members need distinct ordinary basenames")
            if not name.endswith(file_suffix(file.media_type)):
                raise DataError("report bundle filename differs from the file media type")
            (directory / name).write_bytes(file.read_bytes())
            entries.append({"path": name, "digest": file.digest, "media_type": file.media_type,
                            "size_bytes": file.size_bytes})
        (directory / "manifest.json").write_bytes(contract.canonical({
            "schema": "sdxl-report-bundle@1", "files": entries,
        }))
        # Outputs snapshots the regular files now, before this temporary directory closes.
        return out.save_tree(directory)
