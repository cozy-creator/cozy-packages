#!/usr/bin/env python
"""Pack an endpoint tree into a release archive — the pre-hub stand-in for `cozy deploy`.

cl-012 landed the real publish path (manifest from `git ls-files`, declare-digests-first,
presigned PUTs) on the hub side; locally an install still verifies an archive, so this
writes exactly the shape `cozy install --from --digest` checks: `release.json` FIRST, then
exactly the files it declares.

    scripts/pack.py <tree> <org/endpoint> <version> <out.tar.gz>

Prints the archive's sha256 — what `cozy install --digest` checks.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pathlib
import sys
import tarfile

SKIP = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules", "dist"}


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes, mode: int) -> None:
    """Add one normalized regular file.

    The release digest identifies content, not the builder's clock, uid, umask, or host.
    """
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    tree, endpoint, version, out = argv[1:]
    root = pathlib.Path(tree).resolve()

    files: list[dict[str, object]] = []
    for p in sorted(root.rglob("*")):
        if any(part in SKIP for part in p.relative_to(root).parts):
            continue
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        files.append({"path": rel, "sha256": digest(p), "size": p.stat().st_size})

    declaration = (
        json.dumps({"endpoint": endpoint, "version": version, "files": files}, indent=2)
        + "\n"
    ).encode()

    # `tarfile.open(..., "w:gz")` embeds the current time in the gzip header, while
    # `tar.add` preserves host uid/gid/mtime. Normalize both layers so identical input
    # bytes always produce one archive digest.
    with (
        open(out, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        add_bytes(tar, "release.json", declaration, 0o644)
        for f in files:
            path = root / str(f["path"])
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            add_bytes(tar, str(f["path"]), path.read_bytes(), mode)

    print(f"archive:  {out}")
    print(f"endpoint: {endpoint}  version: {version}  files: {len(files)}")
    print(f"digest:   sha256:{digest(pathlib.Path(out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
