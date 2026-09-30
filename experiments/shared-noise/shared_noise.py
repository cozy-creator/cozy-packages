"""Private verified raw-noise input; no default RNG changes or weight-store claims."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ARTIFACTS = Path('/home/fidika/.cozy/outputs/comfy-cozy-memory-20260929/analysis-shared-initial-noise/artifacts')
MANIFEST_SHA256 = '9dea16ec10e3582af052bde91336f9e8c8a6ab62d64c7444ea6336f8ecee4875'


def raw_noise(model: str, seed: int, shape: tuple[int, ...], *, device: Any, dtype: Any) -> Any:
    import torch
    if sys.byteorder != 'little':
        raise RuntimeError('Private noise artifact requires little-endian tensor representation')
    manifest = (ARTIFACTS / 'noise-manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError('Private noise manifest changed')
    rows = [r for r in json.loads(manifest)['rows'] if r['model'] == model and r['seed'] == seed]
    if len(rows) != 1 or rows[0]['shape'] != list(shape):
        raise ValueError('Unsupported private noise request; no random fallback')
    row = rows[0]
    if row['dtype'] != 'float32' or row['byteorder'] != 'little':
        raise ValueError('Unsupported private noise representation')
    data = (ARTIFACTS / row['file']).read_bytes()
    if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise RuntimeError('Private noise payload changed')
    # Writable owner prevents a non-writable-buffer warning; no global RNG is touched.
    source = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shape)
    return source.to(device=device, dtype=dtype)


def comfy_noise(latent: Any, seed: int, batch_inds: Any = None) -> Any:
    if batch_inds is not None:
        raise ValueError('Private noise experiment excludes indexed batches')
    shape = tuple(latent.shape)
    if shape == (1, 16, 128, 128):
        # Explicit single-frame NCTHW→NCHW boundary, never a numel-based reshape.
        noise = raw_noise('anima', seed, (1, 16, 1, 128, 128), device='cpu', dtype=latent.dtype)
        return noise.squeeze(2)
    model = {(1, 4, 128, 128): 'sdxl', (1, 16, 1, 128, 128): 'anima'}.get(shape)
    if model is None:
        raise ValueError('Unsupported private latent geometry')
    return raw_noise(model, seed, shape, device='cpu', dtype=latent.dtype)
