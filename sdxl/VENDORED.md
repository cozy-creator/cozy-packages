# SDXL dependency closure

This directory has one Python dependency authority: `pyproject.toml` plus `uv.lock`.
Cozy-owned libraries are immutable local wheels; third-party packages are exact lock
entries. Before se-019 this endpoint had neither file, so it could not produce a
`ResolvedWheelSet` and could not publish at all.

## Cozy-owned wheels

Byte-identical copies of the wheels H3 vendors — the same Runtime build serves both
endpoints, so the CAS stores one copy.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.3-py3-none-any.whl` | `cozy-runtime` `13ebb22c7dfbfecef18b09806f604f1fc8eee027` | 679,123 | `6ab9228a9dbd7a10a16215576bef06385de23874dbfe797db02f8ee734e7284b` |
| `vendor/tensorfs-0.0.1-cp311-abi3-manylinux_2_34_x86_64.whl` | `tensorfs` `4f3d16dff35d195f3709f06b6dc77f95bddfe64b` | 1,070,730 | `0dc98c4a81a7d7e5dd2b9009bb2035b838f1d60d34720d9121f69a2fc46c8ce9` |

TensorFS is the Runtime fill reader inside the isolated endpoint generation. Cozy-eval is
NOT here: `sdxl.py` imports nothing from it, and `scripts/sdxl-live.py` is a driver that
runs outside this environment.

## Third-party binary closure

The model stack is pinned to:

- `diffusers==0.40.0`
- `transformers==5.16.1`
- `torch==2.9.1+cu129`
- `torchvision==0.24.1+cu129`

**The torch family is exact, not a range.** These pins are H3's, to the patch and to the
local version: `torch 2.9.1+cu129`, `torchvision 0.24.1+cu129`, `triton 3.5.1` and the
fifteen `nvidia-*-cu12` wheels resolve to byte-identical versions in both locks.
Torchvision's compiled ops link against torch's C++ ABI, so an endpoint built against one
torch raises undefined-symbol errors under another; decision #633 gives one PlatformTarget
exactly one torch family, which makes a second family a second substrate variant rather
than a per-endpoint choice. No torchaudio: this endpoint decodes and emits no audio.

`uv lock` resolves 65 packages for Linux x86-64 on Python 3.12. A new environment installed
all 64 non-project packages from the populated uv cache with no network access:

```sh
cd sdxl
uv lock --check

offline_env=$(mktemp -d /tmp/sdxl-lock-offline.XXXXXX)
rmdir "$offline_env"
UV_PROJECT_ENVIRONMENT="$offline_env" \
  uv sync --locked --offline --no-install-project
```

That 7.6 GiB environment then imported the real stack and the endpoint module:
`torch 2.9.1+cu129`, `torchvision 0.24.1+cu129` (including `torchvision.ops`, the compiled
extension that is the actual ABI test), `triton 3.5.1`, and `sdxl:app` with its one
`generate` entrypoint. A fresh builder must still populate its third-party wheel cache or
mirror before entering offline mode.

The Runtime media extra resolves to the same PyAV wheel H3 locks, `av==18.1.0`.

## The publish receipt

`th044-wheelset` (tensorhub `2a5bfa8`) resolved this project against the fleet
PlatformTarget `linux/amd64 · glibc2.34 · cp312 · cuda · cu129`, acquiring every candidate
from the exact `uv.lock` URLs and digests:

```
th044-wheelset --project sdxl --uv $(command -v uv) \
  --os-arch linux/amd64 --libc glibc2.34 --python-abi cp312 \
  --accelerator-backend cuda --accelerator-abi cu129 --acquire --output <cas>
```

| document | value |
| --- | --- |
| `ResolvedWheelSet` | `sha256:6f4746d8015c23b79030718ecb2584fd874d9e8db02c781538560395e2fe4263` (17,296 B) |
| `WheelhouseManifest` | `sha256:30c2f136c9c7bbcd0d5d4a38377e1f1c71abc170c795f2e3d4f7c5f093eddb85` (184 B) |
| wheels | 64 |
| wheel bytes | 4,685,825,279 |

`--verify-output <cas> --resolved-digest sha256:6f4746d8…` replayed the same documents from
the published CAS with no uv and no source wheels. This endpoint could not produce any of
this before se-019: `wheelset.Resolve` reads `<project>/pyproject.toml` and
`<project>/uv.lock`, and neither existed.

The wheelhouse manifest is th-044's shipped EMPTY base wheelhouse, so the torch closure
rides this set — the state decision #633 reverses once th-062 lands the substrate variant
matrix. When it does, every endpoint here re-locks against the fleet's family together;
that is a lock change, not a per-endpoint torch choice.
