# SDXL dependency closure

This directory has one Python dependency authority: `pyproject.toml` plus `uv.lock`.
Cozy-owned libraries are immutable local wheels; third-party packages are exact lock
entries. Before se-019 this package had neither file, so it could not produce a
`ResolvedWheelSet` and could not publish at all.

## Cozy-owned wheels

Byte-identical copies of the wheels H3 vendors — the same Runtime build serves both
packages, so the CAS stores one copy.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.3-py3-none-any.whl` | `cozy-runtime` `a675e1085300d3b21e56f4787be630853c2e8393` | 710,824 | `1816d1b73ca8b0886afd0d511cb949f83dd64d2444b2c53645a77a650e742ef4` |
| `vendor/tensorfs-0.0.1-cp311-abi3-manylinux_2_34_x86_64.whl` | `tensorfs` `4f3d16dff35d195f3709f06b6dc77f95bddfe64b` | 1,070,730 | `0dc98c4a81a7d7e5dd2b9009bb2035b838f1d60d34720d9121f69a2fc46c8ce9` |

TensorFS is the Runtime fill reader inside the isolated package generation. Cozy-eval is
NOT here: `sdxl.py` imports nothing from it; evaluation remains a caller-side concern.

## Third-party binary closure

The model stack is pinned to:

- `diffusers==0.40.0`
- `transformers==5.16.1`
- `torch==2.9.1+cu129`
- `torchvision==0.24.1+cu129`

**The torch family is exact, not a range.** These pins are H3's, to the patch and to the
local version: `torch 2.9.1+cu129`, `torchvision 0.24.1+cu129`, `triton 3.5.1` and the
fifteen `nvidia-*-cu12` wheels resolve to byte-identical versions in both locks.
Torchvision's compiled ops link against torch's C++ ABI, so a package built against one
torch raises undefined-symbol errors under another; decision #633 gives one PlatformTarget
exactly one torch family, which makes a second family a second substrate variant rather
than a per-package choice. No torchaudio: this package decodes and emits no audio.

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

That 7.6 GiB environment then imported the real stack and the package module:
`torch 2.9.1+cu129`, `torchvision 0.24.1+cu129` (including `torchvision.ops`, the compiled
extension that is the actual ABI test), `triton 3.5.1`, and `sdxl:app` with its one
`generate` entrypoint. A fresh builder must still populate its third-party wheel cache or
mirror before entering offline mode.

The Runtime media extra resolves to the same PyAV wheel H3 locks, `av==18.1.0`.

## Publication

The package-domain hardcut changes the project wheel and every document above it. The retired
pre-hardcut receipt is intentionally not carried forward; Tensorhub must publish and qualify a
fresh package release from this exact lock and source tree.
