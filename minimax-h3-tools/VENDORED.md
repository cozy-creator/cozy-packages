# Vendored first-party wheels

Interim local wheels until the first-party libraries are on an index (se-022 relocks
against PyPI and deletes this directory). `uv.lock` independently pins the exact hash.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.1.0-py3-none-any.whl` | `cozy-runtime` `28fa78b24aa70d8d801218d069772671bab0fa12` | 816,385 | `eeb7b224e8a8d19b44ed877faddb801bccced1ea9485dc8b989e29525e922634` |
| `vendor/tensorfs-0.0.5-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `tensorfs` `0f49a4bf3fbe6fc8d41713b7ce9041c80161b7e6` | 2,337,278 | `a131c4688188dacd1518fbe93c417e64d6af575f65b52edac48bcb60f88d43a5` |

The Runtime wheel was built twice from the merged commit with `uv build --wheel` at the
platform epoch `SOURCE_DATE_EPOCH=946684800` on CPython 3.12; the outputs were byte-identical.
TensorFS 0.0.5 is a dev-group resolver anchor only (order-proof driver); the runtime
package declares no TensorFS dependency. torch resolves from PyPI so this repo keeps one
torch family (#633).
