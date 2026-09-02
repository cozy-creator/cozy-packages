# Vendored first-party wheels

Interim local wheels until the first-party libraries are on an index (se-022 relocks
against PyPI and deletes this directory). `uv.lock` independently pins the exact hash.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.1.0-py3-none-any.whl` | `cozy-runtime` `28fa78b24aa70d8d801218d069772671bab0fa12` | 816,385 | `eeb7b224e8a8d19b44ed877faddb801bccced1ea9485dc8b989e29525e922634` |

The Runtime wheel was built twice from the merged commit with `uv build --wheel` at the
platform epoch `SOURCE_DATE_EPOCH=946684800` on CPython 3.12; the outputs were byte-identical.
