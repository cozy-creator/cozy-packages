# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.8-py3-none-any.whl` | `cozy-runtime` `01cb96af87b9c71e09bcb2df46f708dcf18705ef` | 709,911 | `cfa8ceafb7a3026f674ba7bf8e7b0d0a6f7740f2e4ec54de71bd2c34f2339708` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
