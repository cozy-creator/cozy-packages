# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.9-py3-none-any.whl` | `cozy-runtime` `64c82a489c3a297aa4d779133480b0d7a7e0c5a6` | 693,563 | `511ead4219f41e876c9308a77d83e09b744522624fb3893db3c5a80a21a3cddb` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
