# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.3-py3-none-any.whl` | `cozy-runtime` `13ebb22c7dfbfecef18b09806f604f1fc8eee027` | 679,123 | `6ab9228a9dbd7a10a16215576bef06385de23874dbfe797db02f8ee734e7284b` |

Two independent builds from that merged Runtime commit were byte-identical. The `media` extra
admits the exact locked PyAV/FFmpeg wheel; this endpoint owns no decoder, packet, path, or muxer.
